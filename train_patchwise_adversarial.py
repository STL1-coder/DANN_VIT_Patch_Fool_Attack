import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import network
import loss
from data_list import ImageList
import pre_process as prep
from timm.models.vision_transformer import PatchEmbed, Block
import lr_schedule

def patch_wise_pgd_attack(model, images, labels, eps=2/255, alpha=1/255, iters=20, patch_size=16, num_patches_to_perturb=1):
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size

    with torch.no_grad():
        _, _, attn_weights = model(images, return_attention=True)
        attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)  # [batch, seq_len]
        attn_to_patches = attn[:, 0, 1:]  # [batch, num_patches]
        _, patch_indices = attn_to_patches.topk(k=num_patches_to_perturb, dim=1)  # [batch, 1]
        patch_indices = patch_indices.cpu().numpy()

    for _ in range(iters):
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        grad = images.grad.data
        perturbation = torch.zeros_like(images)

        for i in range(batch_size):
            idx = patch_indices[i][0]  # Single highest-attention patch
            px = (idx % num_patches_x) * patch_size
            py = (idx // num_patches_x) * patch_size
            patch_grad = grad[i:i+1, :, py:py+patch_size, px:px+patch_size]
            perturbation[i:i+1, :, py:py+patch_size, px:px+patch_size] = alpha * patch_grad.sign()

        adv_images = images + perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
        images.requires_grad = True
        images.grad = None

    return images

def pgd_robustness_test(loader, model, eps=2/255, alpha=2/255, iters=20):
    correct_clean = 0
    correct_adv = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):
        test_loader = test_loader[0]

    # Clean accuracy
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data[0].cuda(), data[1].cuda()  # Ignore pseudo-labels
            _, outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct_clean += (pred == data[1].cuda()).sum().item()
            total += data[1].size(0)

    # Adversarial accuracy using model predictions
    for data in test_loader:
        inputs, _ = data[0].cuda(), data[1].cuda()  # Ignore pseudo-labels
        with torch.no_grad():
            _, outputs = model(inputs)
            pred_labels = torch.max(outputs, 1)[1]  # Use model predictions
        adv_inputs = patch_wise_pgd_attack(model, inputs, pred_labels, eps=eps, alpha=alpha, iters=iters, patch_size=16, num_patches_to_perturb=1)
        with torch.no_grad():
            _, adv_outputs = model(adv_inputs)
            _, adv_pred = torch.max(adv_outputs, 1)
            correct_adv += (adv_pred == data[1].cuda()).sum().item()

    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Adversarial Accuracy: {adv_acc:.2f}%")
    return clean_acc, adv_acc

def train(config, model_path, pseudo_label_path):
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**prep_config['params'])
    prep_dict["target"] = prep.image_train(**prep_config['params'])
    prep_dict["test"] = prep.image_test(**prep_config['params']) if not prep_config["test_10crop"] else prep.image_test_10crop(**prep_config['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    dsets["target"] = ImageList(open(pseudo_label_path).readlines(), transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    
    if prep_config["test_10crop"]:
        dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"][i]) for i in range(10)]
        dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=2) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=2)

    class_num = config["network"]["params"]["class_num"]

    # Initialize a fresh ViT model without loading a checkpoint
    base_network = config["network"]["name"](**config["network"]["params"]).cuda()
    model = nn.Sequential(base_network).cuda()

    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"]).cuda()
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024).cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024).cuda()

    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **optimizer_config["optim_params"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        print(f"Iteration: {i:05d}")  # Print every iteration
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = pgd_robustness_test(dset_loaders, base_network)
            temp_model = nn.Sequential(base_network)
            if temp_acc[0] > best_acc:
                best_acc = temp_acc[0]
                best_model = temp_model
            log_str = f"iter: {i:05d}, clean acc: {temp_acc[0]:.2f}%, adv acc: {temp_acc[1]:.2f}%"
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], f"iter_{i:05d}_model.pth.tar"))

        loss_params = config["loss"]
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        inputs_source, inputs_target, labels_source, labels_target = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda(), labels_target.cuda()

        if torch.isnan(inputs_source).any() or torch.isnan(inputs_target).any():
            print(f"Iter {i}: NaN detected in input images, skipping batch")
            continue

        adv_inputs_target = patch_wise_pgd_attack(base_network, inputs_target, labels_target, eps=2/255, alpha=1/255, iters=10, patch_size=16, num_patches_to_perturb=1)

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(adv_inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        if config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
            ad_out = ad_net(features)
            ad_out = torch.clamp(ad_out, 1e-6, 1-1e-6)
        else:
            raise ValueError('Method not supported.')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)  # Only source loss
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(base_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), max_norm=1.0)

        optimizer.step()

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch-Wise Adversarial Domain Adaptation with ViT')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN'])
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224', choices=["vit_small_patch16_224"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-home'])
    parser.add_argument('--s_dset_path', type=str, default='data/office-home/Clipart.txt', help="Source dataset path")
    parser.add_argument('--t_dset_path', type=str, default='data/office-home/Art.txt', help="Target dataset path")
    parser.add_argument('--pseudo_label_path', type=str, default='snapshot/vit_pgd_test_run/pseudo_labels/pseudo_labels.txt', help="Path to pseudo-labels")
    parser.add_argument('--model_path', type=str, default='snapshot/vit_pgd_test_run/best_model.pth.tar', help="Path to pretrained model (not used)")
    parser.add_argument('--test_interval', type=int, default=2000, help="Interval for testing")
    parser.add_argument('--snapshot_interval', type=int, default=2000, help="Interval for saving model")
    parser.add_argument('--output_dir', type=str, default='vit_patchwise_adversarial_run', help="Output directory")
    parser.add_argument('--lr', type=float, default='0.0003', help="Learning rate")
    parser.add_argument('--random', type=bool, default=True, help="Whether to use random projection")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config = {
        'method': args.method,
        'gpu': args.gpu_id,
        'num_iterations': 40000,
        'test_interval': args.test_interval,
        'snapshot_interval': args.snapshot_interval,
        'output_path': "snapshot/" + args.output_dir,
        'prep': {"test_10crop": True, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}},
        'loss': {"trade_off": 1.0, "random": args.random, "random_dim": 256},
        'optimizer': {
            "type": optim.SGD,
            "optim_params": {'lr': float(args.lr), "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
            "lr_type": "inv",
            "lr_param": {"lr": float(args.lr), "gamma": 0.001, "power": 0.75}
        },
        'dataset': args.dset,
        'data': {
            "source": {"list_path": args.s_dset_path, "batch_size": 8},
            "target": {"list_path": args.pseudo_label_path, "batch_size": 8},
            "test": {"list_path": args.t_dset_path, "batch_size": 8}
        },
        'network': {
            "name": network.ViTFc,
            "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True, "class_num": 65}
        }
    }

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()

    print(f"Source dataset: {args.s_dset_path.split('/')[-1]} -> Target dataset: {args.t_dset_path.split('/')[-1]}")
    best_acc = train(config, args.model_path, args.pseudo_label_path)
    print(f"Best clean accuracy: {best_acc:.2f}%")
    config["out_file"].close()