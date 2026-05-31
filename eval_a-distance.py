import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import network
import pre_process as prep
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
import random
from data_list import ImageList
from itertools import zip_longest
from tqdm import tqdm

# Functions from second code with renamed conflicting functions
def patch_wise_pgd_attack_second(model, images, labels, eps=2/255, alpha=1/255, iters=250, patch_size=16, num_patches_to_perturb=1):
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size
    num_patches = num_patches_x * num_patches_y

    with torch.no_grad():
        _, _, attn_weights = model(images, return_attention=True)
        attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)
        attn_to_patches = attn[:, 0, 1:]
        patch_indices = attn_to_patches.argmax(dim=1).cpu().numpy()

    for _ in range(iters):
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        grad = images.grad.data
        perturbation = torch.zeros_like(images)

        for i in range(batch_size):
            idx = patch_indices[i]
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

def save_adv_image_with_box_second(image_tensor, patch_idx, image_id, save_dir, patch_size=16):
    img = to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(img)
    W, H = img.size
    num_patches_x = W // patch_size
    px = (patch_idx % num_patches_x) * patch_size
    py = (patch_idx // num_patches_x) * patch_size
    draw.rectangle([px, py, px + patch_size, py + patch_size], outline="red", width=2)
    img.save(os.path.join(save_dir, f"adv_img_{image_id}.png"))

def patch_fool_pgd_attack_second(model, images, labels, eps=2/255, alpha=1/255, iters=250,
                                 patch_size=16, num_patches_to_perturb=1,
                                 save_dir=None, global_step=0, save_count=0, max_save=5):
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, H, W = images.shape
    num_patches_x = W // patch_size
    num_patches_y = H // patch_size
    total_patches = num_patches_x * num_patches_y

    with torch.no_grad():
        _, logits = model(images)
        pred_labels = logits.argmax(dim=1)

    patch_indices = []
    for i in range(batch_size):
        max_drop = -float('inf')
        best_patch = 0
        base_img = ori_images[i:i+1].clone()
        _, clean_logits = model(base_img)
        clean_conf = clean_logits.softmax(dim=1)[0, pred_labels[i]].item()

        for p_idx in range(total_patches):
            px = (p_idx % num_patches_x) * patch_size
            py = (p_idx // num_patches_x) * patch_size
            masked_img = base_img.clone()
            masked_img[:, :, py:py+patch_size, px:px+patch_size] = 0.0

            _, logits_masked = model(masked_img)
            masked_conf = logits_masked.softmax(dim=1)[0, pred_labels[i]].item()
            drop = clean_conf - masked_conf
            if drop > max_drop:
                max_drop = drop
                best_patch = p_idx
        patch_indices.append(best_patch)

    for _ in range(iters):
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        grad = images.grad.data
        perturbation = torch.zeros_like(images)

        for i in range(batch_size):
            idx = patch_indices[i]
            px = (idx % num_patches_x) * patch_size
            py = (idx // num_patches_x) * patch_size
            patch_grad = grad[i:i+1, :, py:py+patch_size, px:px+patch_size]
            perturbation[i:i+1, :, py:py+patch_size, px:px+patch_size] = alpha * patch_grad.sign()

        adv_images = images + perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
        images.requires_grad = True
        images.grad = None

    if save_dir and save_count < max_save:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(batch_size):
            if save_count >= max_save:
                break
            save_adv_image_with_box_second(
                images[i], patch_indices[i],
                image_id=f"{global_step}_{i}",
                save_dir=save_dir,
                patch_size=patch_size
            )
            save_count += 1

    return images, save_count

def patch_wise_robustness_test_second(loader, model, config, eps=2/255, alpha=1/255, iters=250, patch_size=16, num_patches_to_perturb=1):
    correct_clean = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):
        test_loader = test_loader[0]
        print("Using first crop of 10-crop test loader.")

    print("Starting clean accuracy evaluation...")
    batch_idx = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            _, outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct_clean += (pred == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches for clean evaluation. Current correct: {correct_clean}/{total}")

    print("Computing final results...")
    clean_acc = 100. * correct_clean / total
    log_str = f"Clean Accuracy: {clean_acc:.2f}%"
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    
    return clean_acc

def compute_a_distance_second(model, loader_source, loader_target, config, layer='last'):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from tqdm import tqdm

    model.eval()
    source_features = []
    target_features = []

    print("Extracting features for A-distance...")
    with torch.no_grad():
        for (s_batch, t_batch) in tqdm(zip_longest(loader_source, loader_target), total=min(len(loader_source), len(loader_target))):
            if s_batch is None or t_batch is None:
                break
            (x_s, _), (x_t, _) = s_batch, t_batch
            x_s, x_t = x_s.cuda(), x_t.cuda()
            f_s, _ = model(x_s)
            f_t, _ = model(x_t)
            source_features.append(f_s.cpu())
            target_features.append(f_t.cpu())

    source_features = torch.cat(source_features).numpy()
    target_features = torch.cat(target_features).numpy()
    X = np.concatenate([source_features, target_features], axis=0)
    y = np.array([0] * len(source_features) + [1] * len(target_features))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    err = 1 - accuracy_score(y_test, y_pred)
    A_dist = 2 * (1 - 2 * err)
    A_dist = max(0, A_dist)

    print(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})")
    config["out_file"].write(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})\n")
    config["out_file"].flush()
    return A_dist

def main_second():
    parser = argparse.ArgumentParser(description='Patch-Wise Adversarial Robustness Evaluation for ViT-based DANN')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'])
    parser.add_argument('--t_dset_path', type=str, default='data/office-home/Product.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='', help="output directory (in ../snapshot directory)")
    parser.add_argument('--model_path', type=str, default='snapshot/best_model.pth.tar', help="Path to the saved model")
    parser.add_argument('--eps', type=float, default=2/255, help="Maximum perturbation for PGD")
    parser.add_argument('--alpha', type=float, default=1/255, help="Step size for PGD")
    parser.add_argument('--iters', type=int, default=250, help="Number of PGD iterations")
    parser.add_argument('--num_patches', type=int, default=1, help="Number of patches to perturb")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config = {}
    config["gpu"] = args.gpu_id
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "eval_patchwise_log.txt"), "w")

    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}}
    config["dataset"] = args.dset
    config["data"] = {"test": {"list_path": args.t_dset_path, "batch_size": 8}}

    if config["dataset"] == "office-home":
        config["network"] = {"name": network.ViTFc,
                             "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True, "class_num": 65}}
    else:
        raise ValueError('Dataset cannot be recognized.')

    print("Preparing test dataset...")
    prep_dict = {}
    prep_config = config["prep"]
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**prep_config['params'])
    else:
        prep_dict["test"] = prep.image_test(**prep_config['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test"]["batch_size"]

    if prep_config["test_10crop"]:
        dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"][i])
                         for i in range(10)]
        num_samples = min(1000, len(dsets["test"][0]))
        random_indices = random.sample(range(len(dsets["test"][0])), num_samples)
        dsets["test"] = [Subset(dset, random_indices) for dset in dsets["test"]]
        dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)
                                for dset in dsets["test"]]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
        num_samples = min(1000, len(dsets["test"]))
        random_indices = random.sample(range(len(dsets["test"]), num_samples))
        dsets["test"] = Subset(dsets["test"], random_indices)
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Test dataset loaded with {num_samples} randomly selected images.")

    print("Loading model...")
    base_network = config["network"]["name"](**config["network"]["params"])
    base_network = base_network.cuda()
    checkpoint = torch.load(args.model_path, weights_only=False)

    state_dict = checkpoint.state_dict()
    new_state_dict = {k[2:] if k.startswith("0.") else k: v for k, v in state_dict.items()}
    base_network.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    source_path = '../data/office-home/Clipart.txt'
    source_list = open(source_path).readlines()
    source_dataset = ImageList(source_list, transform=prep_dict["test"][0] if config["prep"]["test_10crop"] else prep_dict["test"])
    source_loader = DataLoader(source_dataset, batch_size=8, shuffle=True, num_workers=2)

    compute_a_distance_second(base_network, source_loader, dset_loaders["test"][4] if prep_config["test_10crop"] else dset_loaders["test"], config)

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = torch.nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        print(f"Using multiple GPUs: {gpus}")

    print("Starting clean accuracy evaluation (second code)...")
    clean_acc = patch_wise_robustness_test_second(dset_loaders, base_network, config, eps=args.eps, alpha=args.alpha, iters=args.iters,
                                                  patch_size=16, num_patches_to_perturb=args.num_patches)
    print(f"Second code clean accuracy: {clean_acc:.2f}%")
    config["out_file"].close()

# Functions from first code (unchanged except for patch_wise_robustness_test)
def compute_a_distance(model, loader_source, loader_target, config):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    model.eval()
    source_features = []
    target_features = []

    print("Extracting features for A-distance...")
    with torch.no_grad():
        for (s_batch, t_batch) in tqdm(zip_longest(loader_source, loader_target),
                                       total=min(len(loader_source), len(loader_target))):
            if s_batch is None or t_batch is None:
                break
            x_s, _ = s_batch
            x_t, _ = t_batch
            x_s, x_t = x_s.cuda(), x_t.cuda()
            f_s, _ = model(x_s)
            f_t, _ = model(x_t)
            source_features.append(f_s.cpu())
            target_features.append(f_t.cpu())

    source_features = torch.cat(source_features).numpy()
    target_features = torch.cat(target_features).numpy()
    X = np.concatenate([source_features, target_features], axis=0)
    y = np.array([0] * len(source_features) + [1] * len(target_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    err = 1 - accuracy_score(y_test, y_pred)
    A_dist = 2 * (1 - 2 * err)
    A_dist = max(0, A_dist)

    print(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})")
    config["out_file"].write(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})\n")
    config["out_file"].flush()
    return A_dist

def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def patch_wise_pgd_attack(model, images, labels, eps=2/255, alpha=1/255, iters=250, patch_size=16, num_patches_to_perturb=1):
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size
    num_patches = num_patches_x * num_patches_y

    with torch.no_grad():
        _, _, attn_weights = model(images, return_attention=True)
        attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)
        attn_to_patches = attn[:, 0, 1:]
        patch_indices = attn_to_patches.argmax(dim=1).cpu().numpy()

    for _ in range(iters):
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        grad = images.grad.data
        perturbation = torch.zeros_like(images)

        for i in range(batch_size):
            idx = patch_indices[i]
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

def save_adv_image_with_box(image_tensor, patch_idx, image_id, save_dir, patch_size=16):
    img = to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(img)
    W, H = img.size
    num_patches_x = W // patch_size
    px = (patch_idx % num_patches_x) * patch_size
    py = (patch_idx // num_patches_x) * patch_size
    draw.rectangle([px, py, px + patch_size, py + patch_size], outline="red", width=2)
    img.save(os.path.join(save_dir, f"adv_img_{image_id}.png"))

def patch_fool_pgd_attack(model, images, labels, args, eps=2/255, alpha=1/255, iters=250,
                         patch_size=16, num_patches_to_perturb=1,
                         save_dir=None, global_step=0, save_count=0, max_save=5):
    device = images.device
    images = images.clone().detach().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, H, W = images.shape
    num_patches_x = W // patch_size
    num_patches_y = H // patch_size
    total_patches = num_patches_x * num_patches_y

    mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    images = (images - mu) / std

    model.zero_grad()
    if 'vit' in args.net.lower():
        features, logits, atten = model(images, return_attention=True)
    else:
        logits = model(images)
        atten = None
    init_pred = logits.max(1)[1]

    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        if 'vit' in args.net.lower() and atten is not None:
            atten_layer = atten[args.atten_select].mean(dim=1)
            atten_layer = atten_layer.mean(dim=1)[:, 1:]
            max_patch_indices = atten_layer.argsort(descending=True)[:, :num_patches_to_perturb]
        else:
            raise NotImplementedError("Attention-based selection only implemented for ViT with attention weights.")

    if args.mild_l_inf == 0:
        delta = torch.zeros_like(images, requires_grad=True).to(device)
    else:
        epsilon = args.mild_l_inf / std
        delta = 2 * epsilon * torch.rand_like(images) - epsilon + images
        delta = clamp(delta, (0 - mu) / std, (1 - mu) / std)
        delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    for _ in range(iters):
        model.zero_grad()
        opt.zero_grad()

        perturbed_images = images + delta
        if 'vit' in args.net.lower():
            features, logits, atten = model(perturbed_images, return_attention=True)
        else:
            logits = model(perturbed_images)

        ce_loss = criterion(logits, labels)
        ce_grad = torch.autograd.grad(ce_loss, delta, retain_graph=True)[0]
        ce_grad_flat = ce_grad.view(batch_size, -1).detach()

        if args.attack_mode == 'Attention' and atten is not None:
            atten_loss = 0
            atten_grads = []
            num_patches = atten[0].size(-1) - 1
            max_patch_index_matrix = max_patch_indices[:, 0]
            max_patch_index_matrix = torch.clamp(max_patch_index_matrix, 0, num_patches - 1)

            for atten_num in range(len(atten) // 2):
                if atten_num == 0:
                    continue
                atten_map = atten[atten_num].mean(dim=1)
                atten_map = atten_map[:, 1:].mean(dim=1)
                atten_map = -torch.log(atten_map + 1e-10)
                atten_loss += nn.functional.nll_loss(atten_map, max_patch_index_matrix)
                atten_grad = torch.autograd.grad(atten_loss / (len(atten) // 2), delta, retain_graph=True)[0]
                atten_grads.append(atten_grad.view(batch_size, -1))

            atten_loss = atten_loss / (len(atten) // 2)
            atten_grad_flat = sum(atten_grads) / len(atten_grads) if atten_grads else torch.zeros_like(ce_grad_flat)

            cos_sim = nn.functional.cosine_similarity(atten_grad_flat, ce_grad_flat, dim=1)
            combined_grad = PCGrad(atten_grad_flat, ce_grad_flat, cos_sim, ce_grad.shape)
            grad = - (ce_grad + args.atten_loss_weight * combined_grad)
        else:
            grad = -torch.autograd.grad(ce_loss, delta)[0]

        opt.zero_grad()
        delta.grad = grad
        opt.step()
        scheduler.step()

        if args.mild_l_2 != 0:
            radius = (args.mild_l_2 / std).squeeze()
            perturbation = delta.detach() - images
            mask = torch.zeros_like(images).to(device)
            for j in range(batch_size):
                for idx in max_patch_indices[j]:
                    row = (idx // num_patches_x) * patch_size
                    col = (idx % num_patches_x) * patch_size
                    mask[j, :, row:row+patch_size, col:col+patch_size] = 1
            perturbation = perturbation * mask
            l2 = torch.norm(perturbation.view(batch_size, 3, -1), dim=-1)
            l2_constraint = torch.clamp(radius / l2, min=0.0)
            delta.data = images + perturbation * l2_constraint.view(batch_size, 1, 1, 1)
        elif args.mild_l_inf != 0:
            epsilon = args.mild_l_inf / std
            delta.data = clamp(delta, images - epsilon, images + epsilon)

        delta.data = clamp(delta, (0 - mu) / std, (1 - mu) / std)

    mask = torch.zeros_like(images).to(device)
    for j in range(batch_size):
        for idx in max_patch_indices[j]:
            row = (idx // num_patches_x) * patch_size
            col = (idx % num_patches_x) * patch_size
            mask[j, :, row:row+patch_size, col:col+patch_size] = 1
    adv_images = images + delta * mask
    adv_images = clamp(adv_images, (0 - mu) / std, (1 - mu) / std)
    adv_images = adv_images * std + mu

    if save_dir and save_count < max_save:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(batch_size):
            if save_count >= max_save:
                break
            save_adv_image_with_box(
                adv_images[i], max_patch_indices[i, 0],
                image_id=f"{global_step}_{i}",
                save_dir=save_dir,
                patch_size=patch_size
            )
            save_count += 1

    return adv_images, save_count

def patch_wise_robustness_test(loader, model, config, eps=2/255, alpha=1/255, iters=250, patch_size=16,
                              num_patches_to_perturb=1, args=None):
    correct_adv = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):
        test_loader = test_loader[0]
        print("Using first crop of 10-crop test loader for adversarial evaluation.")

    print("Starting adversarial accuracy evaluation...")
    save_count = 0
    max_save = 5
    batch_idx = 0
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        print(f"Generating adversarial examples for batch {batch_idx + 1}...")
        adv_inputs, save_count = patch_fool_pgd_attack(
            model, inputs, labels,
            args=args,
            eps=eps, alpha=alpha, iters=iters,
            patch_size=patch_size,
            num_patches_to_perturb=num_patches_to_perturb,
            save_dir=osp.join(config["output_path"], "adv_images"),
            global_step=batch_idx,
            save_count=save_count,
            max_save=max_save
        )

        with torch.no_grad():
            _, adv_outputs = model(adv_inputs)
            _, adv_pred = torch.max(adv_outputs, 1)
            correct_adv += (adv_pred == labels).sum().item()
        total += labels.size(0)
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx} batches for adversarial evaluation. Current correct: {correct_adv}/{total}")

    print("Computing final results...")
    adv_acc = 100. * correct_adv / total
    log_str = f"Patch-Fool Adversarial Accuracy (eps={eps:.4f}, attention-based patch): {adv_acc:.2f}%"
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    
    return adv_acc

def main_first():
    parser = argparse.ArgumentParser(description='A-distance: Source vs Target and Adversarial Target')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'])
    parser.add_argument('--t_dset_path', type=str, default='data/office-home/Art.txt', help="Target dataset path list")
    parser.add_argument('--output_dir', type=str, default='adv_distance', help="Output directory (in ../snapshot)")
    parser.add_argument('--model_path', type=str, default='snapshot/best_model.pth.tar', help="Path to the saved model")
    parser.add_argument('--eps', type=float, default=2/255, help="Max PGD perturbation")
    parser.add_argument('--alpha', type=float, default=1/255, help="PGD step size")
    parser.add_argument('--iters', type=int, default=250, help="PGD iterations")
    parser.add_argument('--num_patches', type=int, default=1, help="Patches to perturb")
    parser.add_argument('--atten_select', type=int, default=4, help='Select patch based on which attention layer')
    parser.add_argument('--attack_mode', default='Attention', choices=['CE_loss', 'Attention'], help='Attack mode')
    parser.add_argument('--atten_loss_weight', type=float, default=0.002, help='Weight for attention loss')
    parser.add_argument('--attack_learning_rate', type=float, default=0.22, help='Learning rate for Adam optimizer')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.95, help='Gamma for learning rate scheduler')
    parser.add_argument('--mild_l_2', type=float, default=0., help='L2 constraint range (0-16)')
    parser.add_argument('--mild_l_inf', type=float, default=0., help='Linf constraint range (0-1)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    config = {}
    config["gpu"] = args.gpu_id
    config["output_path"] = "snapshot/" + args.output_dir
    os.makedirs(config["output_path"], exist_ok=True)
    config["out_file"] = open(os.path.join(config["output_path"], "a_distance_adv.txt"), "w")

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}}
    config["dataset"] = args.dset
    config["data"] = {"test": {"list_path": args.t_dset_path, "batch_size": 8}}

    if config["dataset"] == "office-home":
        config["network"] = {"name": network.ViTFc,
                             "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True, "class_num": 65}}
    else:
        raise ValueError('Unknown dataset.')

    print("Preparing data...")
    prep_dict = {}
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    test_list = open(config["data"]["test"]["list_path"]).readlines()
    test_dataset_full = ImageList(test_list, transform=prep_dict["test"])
    random.seed(42)
    subset_indices = random.sample(range(len(test_dataset_full)), 1000)
    test_dataset = Subset(test_dataset_full, subset_indices)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    source_list = open('../data/office-home/Clipart.txt').readlines()
    source_dataset = ImageList(source_list, transform=prep_dict["test"])
    source_loader = DataLoader(source_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    print("Loading model...")
    base_network = config["network"]["name"](**config["network"]["params"]).cuda()
    checkpoint = torch.load(args.model_path, weights_only=False)
    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    base_network.load_state_dict(state_dict, strict=False)
    base_network.eval()
    print("Model loaded.")

    # Compute A-distance between source and target
    print("Computing A-distance between source and target...")
    compute_a_distance(base_network, source_loader, test_loader, config)

    # Generate adversarial dataset
    print("Generating adversarial dataset...")
    adv_images_list = []
    adv_labels_list = []
    save_count = 0
    max_save = 5
    batch_idx = 0
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        print(f"Generating adversarial examples for batch {batch_idx + 1}...")
        adv_inputs, save_count = patch_fool_pgd_attack(
            model=base_network,
            images=inputs,
            labels=labels,
            args=args,
            eps=args.eps,
            alpha=args.alpha,
            iters=args.iters,
            patch_size=16,
            num_patches_to_perturb=args.num_patches,
            save_dir=os.path.join(config["output_path"], "adv_images"),
            global_step=batch_idx,
            save_count=save_count,
            max_save=max_save
        )
        adv_images_list.append(adv_inputs.cpu())
        adv_labels_list.append(labels.cpu())
        batch_idx += 1

    # Create adversarial dataset and loader
    adv_dataset = torch.utils.data.TensorDataset(torch.cat(adv_images_list), torch.cat(adv_labels_list))
    adv_loader = DataLoader(adv_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # Compute A-distance between source and adversarial
    print("Computing A-distance between source and adversarial...")
    compute_a_distance(base_network, source_loader, adv_loader, config)

    # # Commented out robustness evaluation
    # gpus = config['gpu'].split(',')
    # if len(gpus) > 1:
    #     base_network = torch.nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
    #     print(f"Using multiple GPUs: {gpus}")
    #
    # print("Starting robustness evaluation...")
    # adv_acc = patch_wise_robustness_test(
    #     loader={"test": test_loader},
    #     model=base_network,
    #     config=config,
    #     eps=args.eps,
    #     alpha=args.alpha,
    #     iters=args.iters,
    #     patch_size=16,
    #     num_patches_to_perturb=args.num_patches,
    #     args=args
    # )
    #
    # print(f"Adversarial accuracy: {adv_acc:.2f}%")
    # print("Evaluation completed.")

    print("A-distance computations completed.")
    config["out_file"].close()

def main():
    print("Running second code for clean accuracy...")
    main_second()
    print("\nRunning first code for A-distance evaluation...")
    main_first()

if __name__ == "__main__":
    main()