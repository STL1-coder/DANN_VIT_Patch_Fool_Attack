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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image


def fft_low_freq_mask(image, patch_size=16, keep_ratio=0.1):
    B, C, H, W = image.shape
    mask = torch.zeros((B, 1, H, W), device=image.device)

    for i in range(B):
        img = image[i].detach().cpu().numpy()
        img_fft = np.fft.fft2(img, axes=(-2, -1))
        img_fftshift = np.fft.fftshift(img_fft, axes=(-2, -1))

        _, h, w = img.shape
        h_keep = int(h * keep_ratio)
        w_keep = int(w * keep_ratio)

        center_h = h // 2
        center_w = w // 2

        freq_mask = np.zeros_like(img_fft)
        freq_mask[:, center_h - h_keep // 2:center_h + h_keep // 2,
        center_w - w_keep // 2:center_w + w_keep // 2] = 1

        # This will give us a mask in frequency domain, now inverse FFT to get spatial mask
        low_freq_mask = np.fft.ifftshift(freq_mask, axes=(-2, -1))
        low_freq_img = np.abs(np.fft.ifft2(img_fft * low_freq_mask, axes=(-2, -1)))

        # Normalize and binarize to get regions
        low_freq_img = (low_freq_img - low_freq_img.min()) / (low_freq_img.max() - low_freq_img.min())
        low_freq_mask_spatial = (low_freq_img > 0.1).astype(np.uint8)

        # Downsample to match patch-level grid
        mask_h, mask_w = h // patch_size, w // patch_size
        patch_mask = torch.zeros(mask_h, mask_w)

        for y in range(mask_h):
            for x in range(mask_w):
                patch = low_freq_mask_spatial[:, y * patch_size:(y + 1) * patch_size,
                        x * patch_size:(x + 1) * patch_size]
                patch_mask[y, x] = patch.mean() > 0.5  # Majority of pixels are low freq

        mask[i] = patch_mask.unsqueeze(0).repeat(1, patch_size, patch_size)

    return mask  # shape: [B, 1, H, W]


# Functions from first code (unchanged)
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


def patch_wise_pgd_attack(model, images, labels, eps=2 / 255, alpha=1 / 255, iters=250, patch_size=16,
                          num_patches_to_perturb=1):
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
            patch_grad = grad[i:i + 1, :, py:py + patch_size, px:px + patch_size]
            perturbation[i:i + 1, :, py:py + patch_size, px:px + patch_size] = alpha * patch_grad.sign()

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


def patch_fool_pgd_attack(model, images, labels, args, eps=2 / 255, alpha=1 / 255, iters=250,
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
                atten_loss += F.nll_loss(atten_map, max_patch_index_matrix)
                atten_grad = torch.autograd.grad(atten_loss / (len(atten) // 2), delta, retain_graph=True)[0]
                atten_grads.append(atten_grad.view(batch_size, -1))

            atten_loss = atten_loss / (len(atten) // 2)
            atten_grad_flat = sum(atten_grads) / len(atten_grads) if atten_grads else torch.zeros_like(ce_grad_flat)

            cos_sim = F.cosine_similarity(atten_grad_flat, ce_grad_flat, dim=1)
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
                    mask[j, :, row:row + patch_size, col:col + patch_size] = 1
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
            mask[j, :, row:row + patch_size, col:col + patch_size] = 1
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


def patch_wise_robustness_test(loader, model, config, eps=2 / 255, alpha=1 / 255, iters=250, patch_size=16,
                               num_patches_to_perturb=1, args=None):
    correct_clean = 0
    correct_adv = 0
    total = 0
    model.eval()

    print("Preparing 10-crop test loaders for clean accuracy...")
    config_10crop = config.copy()
    config_10crop["prep"]["test_10crop"] = True
    prep_dict_10crop = {}
    prep_dict_10crop["test"] = prep.image_test_10crop(**config_10crop["prep"]['params'])
    test_list = open(config["data"]["test"]["list_path"]).readlines()
    test_datasets_10crop = [ImageList(test_list, transform=prep_dict_10crop["test"][i])
                            for i in range(10)]
    random.seed(42)
    num_samples = min(1000, len(test_datasets_10crop[0]))
    subset_indices = random.sample(range(len(test_datasets_10crop[0])), num_samples)
    test_datasets_10crop = [Subset(dset, subset_indices) for dset in test_datasets_10crop]
    test_loaders_10crop = [DataLoader(dset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
                           for dset in test_datasets_10crop]

    print("Starting clean accuracy evaluation with 10-crop...")
    batch_idx = 0
    with torch.no_grad():
        for batch_data in test_loaders_10crop[0]:
            _, labels = batch_data[0].cuda(), batch_data[1].cuda()
            logits_all = []

            for loader in test_loaders_10crop:
                inputs, _ = next(iter(loader))
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                logits_all.append(outputs)

            logits_avg = torch.stack(logits_all).mean(dim=0)
            _, pred = torch.max(logits_avg, 1)
            correct_clean += (pred == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches for clean evaluation. Current correct: {correct_clean}/{total}")

    # Ensure loader["test"] is handled correctly
    test_loader = loader["test"]
    if isinstance(test_loader, list):
        test_loader = test_loader[0]
        print("Using first crop of 10-crop test loader for adversarial evaluation.")
    elif not isinstance(test_loader, DataLoader):
        raise TypeError(f"Expected loader['test'] to be a DataLoader or list of DataLoaders, got {type(test_loader)}")

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
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx} batches for adversarial evaluation. Current correct: {correct_adv}/{total}")

    print("Computing final results...")
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    log_str = (f"Clean Accuracy (10-crop): {clean_acc:.2f}%\n"
               f"Patch-Fool Adversarial Accuracy (eps={eps:.4f}, attention-based patch): {adv_acc:.2f}%\n"
               f"Robustness Drop: {clean_acc - adv_acc:.2f}%")
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()

    return clean_acc, adv_acc


def run_similarity_map_visualization(model, dataset_path, args, save_dir="similarity_maps"):
    os.makedirs(save_dir, exist_ok=True)

    prep_fn = prep.image_test(**{"resize_size": 224, "crop_size": 224, "alexnet": False, "ViT": True})
    test_list = open(dataset_path).readlines()
    random.seed(42)
    selected = random.sample(test_list, 10)

    images = []
    labels = []
    names = []

    for idx, line in enumerate(selected):
        path, label = line.strip().split()
        full_path = os.path.abspath(path)  # Adjust if your data path is different
        image = Image.open(full_path).convert("RGB")
        image = prep_fn(image)
        images.append(image)
        labels.append(int(label))
        names.append(f"class_{label}_img{idx}_clean")

    images = torch.stack(images).cuda()
    labels = torch.tensor(labels).cuda()

    compute_and_plot_cosine_map(model, images, labels, names, save_dir)

    print("Generating adversarial images...")
    adv_images, _ = patch_fool_pgd_attack(model, images, labels, args, save_dir=None)

    adv_names = [n.replace("_clean", "_adv") for n in names]
    compute_and_plot_cosine_map(model, adv_images, labels, adv_names, save_dir, original_images=images)


def main_first():
    parser = argparse.ArgumentParser(description='A-distance: Source vs Target and Adversarial Target')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224",
                                 "vit_huge_patch14_224"])
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['office', 'image-clef', 'visda', 'office-home'])
    parser.add_argument('--t_dset_path', type=str, default='../data/office-home/Art.txt',
                        help="Target dataset path list")
    parser.add_argument('--output_dir', type=str, default='adv_distance', help="Output directory (in ../snapshot)")
    parser.add_argument('--model_path', type=str, default='snapshot/best_model.pth.tar', help="Path to the saved model")
    parser.add_argument('--eps', type=float, default=2 / 255, help="Max PGD perturbation")
    parser.add_argument('--alpha', type=float, default=1 / 255, help="PGD step size")
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
    config["out_file"] = open(osp.join(config["output_path"], "a_distance_adv.txt"), "w")

    config["prep"] = {"test_10crop": False,
                      'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}}
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

    config_10crop = config.copy()
    config_10crop["prep"]["test_10crop"] = True
    prep_dict_10crop = {}
    prep_dict_10crop["test"] = prep.image_test_10crop(**config_10crop["prep"]['params'])
    test_datasets_10crop = [ImageList(test_list, transform=prep_dict_10crop["test"][i])
                            for i in range(10)]
    test_datasets_10crop = [Subset(dset, subset_indices) for dset in test_datasets_10crop]
    test_loaders_10crop = [DataLoader(dset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
                           for dset in test_datasets_10crop]

    # compute_a_distance(base_network, source_loader, test_loaders_10crop[4], config)

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = torch.nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        print(f"Using multiple GPUs: {gpus}")

    # print("Starting robustness evaluation (first code)...")
    # patch_wise_robustness_test(
    #    loader={"test": test_loader},
    #    model=base_network,
    #    config=config,
    #    eps=args.eps,
    #    alpha=args.alpha,
    #    iters=args.iters,
    #    patch_size=16,
    #    num_patches_to_perturb=args.num_patches,
    #    args=args
    # )
    run_similarity_map_visualization(base_network, args.t_dset_path, args)
    print("Evaluation completed.")
    config["out_file"].close()


def main():
    print("\nRunning first code for full evaluation...")
    main_first()


def compute_and_plot_cosine_map(model, images, labels, image_names, save_dir, patch_size=16, original_images=None):
    model.eval()
    cls_token, patch_tokens = model.get_cls_patch_embeddings(images)

    cosine_sim = F.cosine_similarity(cls_token.unsqueeze(1), patch_tokens, dim=-1)  # [B, N]
    num_patches = patch_tokens.shape[1]
    num_patches_side = int(num_patches ** 0.5)

    low_freq_masks = fft_low_freq_mask(images, patch_size=patch_size, keep_ratio=0.1)

    for i in range(images.size(0)):
        sim_map = cosine_sim[i].reshape(num_patches_side, num_patches_side).cpu().numpy()
        lf_mask = low_freq_masks[i, 0].reshape(num_patches_side, patch_size, num_patches_side, patch_size).mean(
            axis=(1, 3)).cpu().numpy()

        # Image to display
        if original_images is not None:
            img = original_images[i].detach().cpu().permute(1, 2, 0).numpy()
        else:
            img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(img)
        axs[0].axis("off")
        axs[0].set_title("Image")

        im1 = axs[1].imshow(sim_map, cmap="viridis")
        axs[1].axis("off")
        axs[1].set_title("CLS Similarity")
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(lf_mask, cmap="gray")
        axs[2].axis("off")
        axs[2].set_title("Low-Frequency Mask")
        fig.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{image_names[i]}_sim_lf.png"))
        plt.close()


if __name__ == "__main__":
    main()