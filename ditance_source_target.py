import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import network
import pre_process as prep
from torchvision.transforms.functional import to_pil_image
import os
from PIL import ImageDraw
from data_list import ImageList
from itertools import zip_longest

def patch_wise_pgd_attack(model, images, labels, eps=2/255, alpha=1/255, iters=20, patch_size=16, num_patches_to_perturb=1):
    """
    Perform patch-wise PGD attack on the patch with highest attention.
    
    Args:
        model: The DANN model with ViT backbone.
        images: Input images (batch_size, 3, 224, 224).
        labels: Ground truth labels.
        eps: Maximum perturbation (L_inf norm).
        alpha: Step size for each iteration.
        iters: Number of PGD iterations.
        patch_size: Size of each patch (e.g., 16 for 16x16 patches).
        num_patches_to_perturb: Number of patches to perturb (set to 1 for highest attention patch).
    
    Returns:
        adv_images: Perturbed images.
    """
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size  # 224 ÷ 16 = 14
    num_patches_y = height // patch_size  # 14
    num_patches = num_patches_x * num_patches_y  # 196

    # Get attention weights
    with torch.no_grad():
        _, _, attn_weights = model(images, return_attention=True)
        # Average attention across layers and heads
        attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)  # [batch, num_patches+1, num_patches+1]
        # Focus on attention to patches (exclude CLS token)
        attn_to_patches = attn[:, 0, 1:]  # [batch, num_patches]
        # Find patch with highest attention per image
        patch_indices = attn_to_patches.argmax(dim=1).cpu().numpy()  # [batch]

    for _ in range(iters):
        # Forward pass
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        # Get gradients
        grad = images.grad.data

        # Create perturbation tensor (zero everywhere initially)
        perturbation = torch.zeros_like(images)

        # Apply perturbation to the highest-attention patch for each image
        for i in range(batch_size):
            idx = patch_indices[i]
            px = (idx % num_patches_x) * patch_size
            py = (idx // num_patches_x) * patch_size
            patch_grad = grad[i:i+1, :, py:py+patch_size, px:px+patch_size]
            perturbation[i:i+1, :, py:py+patch_size, px:px+patch_size] = alpha * patch_grad.sign()

        # Update images
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


def patch_fool_pgd_attack(model, images, labels, eps=2/255, alpha=1/255, iters=20,
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
            save_adv_image_with_box(
                images[i], patch_indices[i],
                image_id=f"{global_step}_{i}",
                save_dir=save_dir,
                patch_size=patch_size
            )
            save_count += 1

    return images, save_count

def patch_wise_robustness_test(loader, model, config, eps=2/255, alpha=1/255, iters=20, patch_size=16, num_patches_to_perturb=1):
    """
    Evaluate model robustness against patch-wise PGD attacks on highest-attention patch.
    
    Args:
        loader: Dictionary containing test DataLoader(s).
        model: The DANN model with ViT backbone.
        config: Configuration dictionary.
        eps: Maximum perturbation.
        alpha: Step size.
        iters: Number of iterations.
        patch_size: Size of each patch.
        num_patches_to_perturb: Number of patches to perturb (1 for highest attention).
    
    Returns:
        clean_acc: Accuracy on clean images.
        adv_acc: Accuracy on adversarially perturbed images.
    """
    correct_clean = 0
    correct_adv = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):  # 10-crop mode
        test_loader = test_loader[0]  # Use first crop for evaluation
        print("Using first crop of 10-crop test loader.")

    print("Starting clean accuracy evaluation...")
    # Evaluate clean accuracy
    batch_idx = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].cuda(), data[1].cuda()
            _, outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct_clean += (pred == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Processed {batch_idx} batches for clean evaluation. Current correct: {correct_clean}/{total}")

    print("Starting adversarial accuracy evaluation...")

    save_count = 0
    max_save = 5
    # Evaluate adversarial accuracy
    batch_idx = 0
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        print(f"Generating adversarial examples for batch {batch_idx + 1}...")
        adv_inputs, save_count = patch_fool_pgd_attack(
            model, inputs, labels,
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
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Processed {batch_idx} batches for adversarial evaluation. Current correct: {correct_adv}/{total}")

    print("Computing final results...")
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    log_str = (f"Clean Accuracy: {clean_acc:.2f}%\n"
               f"Patch-Wise PGD Adversarial Accuracy (eps={eps:.4f}, highest-attention patch): {adv_acc:.2f}%\n"
               f"Robustness Drop: {clean_acc - adv_acc:.2f}%")
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()
    
    return clean_acc, adv_acc

def main():
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
    parser.add_argument('--iters', type=int, default=20, help="Number of PGD iterations")
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
        dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True) 
                                for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Test dataset loaded with {len(dsets['test'][0]) if prep_config['test_10crop'] else len(dsets['test'])} images.")

    print("Loading model...")
    base_network = config["network"]["name"](**config["network"]["params"])
    base_network = base_network.cuda()
    checkpoint = torch.load(args.model_path, weights_only=False)
    
    state_dict = checkpoint.state_dict()
    new_state_dict = {k[2:] if k.startswith("0.") else k: v for k, v in state_dict.items()}
    base_network.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    def compute_a_distance(model, loader_source, loader_target, config, layer='last'):
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
        A_dist = 2 * (1 - 2 * min(err, 1 - err))  # ✅ Correct formula
        A_dist = max(0, A_dist)  # Avoid negative distance

        print(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})")
        config["out_file"].write(f"A-distance = {A_dist:.4f} (Error rate: {err:.4f})\n")
        config["out_file"].flush()
        return A_dist

    # Load source data
    source_path = '../data/office-home/Clipart.txt'  # Adjust if needed
    source_list = open(source_path).readlines()
    source_dataset = ImageList(source_list, transform=prep_dict["test"][0] if config["prep"]["test_10crop"] else prep_dict["test"])
    source_loader = DataLoader(source_dataset, batch_size=8, shuffle=True, num_workers=2)

    # Compute A-distance on clean features
    compute_a_distance(base_network, source_loader, dset_loaders["test"][4], config)


    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = torch.nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        print(f"Using multiple GPUs: {gpus}")

    print("Starting robustness evaluation...")
    patch_wise_robustness_test(dset_loaders, base_network, config, eps=args.eps, alpha=args.alpha, iters=args.iters,
                                patch_size=16, num_patches_to_perturb=args.num_patches)
    
    print("Evaluation completed.")
    config["out_file"].close()

if __name__ == "__main__":
    main()