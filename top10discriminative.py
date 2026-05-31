import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import network
import pre_process as prep
from data_list import ImageList
import matplotlib.pyplot as plt
import scipy.ndimage  # for smoothing
def extract_attention_map(model, images, output_path, batch_idx, prefix="clean", is_correct=True):
    """
    Extract and visualize attention weights from ViT model as a smooth, interpretable heatmap.

    Args:
        model: The DANN model with ViT backbone.
        images: Input images (batch_size, 3, 224, 224).
        output_path: Directory to save the attention map PNG.
        batch_idx: Batch index (used in filename).
        prefix: 'clean' or 'adv' string for labeling.
        is_correct: Boolean indicating if prediction was correct.
    """
    model.eval()
    with torch.no_grad():
        _, _, attn_weights = model(images, return_attention=True)

        # Average over layers and heads
        attn = torch.stack(attn_weights).mean(dim=0).mean(dim=1)  # [batch_size, num_tokens, num_tokens]
        attn = attn[:, 0, 1:]  # CLS token to all patch tokens
        attn = attn.mean(dim=0)  # Mean over batch
        attn = attn.cpu().numpy().reshape(14, 14)

        # ✅ Smooth attention with Gaussian filter
        attn = scipy.ndimage.gaussian_filter(attn, sigma=1.2)

        # ✅ Normalize to [0, 1]
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # ✅ Optional: Clip outliers (uncomment if needed)
        # vmin, vmax = np.percentile(attn, 5), np.percentile(attn, 95)
        # attn = np.clip(attn, vmin, vmax)

        # ✅ Plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(attn, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Normalized Attention Score')
        plt.title(f'{prefix.capitalize()} Attention Map - Batch {batch_idx + 1}')
        plt.xlabel('Patch X')
        plt.ylabel('Patch Y')
        plt.xticks(np.arange(14))
        plt.yticks(np.arange(14))
        plt.grid(True, which="both", linestyle='--', linewidth=0.5, color='white')

        # ✅ Save
        subfolder = "correct" if is_correct else "incorrect"
        save_path = osp.join(output_path, subfolder)
        os.makedirs(save_path, exist_ok=True)

        filename = osp.join(save_path, f"{prefix}_attention_map_batch_{batch_idx + 1}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {filename}")


def patch_wise_pgd_attack(model, images, labels, eps=2/255, alpha=2/255, iters=20, patch_size=16, patch_indices=None):
    """
    Perform patch-wise PGD attack on the specified top 80 discriminative patches using ground truth labels.
    
    Args:
        model: The DANN model with ViT backbone.
        images: Input images (batch_size, 3, 224, 224).
        labels: Ground truth labels for attack.
        eps: Maximum perturbation (L_inf norm).
        alpha: Step size for each iteration.
        iters: Number of PGD iterations.
        patch_size: Size of each patch (e.g., 16 for 16x16 patches).
        patch_indices: List of patch indices to attack (top 80 discriminative patches).
    
    Returns:
        adv_images: Perturbed images.
    """
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size  # 224 ÷ 16 = 14
    num_patches_y = height // patch_size  # 14

    # Compute patch positions for all patch indices
    patch_positions = []
    for patch_idx in patch_indices:
        px = (patch_idx % num_patches_x) * patch_size
        py = (patch_idx // num_patches_x) * patch_size
        patch_positions.append((px, py))

    for _ in range(iters):
        images.grad = None
        with torch.enable_grad():
            _, outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)  # Use ground truth labels
            loss.backward()
            print(f"PGD Iteration Loss: {loss.item()}")  # Debug print

        grad = images.grad.data
        perturbation = torch.zeros_like(images)

        for i in range(batch_size):
            for px, py in patch_positions:
                patch_grad = grad[i:i+1, :, py:py+patch_size, px:px+patch_size]
                perturbation[i:i+1, :, py:py+patch_size, px:px+patch_size] = alpha * patch_grad.sign()

        adv_images = images + perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
        images.requires_grad_(True)

    return images

def patch_wise_robustness_test(loader, model, config, eps=2/255, alpha=2/255, iters=20, patch_size=16, patch_indices=None):
    correct_clean = 0
    correct_adv = 0
    total = 0
    model.eval()

    test_loader = loader["test"]
    if isinstance(test_loader, list):
        test_loader = test_loader[0]
        print("Using first crop of 10-crop test loader.")

    saved_maps = 0
    MAX_SAVED_MAPS = 10

    print("Starting clean and adversarial evaluation...")
    batch_idx = 0
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()

        with torch.no_grad():
            _, clean_outputs = model(inputs)
            _, clean_preds = torch.max(clean_outputs, 1)
            clean_correct = (clean_preds == labels)

        print(f"Generating adversarial examples for batch {batch_idx + 1}...")
        adv_inputs = patch_wise_pgd_attack(model, inputs, labels, eps=eps, alpha=alpha, iters=iters,
                                           patch_size=patch_size, patch_indices=patch_indices)

        with torch.no_grad():
            _, adv_outputs = model(adv_inputs)
            _, adv_preds = torch.max(adv_outputs, 1)
            adv_correct = (adv_preds == labels)

        for i in range(inputs.size(0)):
            total += 1
            if clean_correct[i]:
                correct_clean += 1
            if adv_correct[i]:
                correct_adv += 1

            if saved_maps < MAX_SAVED_MAPS and clean_correct[i] and adv_correct[i]:
                extract_attention_map(model, inputs[i:i+1], config["output_path"], batch_idx * inputs.size(0) + i, "clean")
                extract_attention_map(model, adv_inputs[i:i+1], config["output_path"], batch_idx * inputs.size(0) + i, "adv")
                saved_maps += 1

        batch_idx += 1
        if saved_maps >= MAX_SAVED_MAPS:
            print(f"Saved {MAX_SAVED_MAPS} attention map pairs. Skipping further saves.")
            break

    print("Computing final results...")
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    patch_indices_str = ", ".join(map(str, patch_indices))
    log_str = (f"Clean Accuracy: {clean_acc:.2f}%\n"
               f"Patch-Wise PGD Adversarial Accuracy (eps={eps:.4f}, top patches [{patch_indices_str}]): {adv_acc:.2f}%\n"
               f"Robustness Drop: {clean_acc - adv_acc:.2f}%")
    print(log_str)
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()

    return clean_acc, adv_acc


def main():
    parser = argparse.ArgumentParser(description='Patch-Wise Adversarial Robustness Evaluation on Top 80 Discriminative Patches for ViT-based DANN (Target Domain - Art)')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'])
    parser.add_argument('--t_dset_path', type=str, default='data/office-home/Art.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='vit_patchwise_eval_top_80_discriminative_art', help="output directory (in ../snapshot directory)")
    parser.add_argument('--model_path', type=str, default='snapshot/vit_patchwise_adversarial_run/iter_10000_model.pth.tar', help="Path to the saved model")
    parser.add_argument('--eps', type=float, default=2/255, help="Maximum perturbation for PGD")
    parser.add_argument('--alpha', type=float, default=2/255, help="Step size for PGD")
    parser.add_argument('--iters', type=int, default=20, help="Number of PGD iterations")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # List of top 80 discriminative patch indices for Target (Art)
    top_10_patch_indices = [26, 16, 21, 22, 20, 25, 15, 40, 36, 96]


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
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('0.'):
            new_key = key[2:]  # Strip '0.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    base_network.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        print(f"Using multiple GPUs: {gpus}")

    print("Starting robustness evaluation...")
    patch_wise_robustness_test(dset_loaders, base_network, config, eps=args.eps, alpha=args.alpha, iters=args.iters,
                               patch_size=16, patch_indices=top_10_patch_indices)
    
    print("Evaluation completed.")
    config["out_file"].close()

if __name__ == "__main__":
    main()