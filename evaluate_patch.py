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

def patch_wise_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=20, patch_size=16, num_patches_to_perturb=5):
    """
    Perform patch-wise PGD attack on selected patches of the input images.
    
    Args:
        model: The DANN model with ViT backbone.
        images: Input images (batch_size, 3, 224, 224).
        labels: Ground truth labels.
        eps: Maximum perturbation (L_inf norm).
        alpha: Step size for each iteration.
        iters: Number of PGD iterations.
        patch_size: Size of each patch (e.g., 16 for 16x16 patches).
        num_patches_to_perturb: Number of patches to perturb per image.
    
    Returns:
        adv_images: Perturbed images.
    """
    images = images.clone().detach().cuda().requires_grad_(True)
    ori_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    num_patches_x = width // patch_size  # 224 ÷ 16 = 14
    num_patches_y = height // patch_size  # 14

    # Randomly select patches to perturb for the entire batch
    patch_indices = np.random.choice(num_patches_x * num_patches_y, num_patches_to_perturb, replace=False)

    for _ in range(iters):
        # Forward pass
        _, outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        # Get gradients
        grad = images.grad.data

        # Create perturbation tensor (zero everywhere initially)
        perturbation = torch.zeros_like(images)

        # Apply perturbation only to selected patches
        for idx in patch_indices:
            px = (idx % num_patches_x) * patch_size
            py = (idx // num_patches_x) * patch_size
            patch_grad = grad[:, :, py:py+patch_size, px:px+patch_size]
            perturbation[:, :, py:py+patch_size, px:px+patch_size] = alpha * patch_grad.sign()

        # Update images
        adv_images = images + perturbation
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()
        images.requires_grad = True
        images.grad = None

    return images

def patch_wise_robustness_test(loader, model, config, eps=8/255, alpha=2/255, iters=20, patch_size=16, num_patches_to_perturb=5):
    """
    Evaluate model robustness against patch-wise PGD attacks.
    
    Args:
        loader: Dictionary containing test DataLoader(s).
        model: The DANN model with ViT backbone.
        config: Configuration dictionary.
        eps: Maximum perturbation.
        alpha: Step size.
        iters: Number of iterations.
        patch_size: Size of each patch.
        num_patches_to_perturb: Number of patches to perturb.
    
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
    # Evaluate adversarial accuracy
    batch_idx = 0
    for data in test_loader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        print(f"Generating adversarial examples for batch {batch_idx + 1}...")
        adv_inputs = patch_wise_pgd_attack(model, inputs, labels, eps=eps, alpha=alpha, iters=iters,
                                           patch_size=patch_size, num_patches_to_perturb=num_patches_to_perturb)
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
               f"Patch-Wise PGD Adversarial Accuracy (eps={eps:.4f}, patches={num_patches_to_perturb}): {adv_acc:.2f}%\n"
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
    parser.add_argument('--model_path', type=str, default='snapshot/best_model.pth.tar', help='')
    parser.add_argument('--eps', type=float, default=8/255, help="Maximum perturbation for PGD")
    parser.add_argument('--alpha', type=float, default=2/255, help="Step size for each")
    parser.add_argument('--iterations', type=int, default=20, help="Number of PGD iterations")
    parser.add_argument('--num_patches', type=int, default=5, help="Number of patches to perturb")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Configuration
    config = {}
    config["gpu"] = args.gpu_id
    config["output_path"] = "snapshot/" + args.output
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "eval_patchwise_log.txt"), "w")
    
    config["prep"] = {"test_10crop": True, 'params': {"resize_size": {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}}}
    config["dataset"] = args.dset
    config["data"] = {"test": {"list_path": args.t_dset_path, "batch_size": 8}}
    
    if config["dataset"] == "office-home":
        config["network"] = {"name": network.ViTFc, 
                             "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, 
                                        "new_cls": True, "class_num": 65}}
    else:
        raise ValueError('Dataset cannot be recognized.')

    print("Preparing test dataset...")
    # Data preparation
    prep_dict = {}
    prep_config = config['prep']
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
    # Load model
    base_network = config["network"]["name"](**prep_config["network"]["params"])
    base_network = base_network.cuda()
    checkpoint = torch.load(args.model_path, weights_only=False)
    
    # Remove '0.' prefix from state_dict keys
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
    # Evaluate robustness
    patch_wise_robustness_test(dset_loaders, base_network, config, eps=args.eps, alpha=args.alpha, iters=args.iters,
                               patch_size=16, num_patches_to_perturb=args.num_patches)
    
    print("Evaluation completed.")
    config["out_file"].close()

if __name__ == "__main__":
    main()