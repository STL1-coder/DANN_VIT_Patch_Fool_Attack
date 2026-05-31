import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import argparse
from data_list import ImageList
import pre_process as prep
import network
from timm.models.vision_transformer import PatchEmbed, Block
from timm.layers import DropPath

def generate_pseudo_labels(config, model_path, output_path, confidence_threshold=0.9):
    print(f"Starting pseudo-label generation...")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Target dataset path: {config['data']['target']['list_path']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load preprocessing
    prep_dict = {}
    prep_config = config["prep"]
    print(f"Preprocessing config: {prep_config}")
    try:
        prep_dict["target"] = prep.image_test(**prep_config['params'])
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
    
    # Load target dataset
    data_config = config["data"]
    dsets = {}
    dset_loaders = {}
    test_bs = data_config["test"]["batch_size"]
    print(f"Loading target dataset from {data_config['target']['list_path']}")
    try:
        with open(data_config["target"]["list_path"], 'r') as f:
            image_list = f.readlines()
        print(f"Number of target images: {len(image_list)}")
        if len(image_list) == 0:
            raise ValueError("Target dataset file is empty!")
    except Exception as e:
        print(f"Error loading target dataset: {e}")
        return None
    
    try:
        dsets["target"] = ImageList(image_list, transform=prep_dict["target"])
        print(f"Target dataset size: {len(dsets['target'])}")
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=test_bs, shuffle=False, num_workers=2, pin_memory=True)
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        return None
    
    # Load pretrained DANN model
    print(f"Loading model from {model_path}")
    try:
        # Allowlist necessary globals for weights_only=True
        torch.serialization.add_safe_globals([nn.Sequential, network.ViTFc, PatchEmbed, Block, nn.Linear, nn.LayerNorm, nn.Dropout, DropPath, nn.Conv2d, nn.Identity])
        checkpoint = torch.load(model_path, weights_only=True)
        if isinstance(checkpoint, nn.Module):
            print("Checkpoint is a full model, loading directly...")
            model = checkpoint.cuda()
        else:
            print("Checkpoint is a state dictionary, loading into new model...")
            base_network = config["network"]["name"](**config["network"]["params"]).cuda()
            model = nn.Sequential(base_network)
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"Error loading model with weights_only=True: {e}")
        print("Attempting to load with weights_only=False as fallback...")
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            if isinstance(checkpoint, nn.Module):
                print("Checkpoint is a full model, loading directly...")
                model = checkpoint.cuda()
            else:
                print("Checkpoint is a state dictionary, loading into new model...")
                base_network = config["network"]["name"](**config["network"]["params"]).cuda()
                model = nn.Sequential(base_network)
                model.load_state_dict(checkpoint)
            model.eval()
        except Exception as e:
            print(f"Fallback failed: {e}")
            return None
    
    # Generate pseudo-labels
    pseudo_labels = []
    confidences = []
    image_paths = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    # Extract image paths from image_list
    image_list_paths = [line.strip().split()[0] for line in image_list]
    
    print("Processing target data...")
    try:
        for i, data in enumerate(dset_loaders["target"]):
            # Inspect DataLoader output
            print(f"Batch {i+1}: DataLoader output type: {type(data)}, length: {len(data) if isinstance(data, (list, tuple)) else 'N/A'}")
            if isinstance(data, (list, tuple)):
                print(f"DataLoader output elements: {[type(x) for x in data]}")
            
            # Expect ImageList to return (image, target)
            images, _ = data[0].to(device), data[1]
            batch_size = images.size(0)
            start_idx = i * test_bs
            end_idx = min((i + 1) * test_bs, len(image_list_paths))
            batch_indices = list(range(start_idx, end_idx))
            batch_paths = [image_list_paths[idx] for idx in batch_indices]
            
            with torch.no_grad():
                outputs = model(images)  # Model returns (features, outputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[1]  # Use classification output
                softmax_out = nn.Softmax(dim=1)(outputs)
                max_probs, predicted = torch.max(softmax_out, 1)
                pseudo_labels.extend(predicted.cpu().numpy())
                confidences.extend(max_probs.cpu().numpy())
                image_paths.extend(batch_paths)
            
            print(f"Batch {i+1}: Processed {batch_size} images")
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    print(f"Total predictions: {len(pseudo_labels)}")
    
    # Filter by confidence threshold
    filtered_data = []
    for path, label, conf in zip(image_paths, pseudo_labels, confidences):
        if conf >= confidence_threshold:
            filtered_data.append(f"{path} {label}\n")
    
    print(f"Filtered samples (confidence >= {confidence_threshold}): {len(filtered_data)}")
    
    # Save pseudo-labels
    pseudo_label_path = osp.join(output_path, "pseudo_labels.txt")
    try:
        with open(pseudo_label_path, "w") as f:
            f.writelines(filtered_data)
        print(f"Pseudo-labels saved to {pseudo_label_path}")
    except Exception as e:
        print(f"Error saving pseudo-labels: {e}")
        return None
    
    return pseudo_label_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Pseudo-Labels for Target Domain')
    parser.add_argument('--model_path', type=str, default='snapshot/vit_pgd_test_run/best_model.pth.tar', help="Path to pretrained DANN model")
    parser.add_argument('--output_dir', type=str, default='snapshot/vit_pgd_test_run/pseudo_labels', help="Output directory for pseudo-labels")
    parser.add_argument('--t_dset_path', type=str, default='..data/office-home/Product.txt', help="Target dataset path list")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224', help="Network type")
    parser.add_argument('--dset', type=str, default='office-home', help="Dataset used")
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help="Confidence threshold for pseudo-labels")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    config = {
        "gpu": "0",
        "output_path": args.output_dir,
        "prep": {"test_10crop": False, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False, 'ViT': True}},
        "data": {"target": {"list_path": args.t_dset_path, "batch_size": 4}, "test": {"list_path": args.t_dset_path, "batch_size": 4}},
        "dataset": args.dset,
        "network": {"name": network.ViTFc, "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True, "class_num": 65}}
    }
    
    print(f"Config: {config}")
    
    if not osp.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
    
    config["out_file"] = open(osp.join(args.output_dir, "pseudo_label_log.txt"), "w")
    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    
    pseudo_label_path = generate_pseudo_labels(config, args.model_path, args.output_dir, args.confidence_threshold)
    if pseudo_label_path:
        print(f"Script completed successfully. Check {pseudo_label_path}")
    else:
        print("Script failed to generate pseudo-labels.")
    
    config["out_file"].close()