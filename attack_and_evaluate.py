import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pre_process as prep
from data_list import ImageList
import network
from timm.layers import PatchEmbed, Mlp, LayerScale, DropPath
from timm.models.vision_transformer import Attention, Block

def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=20, device='cuda'):
    """
    PGD attack to generate adversarial images.
    :param model: The trained model.
    :param images: Batch of input images.
    :param labels: True labels for the images.
    :param epsilon: Maximum perturbation (L_inf norm).
    :param alpha: Step size for each iteration.
    :param num_iter: Number of iterations (PGD-20).
    :param device: Device to run the attack on.
    :return: Adversarial images.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Initialize perturbation
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)  # Ensure pixel values are in [0, 1]

    for _ in range(num_iter):
        adv_images.requires_grad = True
        _, outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        # Compute gradient of loss w.r.t. input
        grad = torch.autograd.grad(loss, adv_images)[0]

        # Update adversarial images
        adv_images = adv_images.detach() + alpha * grad.sign()
        # Project back to epsilon-ball
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

def evaluate_robustness(loader, model, test_10crop=True, epsilon=8/255, alpha=2/255, num_iter=20, device='cuda'):
    """
    Evaluate model accuracy on clean and adversarial images.
    :param loader: DataLoader for the test dataset.
    :param model: The trained model.
    :param test_10crop: Whether to use 10-crop testing.
    :param epsilon: Maximum perturbation for PGD attack.
    :param alpha: Step size for PGD attack.
    :param num_iter: Number of PGD iterations.
    :param device: Device to run the evaluation on.
    """
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    with torch.no_grad():
        if test_10crop:
            # Evaluate clean accuracy with 10-crop
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].to(device)
                labels = labels.to(device)

                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(F.softmax(predict_out, dim=1))
                outputs = sum(outputs) / len(outputs)  # Average over crops
                _, predicted = torch.max(outputs, 1)
                clean_correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Generate adversarial images for each crop
                adv_outputs = []
                for j in range(10):
                    adv_inputs = pgd_attack(model, inputs[j], labels, epsilon, alpha, num_iter, device)
                    _, adv_predict_out = model(adv_inputs)
                    adv_outputs.append(F.softmax(adv_predict_out, dim=1))
                adv_outputs = sum(adv_outputs) / len(adv_outputs)
                _, adv_predicted = torch.max(adv_outputs, 1)
                adv_correct += (adv_predicted == labels).sum().item()
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # Clean evaluation
                _, outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                clean_correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Adversarial evaluation
                adv_inputs = pgd_attack(model, inputs, labels, epsilon, alpha, num_iter, device)
                _, adv_outputs = model(adv_inputs)
                _, adv_predicted = torch.max(adv_outputs, 1)
                adv_correct += (adv_predicted == labels).sum().item()

    clean_accuracy = clean_correct / total
    adv_accuracy = adv_correct / total
    print(f"Clean Accuracy on Product: {clean_accuracy:.5f}")
    print(f"Adversarial Accuracy (PGD-20) on Product: {adv_accuracy:.5f}")

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Set up preprocessing
    prep_dict = {}
    prep_config = config["prep"]
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## Prepare data (Product dataset)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    test_bs = data_config["test"]["batch_size"]

    if prep_config["test_10crop"]:
        dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                   transform=prep_dict["test"][i]) for i in range(10)]
        dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                           shuffle=False, num_workers=2) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=2)

    ## Load the saved model
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(device)
    saved_model_path = osp.join(config["output_path"], "best_model.pth.tar")
    
    # Load the state dict and remove only the leading '0.' prefix from keys
    state_dict = torch.load(saved_model_path, weights_only=False).state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove the leading '0.' prefix (from nn.Sequential wrapper)
        if key.startswith("0."):
            new_key = key[2:]  # Strip the first 2 characters ('0.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    # Debug: Print the keys in new_state_dict to confirm
    print("Keys in new_state_dict:", list(new_state_dict.keys()))
    
    base_network.load_state_dict(new_state_dict)
    base_network.eval()

    ## Evaluate robustness
    evaluate_robustness(dset_loaders, base_network, test_10crop=prep_config["test_10crop"], device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Model Robustness with PGD-20 Attack')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='vit_small_patch16_224',
                        choices=["vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224", "ResNet18", "ResNet34",
                                 "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset used")
    parser.add_argument('--t_dset_path', type=str, default='../data/office-home/Product.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='', help="output directory of the model (in ../snapshot directory)")
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Config setup (similar to train_image.py)
    config = {}
    config["dataset"] = args.dset
    config["gpu"] = args.gpu_id
    config["output_path"] = "snapshot/" + args.output_dir
    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False, 'ViT': False}}
    if "vit" in args.net:
        config["prep"]['params']['ViT'] = True
        config["prep"]['params']['resize_size'] = args.input_size
        config["network"] = {"name": network.ViTFc, \
                             "params": {"vit_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    config["data"] = {"test": {"list_path": args.t_dset_path, "batch_size": 16}}

    if config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized.')

    main(config)