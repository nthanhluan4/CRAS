"""
CRAS Single-Image / Batch Inference Script
Usage:
    python infer.py --model_dir results/models/backbone_0/mvtec_dataset --input path/to/image.png
    python infer.py --model_dir results/models/backbone_0/mvtec_dataset --input path/to/image_folder/
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import backbones
import cras
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="CRAS Defect Detection Inference")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing ckpt_best_*.pth and center.pth")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an input image or directory of images")
    parser.add_argument("--output_dir", type=str, default="results/inference",
                        help="Directory to save output visualizations")
    parser.add_argument("--backbone", type=str, default="wideresnet50",
                        help="Backbone architecture name (default: wideresnet50)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to use ('cpu' or 'cuda'). Auto-detected if empty.")
    parser.add_argument("--resize", type=int, default=329)
    parser.add_argument("--imagesize", type=int, default=288)
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Anomaly score threshold for defect decision")
    return parser.parse_args()


def load_model(args, device):
    # Load backbone
    backbone = backbones.load(args.backbone)
    backbone.name = args.backbone
    backbone.seed = None

    cras_model = cras.CRAS(device)
    cras_model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=args.patchsize,
        meta_epochs=1,
        eval_epochs=1,
        dsc_layers=3,
        train_backbone=False,
        pre_proj=1,
        noise=0.015,
        k=0.3,
        lr=0.0001,
        limit=-1,
    )

    # Find checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(args.model_dir, "ckpt_best*.pth")))
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(args.model_dir, "*.pth"))
        ckpt_files = [f for f in ckpt_files if "center" not in os.path.basename(f)]

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint file found in {args.model_dir}")

    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)

    if "discriminator" in state_dict:
        cras_model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict and cras_model.pre_proj > 0:
            cras_model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        cras_model.load_state_dict(state_dict, strict=False)

    # Load center features
    center_path = os.path.join(args.model_dir, "center.pth")
    if not os.path.exists(center_path):
        raise FileNotFoundError(f"Center file not found at: {center_path}")
    print(f"Loading center features from: {center_path}")
    cras_model.c2 = torch.load(center_path, map_location=device)

    cras_model.to(device)
    cras_model.eval()
    return cras_model


def get_image_list(input_path):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for root, _, filenames in os.walk(input_path):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    files.append(os.path.join(root, f))
        return files
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args, device)
    image_paths = get_image_list(args.input)
    print(f"Found {len(image_paths)} image(s) to process.")

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=utils.IMAGENET_MEAN, std=utils.IMAGENET_STD),
    ])

    for idx, img_path in enumerate(image_paths):
        raw_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_pil.size

        img_tensor = transform(raw_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            scores, masks = model._predict(img_tensor)

        score = float(scores[0])
        mask = masks[0]  # (H, W) numpy array

        # Mask is a Sigmoid probability: keep the absolute scale (min-max stretching invents defects on good fabric)
        norm_mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

        # Resize for display
        raw_bgr = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        overlay = cv2.addWeighted(raw_bgr, 0.6, heatmap_resized, 0.4, 0)

        # Decision
        is_defective = score > args.threshold
        status_text = f"DEFECT (Score: {score:.3f})" if is_defective else f"GOOD (Score: {score:.3f})"
        status_color = (0, 0, 255) if is_defective else (0, 255, 0)

        # Draw label on overlay
        cv2.putText(overlay, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Concat original and overlay
        concat_img = np.hstack([raw_bgr, overlay])

        base_name = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, f"result_{base_name}")
        cv2.imwrite(out_path, concat_img)

        print(f"[{idx + 1}/{len(image_paths)}] {base_name}: {status_text} -> Saved to {out_path}")

    print(f"\nAll results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
