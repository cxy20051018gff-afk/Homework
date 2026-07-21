"""
goldfish_detector.py
-------------------
This program estimates the probability that a goldfish appears in a given image.
It uses a ResNet-50 model pre-trained on ImageNet (which includes the "goldfish" class).

Usage:
    python goldfish_detector.py path/to/image.jpg

Requirements:
    pip install torch torchvision Pillow
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_model():
    """Load a pretrained ResNet-50 and set it to evaluation mode."""
    # weights='DEFAULT' downloads the latest available weights with the model.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    return model


def get_goldfish_index():
    """Return the index of the 'goldfish' class in ImageNet 1000 classes."""
    # The meta data from the pretrained weights contains the list of category names.
    weights = models.ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]

    # The goldfish class is called "goldfish" exactly in the ImageNet labels.
    try:
        goldfish_idx = categories.index("goldfish")
    except ValueError:
        # Fallback: in older torchvision versions "goldfish" might be "goldfish, Carassius auratus"
        goldfish_idx = categories.index("goldfish, Carassius auratus")
    return goldfish_idx


def preprocess_image(image_path):
    """Load an image from disk and apply the standard preprocessing for ResNet."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],   # ImageNet stds
            ),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img)
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Calculate probability that an image contains a goldfish."
    )
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: File '{args.image}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load model and goldfish class index
    print("Loading model...")
    model = load_model()
    goldfish_idx = get_goldfish_index()

    # Prepare image
    print(f"Processing image: {args.image}")
    try:
        input_tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)        # logits (1, 1000)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    goldfish_prob = probabilities[goldfish_idx].item()

    # Display result
    print(f"Probability of containing a goldfish: {goldfish_prob:.4f}  ({goldfish_prob*100:.2f}%)")


if __name__ == "__main__":
    main()