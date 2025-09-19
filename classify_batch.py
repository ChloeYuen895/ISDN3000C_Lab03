# classify_batch.py
#
# FOR THE ADVANCED TASK
# This script processes all images in a specified folder, runs inference on each,
# and saves the results to a CSV file.
#
# Usage:
# python classify_batch.py <path_to_image_folder>

import torch
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request
import csv
import os
import sys
import glob
from pathlib import Path

# --- Configuration ---
MODEL_NAME = "ResNet18"
# URL to a raw JSON file containing the 1000 ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def get_model():
    """
    Loads and returns a pre-trained ResNet18 model in evaluation mode.
    The first time this runs, it will download the model weights.
    """
    print(f"Loading pre-trained model: {MODEL_NAME}...")
    # Load a model pre-trained on the ImageNet dataset
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Set the model to evaluation mode. This is important for inference.
    model.eval()
    print("Model loaded successfully.")
    return model


def get_labels():
    """
    Downloads and returns the list of ImageNet class labels.
    """
    print(f"Downloading class labels from {LABELS_URL}...")
    with urllib.request.urlopen(LABELS_URL) as url:
        labels = json.loads(url.read().decode())
    print("Labels downloaded successfully.")
    return labels


def process_image(image_path):
    """
    Loads an image and applies the necessary transformations for the model.
    """
    # Transformations must match what the model was trained on.
    # 1. Resize to 256x256
    # 2. Center crop to 224x224
    # 3. Convert to a PyTorch Tensor
    # 4. Normalize with ImageNet's mean and standard deviation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open the image file
    img = Image.open(image_path).convert('RGB')
    
    # Apply the transformations
    img_t = preprocess(img)
    
    # The model expects a batch of images, so we add a "batch" dimension of 1.
    # [3, 224, 224] -> [1, 3, 224, 224]
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def predict(model, image_tensor, labels):
    """
    Performs inference and returns the top prediction.
    """
    # Perform inference without calculating gradients
    with torch.no_grad():
        output = model(image_tensor)

    # The output contains raw scores (logits). We apply a softmax function
    # to convert these scores into probabilities.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 1 prediction
    top1_prob, top1_cat_id = torch.topk(probabilities, 1)
    
    # Look up the category name from the labels list
    category_name = labels[top1_cat_id.item()]
    confidence_score = top1_prob.item() * 100
    
    return category_name, confidence_score


def get_image_files(folder_path):
    """
    Find all supported image files in the specified folder.
    """
    image_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory.")
    
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(file_path)
    
    return sorted(image_files)


def process_batch(folder_path, output_csv="results.csv"):
    """
    Process all images in the specified folder and save results to CSV.
    """
    print(f"Processing all images in folder: {folder_path}")
    
    # Load model and labels once
    model = get_model()
    labels = get_labels()
    
    # Get all image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"No supported image files found in '{folder_path}'")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print(f"Found {len(image_files)} image(s) to process.")
    
    # Process each image and collect results
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            # Process the image
            image_tensor = process_image(image_path)
            category, confidence = predict(model, image_tensor, labels)
            
            # Store result
            results.append({
                'image_name': image_path.name,
                'detected_class': category,
                'confidence_level': f"{confidence:.2f}"
            })
            
            print(f"  -> {category} ({confidence:.2f}% confidence_level)")
            
        except Exception as e:
            print(f"  -> Error processing {image_path.name}: {e}")
            results.append({
                'image_name': image_path.name,
                'detected_class': 'ERROR',
                'confidence_level': f"Error: {e}"
            })
    
    # Write results to CSV
    output_path = Path(folder_path) / output_csv
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'detected_class', 'confidence_level']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Processed {len(results)} images successfully.")


if __name__ == "__main__":
    # Check if the user provided a folder path argument
    if len(sys.argv) != 2:
        print("Usage: python classify_batch.py <path_to_image_folder>")
        print("\nExample:")
        print("  python classify_batch.py test_images")
        print("  python classify_batch.py /path/to/your/images")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    try:
        # Process all images in the specified folder
        process_batch(folder_path)
        print("\n--- Batch Processing Complete ---")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please check that the folder path is correct.")
    except NotADirectoryError as e:
        print(f"\n[ERROR] {e}")
        print("Please provide a valid directory path.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("Please check your internet connection and that all libraries are installed correctly.")