# rotateImages_test.py

import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# Import ProjectilePointDataset from preprocessing.py
from utils.preprocessing import ProjectilePointDataset, collate_fn  # Ensure collate_fn is imported

# Import bounding box utility functions
from utils.bbox_utils import (
    cwh_theta_to_corners,
    clip_coordinates
)

import numpy as np
import matplotlib.pyplot as plt

def draw_bounding_box(image, bbox_cwh_theta, color=(255, 0, 0), width=2):
    """
    Draws a bounding box on a PIL Image based on CWHθ format, ensuring coordinates are within image boundaries.

    Args:
        image (PIL.Image.Image): The image to draw on.
        bbox_cwh_theta (list or array-like): Bounding box parameters [x_center, y_center, width, height, angle].
        color (tuple): RGB color of the bounding box.
        width (int): Width of the bounding box lines.

    Returns:
        PIL.Image.Image: Image with bounding box drawn.
    """
    if len(bbox_cwh_theta) != 5:
        print("Invalid bbox length. Expected 5 values [x_center, y_center, width, height, angle].")
        return image

    x_center, y_center, bbox_width, bbox_height, angle = bbox_cwh_theta

    # Convert CWHθ to corner points
    corner_points = cwh_theta_to_corners(x_center, y_center, bbox_width, bbox_height, angle)

    # Convert float coordinates to integers
    corner_points_int = [(int(round(x)), int(round(y))) for x, y in corner_points]

    # Debugging: Print corner points and their types
    print(f"  Drawing bounding box with points: {corner_points_int}")
    for point in corner_points_int:
        print(f"    Point: {point}, Types: {type(point[0])}, {type(point[1])}")

    # Get image dimensions
    image_width, image_height = image.size

    # Clip coordinates to image boundaries
    corner_points_clipped = [clip_coordinates(x, y, image_width, image_height) for x, y in corner_points_int]

    # Debugging: Print clipped corner points
    print(f"  Clipped corner points: {corner_points_clipped}")

    # Flatten the list of tuples into a flat list of integers
    flat_points = [coord for point in corner_points_clipped + [corner_points_clipped[0]] for coord in point]

    # Debugging: Print flat_points and their types
    print(f"  Flat points: {flat_points}")
    print(f"  Flat points types: {[type(coord) for coord in flat_points]}")

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
        print(f"  Converted image mode to 'RGB'.")

    # Explicitly convert all flat_points to Python's int type
    flat_points = [int(coord) for coord in flat_points]

    # Debugging: Print flat_points after conversion
    print(f"  Flat points after int conversion: {flat_points}")
    print(f"  Flat points types after int conversion: {[type(coord) for coord in flat_points]}")

    # Draw the bounding box
    draw = ImageDraw.Draw(image)
    try:
        # Convert flat_points to tuple for Pillow
        draw.line(tuple(flat_points), fill=color, width=int(width))
    except Exception as e:
        print(f"  Error drawing line: {e}")
        return image

    return image

def overlay_text(image, text, position=(10, 10), font_size=20, color=(255, 255, 255)):
    """
    Overlays text on a PIL Image.

    Args:
        image (PIL.Image.Image): The image to overlay text on.
        text (str): The text to overlay.
        position (tuple): (x, y) position for the text.
        font_size (int): Font size of the text.
        color (tuple): RGB color of the text.

    Returns:
        PIL.Image.Image: Image with overlaid text.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return image

def save_preprocessed_images(dataset, dataloader, save_dir, num_images=10):
    """
    Saves a specified number of preprocessed images from the dataset to a directory,
    overlaying the rotated bounding boxes and rotation angles.

    Args:
        dataset (Dataset): The dataset to retrieve images from.
        dataloader (DataLoader): DataLoader for batching and shuffling.
        save_dir (str): Directory where images will be saved.
        num_images (int): Number of images to save.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    saved_count = 0
    to_pil = ToPILImage()

    for batch_idx, batch in enumerate(dataloader):
        # Skip batches that are None or empty
        if batch is None or not batch:
            continue

        images, angles, bboxes = batch  # Unpack the batch

        # Ensure images, angles, and bboxes are not empty
        if images is None or angles is None or bboxes is None:
            continue

        for i in range(len(images)):
            if saved_count >= num_images:
                break  # Exit after saving the desired number of images

            image_tensor = images[i]  # Image tensor: [C, H, W]
            angle = angles[i].item()  # Normalized angle [0, 1]
            bbox = bboxes[i].tolist()  # [x_center, y_center, width, height, angle]

            # Debugging: Print bounding box parameters
            print(f"Image {saved_count + 1}:")
            print(f"  Center: ({bbox[0]:.4f}, {bbox[1]:.4f})")
            print(f"  Width: {bbox[2]:.4f}, Height: {bbox[3]:.4f}")
            print(f"  Angle: {bbox[4]:.4f} (Normalized)")

            # Validate bounding box parameters
            if not (0.0 <= bbox[0] <= 1.0 and 0.0 <= bbox[1] <= 1.0):
                print("  Invalid center coordinates. Skipping this image.\n")
                continue
            if not (0.05 <= bbox[2] <= 1.0 and 0.05 <= bbox[3] <= 1.0):
                print("  Invalid width or height. Skipping this image.\n")
                continue
            if not (0.0 <= bbox[4] <= 1.0):
                print("  Invalid angle. Skipping this image.\n")
                continue

            # Convert tensor to PIL Image and ensure 'RGB' mode
            image_pil = to_pil(image_tensor.cpu()).convert("RGB")

            # Draw the bounding box
            image_pil = draw_bounding_box(image_pil, bbox, color=(255, 0, 0), width=2)

            # Overlay the rotation angle as text
            angle_deg = bbox[4] * 360.0  # Denormalize angle
            angle_text = f"Angle: {angle_deg:.2f}°"
            image_pil = overlay_text(image_pil, angle_text, position=(10, 10), font_size=20, color=(255, 255, 255))

            # Define the filename
            image_filename = f"image_{saved_count + 1}.png"

            # Define the full path
            image_path = os.path.join(save_dir, image_filename)

            try:
                # Save the image
                image_pil.save(image_path)
                print(f"Saved {image_filename} | Angle: {angle_deg:.2f}°\n")
                saved_count += 1

                if saved_count >= num_images:
                    break  # Exit after saving the desired number of images

            except Exception as e:
                print(f"Failed to save {image_filename}: {e}\n")

        if saved_count >= num_images:
            break  # Exit outer loop as well

    print(f"Successfully saved {saved_count} images to '{save_dir}'.")

def test_visualization():
    # Create a dummy image (128x128) with 3 channels (RGB)
    dummy_image = torch.zeros(3, 128, 128)
    dummy_image[0, :, :] = 0.5  # R channel
    dummy_image[1, :, :] = 0.5  # G channel
    dummy_image[2, :, :] = 0.5  # B channel

    # Define a valid bounding box in CWHθ format
    # Center at (0.5, 0.5), width=0.5, height=0.3, angle=0.25 (90 degrees)
    bbox_cwh_theta = [0.5, 0.5, 0.5, 0.3, 0.25]

    # Convert to PIL Image and ensure it's in 'RGB' mode
    image_pil = ToPILImage()(dummy_image).convert("RGB")

    # Draw the bounding box
    image_pil = draw_bounding_box(image_pil, bbox_cwh_theta, color=(255, 0, 0), width=2)

    # Overlay the angle
    angle_deg = bbox_cwh_theta[4] * 360.0  # Denormalize angle
    angle_text = f"Angle: {angle_deg:.2f}°"
    image_pil = overlay_text(image_pil, angle_text, position=(10, 10), font_size=20, color=(255, 255, 255))

    # Display the image
    image_pil.show()

def main():
    # Optionally, run a test visualization first
    print("Running test visualization with known bounding box...")
    test_visualization()
    print("Test visualization completed.\n")

    # Define the directory where preprocessed images will be saved
    save_directory = "saved_images"

    # Define the number of images to save
    number_of_images_to_save = 10

    # Define the transformations as used during training, excluding Normalize since it's handled differently here
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
        # Note: Normalize is already handled in the dataset
    ])


    # Specify the image folder (modify this path as needed)
    image_folder = '../ColoradoProjectilePointdatabase/cropped'  # Update this path based on your dataset location

    # Initialize the dataset with the defined transformations
    dataset = ProjectilePointDataset(image_folder=image_folder, transform=transform)

    # Initialize the DataLoader with batch_size=1 and shuffle=True to get random images
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Call the function to save preprocessed images
    save_preprocessed_images(dataset, dataloader, save_directory, num_images=number_of_images_to_save)

if __name__ == "__main__":
    main()
