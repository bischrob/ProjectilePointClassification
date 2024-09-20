import os
import random
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision import models

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset to dynamically apply random rotations and handle bad images
class ProjectilePointDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGBA')

            # Calculate the bounding box for the original image
            bbox = self.get_bounding_box(image)

            # Apply random rotation dynamically
            angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
            rotated_image = transforms.functional.rotate(image, angle, expand=True)

            # Rotate the bounding box using the same angle
            rotated_bbox = self.rotate_bounding_box(bbox, angle, image.size)

            # Resize the rotated image to a fixed size (e.g., 128x128) for batching
            fixed_size = (128, 128)
            original_size = rotated_image.size
            rotated_image = rotated_image.resize(fixed_size)

            # Scale the bounding box to match the resized image
            scaled_bbox = self.scale_bounding_box(rotated_bbox, original_size, fixed_size)

            # Convert the scaled bounding box corners to [x_min, y_min, x_max, y_max] format
            x_min = scaled_bbox[:, 0].min()
            y_min = scaled_bbox[:, 1].min()
            x_max = scaled_bbox[:, 0].max()
            y_max = scaled_bbox[:, 1].max()
            bbox = [x_min, y_min, x_max, y_max]

            # Apply other transformations (e.g., augmentations)
            if self.transform:
                rotated_image = self.transform(rotated_image)

            return rotated_image, torch.tensor(angle).float(), torch.tensor(bbox, dtype=torch.float32)
        
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # Return None to signal a corrupted or bad image

    def get_bounding_box(self, image):
        # Calculate bounding box based on non-transparent pixels (alpha > 0)
        image_array = np.array(image)
        alpha_channel = image_array[:, :, 3]  # Extract alpha channel
        non_zero_coords = np.argwhere(alpha_channel > 0)  # Get non-zero alpha pixel coordinates
        
        if non_zero_coords.size > 0:
            y_min, x_min = non_zero_coords.min(axis=0)
            y_max, x_max = non_zero_coords.max(axis=0)
            return [x_min, y_min, x_max, y_max]
        else:
            # If no non-transparent pixels are found, return a default bounding box
            return [0, 0, image.size[0], image.size[1]]

    def rotate_bounding_box(self, bbox, angle, image_size):
        width, height = image_size
        angle_rad = np.deg2rad(angle)  # Convert angle to radians

        # Coordinates of the four corners of the bounding box
        corners = np.array([
            [bbox[0], bbox[1]],  # Top-left
            [bbox[2], bbox[1]],  # Top-right
            [bbox[2], bbox[3]],  # Bottom-right
            [bbox[0], bbox[3]]   # Bottom-left
        ])

        # Find the center of the image
        center = np.array([width / 2, height / 2])

        # Translate corners to origin (center of the image)
        translated_corners = corners - center

        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Rotate corners
        rotated_corners = np.dot(translated_corners, rotation_matrix)

        # Translate corners back
        rotated_corners = rotated_corners + center

        return rotated_corners

    def scale_bounding_box(self, rotated_corners, original_size, new_size):
        # Scale bounding box from original size to new size
        orig_w, orig_h = original_size
        new_w, new_h = new_size

        x_scale = new_w / orig_w
        y_scale = new_h / orig_h

        scaled_corners = rotated_corners * np.array([x_scale, y_scale])

        return scaled_corners

# Function to save transformed images with bounding boxes
def save_transformed_images(dataset, save_folder, num_images=10):
    os.makedirs(save_folder, exist_ok=True)

    for i in range(min(num_images, len(dataset))):
        image, angle, bbox = dataset[i]

        # Convert image tensor back to PIL for saving
        image_pil = transforms.ToPILImage()(image)

        # Draw bounding box on image
        draw = ImageDraw.Draw(image_pil)
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Save image with bounding box
        image_path = os.path.join(save_folder, f"transformed_image_{i+1}.png")
        image_pil.save(image_path)
        print(f"Saved image {i+1} with bounding box to {image_path}")

# Example usage
if __name__ == "__main__":
    image_folder = '../ColoradoProjectilePointdatabase/cropped'  # Your image folder
    save_folder = 'saved_images'  # Folder to save transformed images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = ProjectilePointDataset(image_folder=image_folder, transform=transform)

    # Save 10 transformed images with bounding boxes
    save_transformed_images(dataset, save_folder, num_images=10)
