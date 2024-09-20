# utils/preprocessing.py

import os
import random
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Import bounding box utility functions
from utils.bbox_utils import (
    convert_to_cwh_theta,
    cwh_theta_to_corners,
    clip_coordinates,
    rotate_bounding_box,
    scale_bounding_box,
    get_bounding_box
)


def collate_fn(batch):
    """
    Custom collate function to handle None values and batch images, angles, and bboxes separately.

    Args:
        batch (list): List of tuples containing (image, angle_tensor, bbox_tensor).

    Returns:
        tuple: Batched images, angles, and bounding boxes.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Handle empty batch

    images, angles, bboxes = zip(*batch)
    images = torch.stack(images, dim=0)   # [batch_size, 4, 128, 128]
    angles = torch.stack(angles, dim=0)   # [batch_size, 1]
    bboxes = torch.stack(bboxes, dim=0)   # [batch_size, 5]
    return images, angles, bboxes


from PIL import Image, UnidentifiedImageError

class ProjectilePointDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGBA')  # Ensure 4 channels

            # Calculate the bounding box for the original image
            bbox = get_bounding_box(image)

            # Apply random rotation
            angle = random.uniform(0, 360)
            rotated_image = image.rotate(angle, expand=True)

            # Rotate the bounding box
            rotated_bbox = rotate_bounding_box(bbox, angle, image.size)

            # Resize the image and scale the bounding box
            fixed_size = (128, 128)
            original_size = rotated_image.size
            rotated_image = rotated_image.resize(fixed_size)
            scaled_bbox = scale_bounding_box(rotated_bbox, original_size, fixed_size)
            bbox = scaled_bbox.flatten()

            # **Convert corner points to CWHθ using utility function**
            cwh_theta = convert_to_cwh_theta(scaled_bbox)

            # **Normalize the angle and convert to tensor**
            normalized_angle = cwh_theta[4] / 360.0  # Normalize angle to [0, 1]
            cwh_theta_normalized = np.array([
                cwh_theta[0] / 128.0,  # x_center
                cwh_theta[1] / 128.0,  # y_center
                cwh_theta[2] / 128.0,  # width
                cwh_theta[3] / 128.0,  # height
                normalized_angle        # angle
            ])

            angle_tensor = torch.tensor([normalized_angle], dtype=torch.float32)  # Shape: [1]
            bbox_tensor = torch.tensor(cwh_theta_normalized, dtype=torch.float32)  # Shape: [5]

            # Apply transformations
            if self.transform:
                rotated_image = self.transform(rotated_image)  # Should retain 4 channels

            # Return image, angle, and bbox separately
            return rotated_image, angle_tensor, bbox_tensor

        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None