import os
import random
import numpy as np
from PIL import Image, UnidentifiedImageError

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

            # Here is the critical change:
            # DO NOT convert the bounding box to [x_min, y_min, x_max, y_max]
            # Instead, return the scaled corners (rotated and scaled bounding box)
            bbox = scaled_bbox  # This keeps the rotated corners intact

            # Apply other transformations (e.g., augmentations)
            if self.transform:
                rotated_image = self.transform(rotated_image)

            # Return the rotated image, the angle, and the rotated bounding box corners
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

# Model with ResNet50 backbone for rotation and bounding box prediction (modifying the first layer for 4-channel input)
class RotationAndBBoxModel(nn.Module):
    def __init__(self):
        super(RotationAndBBoxModel, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 4 input channels (RGBA)
        self.model.conv1 = nn.Conv2d(4, self.model.conv1.out_channels, 
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride, 
                                     padding=self.model.conv1.padding, 
                                     bias=False)

        # Modify the fully connected layer to output 9 values (1 angle + 8 bbox corner coordinates)
        self.model.fc = nn.Linear(self.model.fc.in_features, 9)

    def forward(self, x):
        return self.model(x)

from shapely.geometry import Polygon

from shapely.geometry import Polygon
from shapely.validation import explain_validity

def calculate_iou(pred_boxes, true_boxes):
    """
    Calculate the IoU (Intersection over Union) between two sets of rotated bounding boxes.
    Both pred_boxes and true_boxes are assumed to be arrays of shape (batch_size, 4, 2)
    representing the 4 corners of the bounding boxes.

    :param pred_boxes: Predicted bounding boxes as corners (batch_size, 4, 2)
    :param true_boxes: Ground truth bounding boxes as corners (batch_size, 4, 2)
    :return: Average IoU across the batch
    """
    pred_boxes = pred_boxes.cpu().detach().numpy()
    true_boxes = true_boxes.cpu().detach().numpy()

    ious = []
    for i in range(len(pred_boxes)):
        # Create polygons for the predicted and true boxes using their corners
        pred_polygon = Polygon(pred_boxes[i])  # Create polygon for predicted box
        true_polygon = Polygon(true_boxes[i])  # Create polygon for true box

        # Validate the polygons
        if not pred_polygon.is_valid:
            print(f"Invalid predicted polygon: {explain_validity(pred_polygon)}")
            continue  # Skip invalid polygons
        if not true_polygon.is_valid:
            print(f"Invalid true polygon: {explain_validity(true_polygon)}")
            continue  # Skip invalid polygons

        # Compute intersection and union areas
        intersection_area = pred_polygon.intersection(true_polygon).area
        union_area = pred_polygon.union(true_polygon).area

        # Compute IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        ious.append(iou)

    return sum(ious) / len(ious) if ious else 0  # Return average IoU, or 0 if none are valid



# Custom collate function to handle None values (bad images)
def collate_fn(batch):
    # Filter out None values (for corrupted or bad images)
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

# Training function with accuracy logging and model saving
def train_model(image_folder, epochs=10, batch_size=16, learning_rate=0.001, log_file='training_log.csv'):
    # Define transformations with data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # Load dataset and DataLoader
    dataset = ProjectilePointDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model, loss function, and optimizer
    model = RotationAndBBoxModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Open log file for writing
    with open(log_file, 'w') as log:
        log.write("Epoch,Train Loss,MAE,IoU\n")  # Header for logging

        # Loop through epochs
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            mae_total = 0
            iou_total = 0

            for batch_idx, (images, angles, bboxes) in enumerate(dataloader):
                images, angles, bboxes = images.to(device), angles.to(device), bboxes.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                angle_preds = outputs[:, 0]  # First output is the predicted angle
                bbox_preds = outputs[:, 1:].view(-1, 4, 2)  # Remaining 8 values are the predicted bounding box corners

                # Compute the loss for both angle prediction and bounding box corners
                angle_loss = criterion(angle_preds, angles)  # Loss for angle prediction
                bbox_loss = criterion(bbox_preds, bboxes)  # Loss for the 8 bounding box corner values
                loss = angle_loss + bbox_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                mae_total += torch.abs(angle_preds - angles).mean().item()
                iou_total += calculate_iou(bbox_preds, bboxes)

                epoch_loss = running_loss / total_samples
                avg_mae = mae_total / len(dataloader)
                avg_iou = iou_total / len(dataloader)

                log.write(f"{epoch+1},{epoch_loss:.4f},{avg_mae:.4f},{avg_iou:.4f}\n")
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, MAE: {avg_mae:.4f}, IoU: {avg_iou:.4f}")

                model_save_path = f"rotate_model_object_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_save_path)

    print("Training complete. Loss and metrics logged to:", log_file)

# Main function to call training
if __name__ == "__main__":
    image_folder = 'cropped'  # Your image folder
    train_model(image_folder=image_folder, epochs=30, batch_size=16, learning_rate=0.001)