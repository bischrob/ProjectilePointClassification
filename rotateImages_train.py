import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import random
import numpy as np
import timm  # For EfficientNet or other pre-trained models
import re

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset to dynamically apply random rotations and generate labels (angles + bounding boxes)
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
            # Try loading the image as RGBA
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

            # Apply other transformations (e.g., augmentations)
            if self.transform:
                rotated_image = self.transform(rotated_image)

            # Convert the four corners of the bounding box to [x_min, y_min, x_max, y_max] format
            x_min, y_min = scaled_bbox.min(axis=0)
            x_max, y_max = scaled_bbox.max(axis=0)
            bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

            return rotated_image, torch.tensor(angle).float(), bbox_tensor

        except (OSError, UnidentifiedImageError) as e:
            # If image loading fails, return None to signal a corrupted/bad image
            print(f"Skipping file {img_path} due to error: {e}")
            return None

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
            # If no non-transparent pixels are found, return a default bounding box (whole image)
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

        # Find center of the image
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

# Data loader wrapper to skip None values (corrupted or failed images)
def collate_fn(batch):
    # Filter out None values
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None, None, None
    images, labels, bboxes = zip(*batch)
    return torch.stack(images), torch.stack(labels), torch.stack(bboxes)

# Create a model, for example, using EfficientNetV2 or ResNet50
class RotationAndBBoxModel(nn.Module):
    def __init__(self):
        super(RotationAndBBoxModel, self).__init__()
        # Here, using EfficientNetV2 without pretrained weights (random initialization)
        self.base_model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=5)  # 1 for angle + 4 for bbox

        # Modify the first convolutional layer to accept 4 input channels (RGBA)
        self.base_model.conv_stem = nn.Conv2d(4, self.base_model.conv_stem.out_channels,
                                              kernel_size=self.base_model.conv_stem.kernel_size,
                                              stride=self.base_model.conv_stem.stride,
                                              padding=self.base_model.conv_stem.padding,
                                              bias=False)

        # Initialize weights for the new 4-channel conv layer by copying the pretrained 3-channel weights
        with torch.no_grad():
            self.base_model.conv_stem.weight[:, :3] = self.base_model.conv_stem.weight[:, :3]  # Copy RGB weights
            self.base_model.conv_stem.weight[:, 3] = self.base_model.conv_stem.weight[:, 0]  # Initialize alpha channel weights

    def forward(self, x):
        return self.base_model(x)

# Training function
def train_model(image_folder, epochs=30, batch_size=32, save_path='models/'):
    try:
        # Define transformations (augmentations, resize, normalize)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip horizontally
            transforms.RandomVerticalFlip(p=0.5),     # Randomly flip vertically
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),   # Random affine transformation
            transforms.Resize((128, 128)),  # Resize to match input size expected by the model
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])  # Normalize the RGBA channels
        ])
        
        # Load dataset with dynamic rotations and transformations
        dataset = ProjectilePointDataset(image_folder, transform=transform)
        
        # Split the dataset into train and validation sets (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Data loaders for train and validation sets with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize the model, loss function, and optimizer
        model = RotationAndBBoxModel().to(device)
        criterion = nn.MSELoss()  # Mean Squared Error for regression task (predicting angle and bbox)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Open log file for appending
        with open("training_log.txt", "a") as log_file:
            for epoch in range(epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                for inputs, labels, bboxes in train_loader:
                    if inputs is None:  # Skip empty batches (all None)
                        continue

                    inputs, labels, bboxes = inputs.to(device), labels.to(device), bboxes.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    angles_pred = outputs[:, 0]  # First output is the predicted angle
                    bbox_pred = outputs[:, 1:]   # Remaining outputs are the bounding box

                    # Compute the losses for angle prediction and bounding box
                    angle_loss = criterion(angles_pred, labels)
                    bbox_loss = criterion(bbox_pred, bboxes)
                    loss = angle_loss + bbox_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)

                # Validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels, bboxes in val_loader:
                        if inputs is None:
                            continue
                        inputs, labels, bboxes = inputs.to(device), labels.to(device), bboxes.to(device)
                        outputs = model(inputs)
                        angles_pred = outputs[:, 0]
                        bbox_pred = outputs[:, 1:]
                        angle_loss = criterion(angles_pred, labels)
                        bbox_loss = criterion(bbox_pred, bboxes)
                        val_loss += (angle_loss + bbox_loss).item()

                val_loss /= len(val_loader)

                # Log to console and file
                log_message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                print(log_message)
                log_file.write(log_message + "\n")  # Write to log file

                # Save the model and log the results for each epoch
                model_save_path = os.path.join(save_path, f'rotate_model_object_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), model_save_path)
    
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")


# Train the model with dynamically rotated images
if __name__ == "__main__":
    image_folder = 'cropped'  # Folder containing your original images
    train_model(image_folder=image_folder, epochs=30, batch_size=32)
