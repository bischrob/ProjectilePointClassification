# test training dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset to dynamically apply random rotations and generate labels (angles)
class ProjectilePointDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGBA')

        # Calculate the bounding box for the original image
        bbox = self.get_bounding_box(image)

        # Apply random rotation dynamically
        angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
        rotated_image = transforms.functional.rotate(image, angle)

        # Resize the rotated image to a fixed size (e.g., 128x128) for batching
        fixed_size = (128, 128)
        rotated_image = rotated_image.resize(fixed_size, Image.ANTIALIAS)

        # Rotate bounding box
        rotated_bbox = self.rotate_bounding_box(bbox, angle, image.size)

        # Apply other transformations (e.g., augmentations)
        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image, angle, rotated_bbox  # Return image, angle, and rotated bounding box

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
            return None

    def rotate_bounding_box(self, bbox, angle, image_size):
        if bbox is None:
            return None
        
        width, height = image_size
        angle = np.deg2rad(angle)  # Convert angle to radians

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
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        # Rotate corners
        rotated_corners = np.dot(translated_corners, rotation_matrix)

        # Translate corners back
        rotated_corners = rotated_corners + center

        # Get new bounding box coordinates
        x_min, y_min = rotated_corners.min(axis=0)
        x_max, y_max = rotated_corners.max(axis=0)

        return [int(x_min), int(y_min), int(x_max), int(y_max)]

# Function to visualize the image with the rotated bounding box
def visualize_data(image, bbox, angle):
    # Convert tensor image to numpy array if necessary
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw bounding box
    if bbox is not None:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Rotated by {angle:.2f} degrees")
    plt.show()

# Function to load and visualize the data from the dataset
def visualize_dataset(image_folder, batch_size=8):
    # Define the same transformation as used in the training data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ProjectilePointDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Visualize a few examples
    for images, angles, bboxes in dataloader:
        for i in range(len(images)):
            visualize_data(images[i], bboxes[i], angles[i])

        break  # Show one batch of images

if __name__ == "__main__":
    # Specify the image folder path (replace 'cropped' with your actual path)
    image_folder = '../ColoradoProjectilePointdatabase/cropped'

    # Visualize the dataset with bounding boxes
    visualize_dataset(image_folder)
