import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
from torch.utils.data import DataLoader
import random

size = 2048

class ProjectilePointDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # mask filenames match image filenames

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Ensure that the same transformations are applied to both image and mask
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((size, size))(mask)  # Resize the mask
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor
            mask = (mask > 0.5).float()  # Ensure the mask is binary (0 or 1)

        return image, mask


# Define transformations (resize, convert to tensor, and normalize)
train_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model():
    # Load all problematic images from training_dataset_tmp and training_mask_tmp
    tmp_dataset = ProjectilePointDataset(image_dir="training_dataset_tmp",
                                         mask_dir="training_masks_tmp",
                                         transform=train_transform)

    # Load the full original dataset
    original_dataset = ProjectilePointDataset(image_dir="training_dataset",
                                              mask_dir="training_masks",
                                              transform=train_transform)

    # Total size of the original dataset
    original_dataset_size = len(original_dataset)
    
    # The percentage of the original dataset to be used in each epoch (e.g., 25%)
    fraction_to_use = 0.25
    original_sample_size = int(fraction_to_use * original_dataset_size)

    # Training loop
    num_epochs = 25  # adjustable
    log_file_path = "training_log.txt"  # Log file path

    # Load the pre-trained U-Net model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights="imagenet",     # Use weights pre-trained on ImageNet
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )

    # Loss function (Binary Cross-Entropy + Dice Loss can be used too)
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Open the log file to record the accuracy and epoch
    with open(log_file_path, "a") as log_file:
        for epoch in range(num_epochs):
            # Each epoch, take a new random sample of 25% of the original dataset
            original_indices = random.sample(range(original_dataset_size), original_sample_size)
            original_sampled_dataset = torch.utils.data.Subset(original_dataset, original_indices)

            # Combine the sampled original dataset with the temporary dataset
            combined_dataset = ConcatDataset([tmp_dataset, original_sampled_dataset])

            # Create a DataLoader for the combined dataset
            train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=0)

            model.train()
            running_loss = 0.0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Save the model with epoch number appended to the filename
            model_save_path = f"resnet34_pointsv4_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved: {model_save_path}")

            # Log the accuracy (or loss) for each epoch
            log_file.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}\n")
            log_file.flush()  # Ensure it's written to the file immediately

if __name__ == '__main__':
    # Wrap the training in the main guard to handle multiprocessing on Windows
    train_model()
