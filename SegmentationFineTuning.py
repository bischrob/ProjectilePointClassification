import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
import random

# Set random seeds for reproducibility
def set_seed(seed=1010):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1010)

# Define image size
IMAGE_SIZE = 2048

# 1. Define the custom dataset with enhanced error handling
class ProjectilePointDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.valid_images = []
        self.valid_masks = []

        # Verify that each image has a corresponding mask
        for img_name in self.images:
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)  # Assuming mask filenames match image filenames
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.valid_images.append(img_name)
                self.valid_masks.append(img_name)
            else:
                print(f"Warning: Missing image or mask for '{img_name}'. This sample will be ignored.")

        print(f"Initialized ProjectilePointDataset with {len(self.valid_images)} samples.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.valid_images[idx])
        mask_path = os.path.join(self.mask_dir, self.valid_masks[idx])

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(mask)  # Resize the mask
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor
            mask = (mask > 0.5).float()  # Ensure the mask is binary (0 or 1)

        return image, mask

# 2. Define transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    # Normalize using ImageNet's mean and std (since we'll use a pre-trained model)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # Normalize using ImageNet's mean and std (since we'll use a pre-trained model)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Function to create DataLoaders with enhanced filtering
def create_dataloaders(tmp_image_dir, tmp_mask_dir, original_image_dir, original_mask_dir,
                      batch_size=8, test_size=0.2, random_state=1010):
    """
    Creates training and testing DataLoaders by combining temporary and original datasets.

    Args:
        tmp_image_dir (str): Directory containing temporary training images.
        tmp_mask_dir (str): Directory containing temporary training masks.
        original_image_dir (str): Directory containing original training images.
        original_mask_dir (str): Directory containing original training masks.
        batch_size (int): Batch size.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        DataLoader, DataLoader: Training and testing DataLoaders.
    """
    # Initialize temporary and original datasets
    tmp_dataset = ProjectilePointDataset(image_dir=tmp_image_dir,
                                         mask_dir=tmp_mask_dir,
                                         transform=train_transform)

    original_dataset = ProjectilePointDataset(image_dir=original_image_dir,
                                              mask_dir=original_mask_dir,
                                              transform=train_transform)

    # Combine the two datasets
    combined_dataset = ConcatDataset([original_dataset, tmp_dataset])
    print(f"Combined dataset contains {len(combined_dataset)} samples.")

    # Split combined dataset into training and testing subsets
    train_size = int((1 - test_size) * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_subset, test_subset = random_split(combined_dataset, [train_size, test_size],
                                            generator=torch.Generator().manual_seed(random_state))
    print(f"Training subset: {len(train_subset)} samples")
    print(f"Testing subset: {len(test_subset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

# 4. Function to define the model
def get_model():
    """
    Initializes the U-Net model with a ResNet-34 encoder and pre-trained weights.

    Returns:
        smp.Unet: Initialized U-Net model.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights="imagenet",     # Use weights pre-trained on ImageNet
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )
    return model

# 5. Function to calculate per-pixel accuracy
def calculate_accuracy(outputs, masks):
    """
    Calculates per-pixel accuracy for segmentation.

    Args:
        outputs (torch.Tensor): Model outputs (logits) with shape (B, 1, H, W).
        masks (torch.Tensor): Ground truth masks with shape (B, 1, H, W).

    Returns:
        float: Per-pixel accuracy.
    """
    preds = torch.sigmoid(outputs) > 0.5  # Thresholding at 0.5
    correct = (preds == masks).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

# 6. Training function with logging
def train_model(tmp_image_dir, tmp_mask_dir, original_image_dir, original_mask_dir,
               batch_size=8, test_size=0.2, random_state=1010,
               num_epochs=30, learning_rate=1e-4,
               log_file_path="segmentationFineTuningResults.txt",
               save_path='models/segmentation_best_model.pth'):
    """
    Trains the U-Net model and logs the training and testing metrics.

    Args:
        tmp_image_dir (str): Directory containing temporary training images.
        tmp_mask_dir (str): Directory containing temporary training masks.
        original_image_dir (str): Directory containing original training images.
        original_mask_dir (str): Directory containing original training masks.
        batch_size (int): Batch size.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        log_file_path (str): Path to the log file.
        save_path (str): Path to save the best model.

    Returns:
        None
    """
    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        tmp_image_dir=tmp_image_dir,
        tmp_mask_dir=tmp_mask_dir,
        original_image_dir=original_image_dir,
        original_mask_dir=original_mask_dir,
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state
    )

    # Initialize the model
    model = get_model()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Define loss function (Binary Cross-Entropy with logits)
    criterion = nn.BCEWithLogitsLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize variables to track the best accuracy
    best_accuracy = 0.0

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open the log file
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch,Train Loss,Train Accuracy,Test Loss,Test Accuracy\n")  # Header

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            total_train_batches = 0

            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss
                train_loss += loss.item()
                
                # Calculate accuracy
                acc = calculate_accuracy(outputs, masks)
                train_accuracy += acc

                total_train_batches += 1

            # Average training loss and accuracy
            avg_train_loss = train_loss / total_train_batches
            avg_train_accuracy = train_accuracy / total_train_batches

            # Evaluation phase
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0
            total_test_batches = 0

            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    # Accumulate loss
                    test_loss += loss.item()

                    # Calculate accuracy
                    acc = calculate_accuracy(outputs, masks)
                    test_accuracy += acc

                    total_test_batches += 1

            # Average testing loss and accuracy
            avg_test_loss = test_loss / total_test_batches
            avg_test_accuracy = test_accuracy / total_test_batches

            # Check if this epoch has the best accuracy so far
            if avg_test_accuracy > best_accuracy:
                best_accuracy = avg_test_accuracy
                torch.save(model.state_dict(), save_path)
                is_best = True
            else:
                is_best = False

            # Log the results
            log_line = f"{epoch+1},{avg_train_loss:.4f},{avg_train_accuracy:.4f},{avg_test_loss:.4f},{avg_test_accuracy:.4f}\n"
            log_file.write(log_line)

            # Optionally, print the epoch results
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {avg_train_accuracy:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Test Acc: {avg_test_accuracy:.4f} "
                  f"{'*BEST*' if is_best else ''}")

    print(f"Training complete. Best model saved at '{save_path}' with Test Accuracy: {best_accuracy:.4f}")
    print(f"Training and evaluation results have been logged to '{log_file_path}'.")

# 7. Main function to encapsulate the training process
if __name__ == '__main__':
    # Define directories
    tmp_image_dir = "training_dataset_tmp"     # Directory containing temporary training images
    tmp_mask_dir = "training_masks_tmp"        # Directory containing temporary training masks
    original_image_dir = "training_dataset"     # Directory containing original training images
    original_mask_dir = "training_masks"        # Directory containing original training masks

    # Define hyperparameters
    batch_size = 8
    test_size = 0.2
    random_state = 1010
    num_epochs = 30
    learning_rate = 1e-4
    log_file_path = "segmentationFineTuningResults.txt"
    save_path = 'models/segmentation_best_model.pth'

    # Start training
    train_model(
        tmp_image_dir=tmp_image_dir,
        tmp_mask_dir=tmp_mask_dir,
        original_image_dir=original_image_dir,
        original_mask_dir=original_mask_dir,
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        log_file_path=log_file_path,
        save_path=save_path
    )
