import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
import random

# Ensure truncated images are loaded properly
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Custom Dataset for loading images and applying random rotations
class ProjectilePointDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Load the image in RGBA format to keep the alpha channel
            image = Image.open(img_path).convert('RGBA')
        except (IOError, Image.DecompressionBombError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        # Generate a random rotation angle (continuous)
        angle = random.uniform(0, 360)

        # Convert the angle to a bin (0-71), where each bin represents 5 degrees
        angle_bin = int(angle // 5)  # Bins are 0 to 71, each representing a 5-degree increment

        # Rotate the image by the random angle
        image = transforms.functional.rotate(image, angle)

        # Apply transformations (resize, to tensor, etc.)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(angle_bin).long()  # Return the image and its angle bin

# Create EfficientNetV2 model with classification output (72 classes) from PyTorch Hub
class EfficientNetV2Model(nn.Module):
    def __init__(self):
        super(EfficientNetV2Model, self).__init__()
        # Load the EfficientNetV2 model from PyTorch Hub
        self.base_model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnetv2_s', pretrained=False)
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 72)  # 72 bins for 360 degrees
    
    def forward(self, x):
        return self.base_model(x)

# Function to load the last saved model weights
def load_last_checkpoint(model, save_path):
    checkpoint_files = [f for f in os.listdir(save_path) if f.startswith('rotation_model_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        last_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        model_path = os.path.join(save_path, last_checkpoint)
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")
    else:
        print("No previous model found, training from scratch.")

# Training function
def train_model(image_folder, epochs=10, batch_size=32, save_path='models/'):
    try:
        # Create transformations (resize, normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])  # Normalize RGB channels only, leave alpha unchanged
        ])
        
        # Load dataset and split into train and validation sets (80% train, 20% val)
        dataset = ProjectilePointDataset(image_folder, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model, loss function, and optimizer
        model = EfficientNetV2Model().to(device)
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Load the last saved model if available
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            load_last_checkpoint(model, save_path)

        for epoch in range(epochs):
            # Randomly sample 25% of the training data for each epoch
            subset_size = int(0.25 * len(train_dataset))
            subset_indices = np.random.choice(range(len(train_dataset)), subset_size, replace=False)
            subset = torch.utils.data.Subset(train_dataset, subset_indices)
            train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

            # Training
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Skip invalid data
                if inputs is None or labels is None:
                    continue
                
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # CrossEntropyLoss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model and log the epoch results
            model_save_path = os.path.join(save_path, f'rotation_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            
            with open(os.path.join(save_path, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"Epoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

# Train the model with 10 epochs
train_model(image_folder='../ColoradoProjectilePointdatabase/cropped', epochs=50, batch_size=32)
