import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import timm  # For EfficientNetV2

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Custom Dataset for loading images and applying random rotations
class ProjectilePointDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg'))]
        self.labels = [0] * len(self.image_paths)  # Labels are all zero since images are already correctly oriented

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply random rotation
        angle = random.uniform(0, 360)
        image = transforms.functional.rotate(image, angle)
        
        # Apply transformations (resize, to tensor, etc.)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(angle).float()  # Return image and its rotation angle

# Create EfficientNetV2 model with regression output
class EfficientNetV2Model(nn.Module):
    def __init__(self):
        super(EfficientNetV2Model, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=1)
    
    def forward(self, x):
        return self.base_model(x)

# Training function
def train_model(image_folder, epochs=10, batch_size=32, save_path='models/'):
    try:
        # Create transformations (resize, normalize, etc.)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
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
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
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
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model and log the epoch results
            model_save_path = os.path.join(save_path, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            
            with open(os.path.join(save_path, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"Epoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

# Train the model with 10 epochs
train_model(image_folder='../ColoradoProjectilePointdatabase/cropped', epochs=10, batch_size=32)
