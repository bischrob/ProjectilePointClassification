import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import random
import timm  # For EfficientNet or other pre-trained models

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset to dynamically apply random rotations and generate labels (angles)
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
            
            # Apply random rotation dynamically
            angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
            rotated_image = transforms.functional.rotate(image, angle)
            
            # Apply other transformations (e.g., resize, tensor conversion, normalization)
            if self.transform:
                rotated_image = self.transform(rotated_image)
            
            return rotated_image, torch.tensor(angle).float()  # Return the image and its rotation angle as label
        
        except (OSError, UnidentifiedImageError) as e:
            # Handle the truncated image or other loading errors
            print(f"Skipping file {img_path} due to error: {e}")
            return None  # Return None to indicate failure

# Data loader wrapper to skip None values (corrupted or failed images)
def collate_fn(batch):
    # Filter out None values
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

# Create a model, for example, using EfficientNetV2 or ResNet50
class RotationModel(nn.Module):
    def __init__(self):
        super(RotationModel, self).__init__()
        # Here, using EfficientNetV2 without pretrained weights (random initialization)
        self.base_model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=1)  # 1 output for regression (angle)

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
def train_model(image_folder, epochs=10, batch_size=32, save_path='models/'):
    try:
        # Define transformations (resize, normalize)
        transform = transforms.Compose([
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
        model = RotationModel().to(device)
        criterion = nn.MSELoss()  # Mean Squared Error for regression task (predicting angle)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                if inputs is None:  # Skip empty batches (all None)
                    continue

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if inputs is None:
                        continue
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save the model and log the results for each epoch
            model_save_path = os.path.join(save_path, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_save_path)
            
            with open(os.path.join(save_path, 'training_log.txt'), 'a') as log_file:
                log_file.write(f"Epoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

# Train the model with dynamically rotated images
if __name__ == "__main__":
    image_folder = 'cropped'  # Folder containing your original images
    train_model(image_folder=image_folder, epochs=10, batch_size=32)
