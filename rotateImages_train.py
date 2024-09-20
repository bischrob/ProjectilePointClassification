# rotateImages_train.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.bbox_utils import convert_to_cwh_theta, cwh_theta_to_corners, clip_coordinates
from utils.preprocessing import ProjectilePointDataset, collate_fn  # Assuming similar structure
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Import the correct model
from models.rotation_bbox_model import RotationBBoxModel

# Initialize dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Add other transformations if needed
])

dataset = ProjectilePointDataset(image_folder='cropped', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss, optimizer
model = RotationBBoxModel()
criterion = nn.MSELoss()  # Example loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
if not os.path.exists('models'):
    os.makedirs('models')

num_epochs = 10  # Example
log_file_path = 'training_log.txt'

# Open the log file in append mode
with open(log_file_path, 'a') as log_file:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            if batch is None:
                continue  # Skip empty batches
            images, angles, bboxes = batch
            images = images.to(device)
            targets = bboxes.to(device)  # Assuming targets are already normalized

            optimizer.zero_grad()
            outputs = model(images)  # [batch_size, num_classes]

            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Calculate average loss and accuracy for the epoch
        average_loss = running_loss / len(dataloader)
        accuracy = (correct / total) * 100  # Percentage

        # Construct epoch message
        epoch_message = f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%"
        
        # Print epoch results
        print(epoch_message)

        # Log the accuracy and loss to the training_log.txt file
        log_file.write(f"{epoch_message}\n")

        # Save the model every epoch
        model_save_path = os.path.join('models', f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}\n")