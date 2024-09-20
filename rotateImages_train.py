# rotateImages_train.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.bbox_utils import convert_to_cwh_theta, cwh_theta_to_corners, clip_coordinates
from utils.preprocessing import ProjectilePointDataset, collate_fn  # Assuming similar structure
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import the correct model
from models.rotation_bbox_model import RotationBBoxModel

# Initialize dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Add other transformations if needed
])

dataset = ProjectilePointDataset(image_folder='../ColoradoProjectilePointdatabase/cropped', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss, optimizer
model = RotationBBoxModel()
criterion = nn.MSELoss()  # Example loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10  # Example
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        if batch is None:
            continue  # Skip empty batches
        images, angles, bboxes = batch
        images = images.to(device)
        targets = bboxes.to(device)  # Assuming targets are already normalized

        optimizer.zero_grad()
        outputs = model(images)  # [batch_size, 5]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")
