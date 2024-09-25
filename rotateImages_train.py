import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.bbox_utils import convert_to_cwh_theta, cwh_theta_to_corners, clip_coordinates
from utils.preprocessing import ProjectilePointDataset, collate_fn
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Import the RotationBBoxModel
from models.rotation_bbox_model import RotationBBoxModel

log_file_path = "training_log.txt"
checkpoint_dir = "models"

# Define IoU function (as above)
def calculate_iou(pred_boxes, target_boxes):
    # [Same as above]
    pred_x_min = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y_min = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x_max = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y_max = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x_min = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y_min = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x_max = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y_max = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x_min = torch.max(pred_x_min, target_x_min)
    inter_y_min = torch.max(pred_y_min, target_y_min)
    inter_x_max = torch.min(pred_x_max, target_x_max)
    inter_y_max = torch.min(pred_y_max, target_y_max)

    inter_width = (inter_x_max - inter_x_min).clamp(min=0)
    inter_height = (inter_y_max - inter_y_min).clamp(min=0)
    inter_area = inter_width * inter_height

    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    target_area = (target_x_max - target_x_min) * (target_y_max - target_y_min)

    union_area = pred_area + target_area - inter_area

    iou = inter_area / union_area.clamp(min=1e-6)

    return iou

# Initialize dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Add normalization if desired
])

dataset = ProjectilePointDataset(image_folder='cropped', transform=transform)

# Split the dataset into training and test sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Initialize model, loss, optimizer
model = RotationBBoxModel()
model.load_state_dict(torch.load("models/rotation_model.pth"))

criterion = nn.MSELoss()  # Example loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10  # Example
best_iou = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    for batch in train_loader:
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

        # Calculate IoU
        with torch.no_grad():
            iou = calculate_iou(outputs, targets)
            running_iou += iou.mean().item()

    average_loss = running_loss / len(train_loader)
    average_iou = running_iou / len(train_loader)

    # Testing phase
    model.eval()
    test_running_iou = 0.0
    test_running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue  # Skip empty batches
            images, angles, bboxes = batch
            images = images.to(device)
            targets = bboxes.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            test_running_loss += loss.item()

            iou = calculate_iou(outputs, targets)
            test_running_iou += iou.mean().item()

    test_average_loss = test_running_loss / len(test_loader)
    test_average_iou = test_running_iou / len(test_loader)

    # Log the results to the training_log.txt file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, IoU: {average_iou:.4f}, "
                       f"Test Loss: {test_average_loss:.4f}, Test IoU: {test_average_iou:.4f}\n")

    # Save the model if it has the best test IoU
    if test_average_iou > best_iou:
        best_iou = test_average_iou
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f"Saved Best Model with Test IoU: {best_iou:.4f}")
