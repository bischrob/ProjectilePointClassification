# models/rotation_bbox_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationBBoxModel(nn.Module):
    def __init__(self):
        super(RotationBBoxModel, self).__init__()
        # Define your convolutional layers
        self.features = nn.Sequential(
            # Convolutional Layer Block 1
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  # Input channels: 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (16, 64, 64)

            # Convolutional Layer Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (32, 32, 32)

            # Convolutional Layer Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: (64, 16, 16)
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # Initialize the fully connected layer
        self._initialize_fc()

    def _initialize_fc(self):
        # Create a dummy input to calculate the in_features
        dummy_input = torch.zeros(1, 4, 128, 128)  # Input shape
        features_output = self.features(dummy_input)
        num_features = features_output.view(1, -1).size(1)
        self.fc = nn.Linear(num_features, 5)  # Output: 5 values per sample

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)  # [batch_size, 5]

        # Apply sigmoid to constrain outputs between 0 and 1
        center_x = torch.sigmoid(x[:, 0])
        center_y = torch.sigmoid(x[:, 1])
        width = torch.sigmoid(x[:, 2])
        height = torch.sigmoid(x[:, 3])
        angle = torch.sigmoid(x[:, 4])  # [0,1]

        # Enforce minimum size to prevent collapsing
        min_size = 0.05  # Example minimum size
        width = width * 0.95 + 0.05  # Scale to [0.05, 1.0]
        height = height * 0.95 + 0.05

        # Combine into a single tensor
        outputs = torch.stack([center_x, center_y, width, height, angle], dim=1)  # [batch_size, 5]
        return outputs

# Test the model
if __name__ == '__main__':
    model = RotationBBoxModel()
    test_input = torch.randn(1, 4, 128, 128)
    output = model(test_input)
    print(f'Output shape: {output.shape}')  # Should be [1, 5]
