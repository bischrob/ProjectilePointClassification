import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models  # For ResNet50

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create ResNet50 model for rotation angle and bounding box prediction
class RotationAndBBoxModel(nn.Module):
    def __init__(self):
        super(RotationAndBBoxModel, self).__init__()
        # Load a pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 4 input channels (RGBA)
        self.model.conv1 = nn.Conv2d(4, self.model.conv1.out_channels, 
                                     kernel_size=self.model.conv1.kernel_size,
                                     stride=self.model.conv1.stride, 
                                     padding=self.model.conv1.padding, 
                                     bias=False)

        # Modify the fully connected layer to output 5 values (1 angle + 4 bbox coordinates)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)

# Load model weights from a specified .pth file
def load_model(model_path):
    model = RotationAndBBoxModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    return model

# Function to preprocess the image (resize, normalize, etc.)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])  # Normalize RGB channels only, leave alpha unchanged
    ])
    
    # Load the image as RGBA (to keep the alpha channel)
    image = Image.open(image_path).convert('RGBA')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor  # Return both original image and tensor

# Function to predict the rotation angle and bounding box
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        angle_pred = outputs[:, 0].item()  # Predicted angle
        bbox_pred = outputs[:, 1:].cpu().numpy().flatten()  # Predicted bounding box
    return angle_pred, bbox_pred

# Function to rotate an image by a given angle and preserve the alpha channel (transparency)
def rotate_image(image, angle):
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image.convert("RGBA")

# Function to plot original and rotated images with bounding box
def plot_images_with_bbox(original_image, rotated_image, bbox, angle):
    plt.figure(figsize=(8, 4))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          linewidth=2, edgecolor='red', facecolor='none'))
    plt.title("Original Image with Bounding Box")
    
    # Rotated Image
    plt.subplot(1, 2, 2)
    plt.imshow(rotated_image)
    plt.title(f"Rotated by {angle:.2f} degrees")
    
    plt.show()

# Main function to load model, preprocess image, and predict rotation
def main(model_path, image_path):
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    original_image, image_tensor = preprocess_image(image_path)
    
    # Predict the rotation angle and bounding box
    predicted_angle, predicted_bbox = predict(model, image_tensor)
    
    print(f"Predicted rotation angle: {predicted_angle:.2f} degrees")
    print(f"Predicted bounding box: {predicted_bbox}")
    
    # Rotate the image based on the predicted angle
    rotated_image = rotate_image(original_image, predicted_angle)
    
    # Convert predicted_bbox to a list if necessary
    predicted_bbox = predicted_bbox.tolist() if isinstance(predicted_bbox, np.ndarray) else predicted_bbox
    
    # Plot the original and rotated images with bounding box
    plot_images_with_bbox(original_image, rotated_image, predicted_bbox, predicted_angle)
    
    return predicted_angle, predicted_bbox

if __name__ == "__main__":
    # Specify paths to the model and the image
    model_path = 'models/rotation_model_object.pth'  # Path to your saved model
    image_path = '5GN191.11899_side-2.png'  # Path to the image you want to process
    
    # Run the main function and get the predicted rotation and bounding box
    angle, bbox = main(model_path, image_path)
