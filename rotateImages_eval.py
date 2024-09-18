import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create ResNet50 model with regression output (for continuous rotation angles), modified for 4-channel input
class ResNet50Model(nn.Module):
    def __init__(self):
        super(ResNet50Model, self).__init__()
        # Load the ResNet-50 model from torchvision
        self.base_model = models.resnet50(pretrained=False)

        # Modify the first convolutional layer to accept 4 input channels (RGBA)
        self.base_model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Initialize weights for the new 4-channel conv layer
        with torch.no_grad():
            self.base_model.conv1.weight[:, :3] = self.base_model.conv1.weight[:, :3]  # Copy the RGB weights
            self.base_model.conv1.weight[:, 3] = self.base_model.conv1.weight[:, 0]  # Initialize the 4th channel (alpha) with the first channel

        # Modify the final fully connected layer to output 1 value for regression (predicting the rotation angle)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)  # 1 output for regression
    
    def forward(self, x):
        return self.base_model(x)

# Load model weights from a specified .pth file
def load_model(model_path):
    model = ResNet50Model().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    return model

# Function to preprocess the image (resize, normalize, etc.)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])  # Normalize RGB channels only, leave alpha unchanged
    ])
    
    # Load the image as RGBA (to keep the alpha channel)
    image = Image.open(image_path).convert('RGBA')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor  # Return both original image and tensor

# Function to predict the rotation angle (regression task)
def predict_rotation_angle(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predicted_angle = model(image_tensor).item()  # Get the predicted rotation angle (single output)
    return predicted_angle

# Function to rotate an image by a given angle and preserve the alpha channel (transparency)
def rotate_image(image, angle):
    # Rotate the image with expand=True (expands canvas to fit rotated image)
    rotated_image = image.rotate(angle, expand=True)
    
    # Ensure the image remains in RGBA mode to preserve transparency
    return rotated_image.convert("RGBA")

# Function to plot original and rotated images
def plot_images(original_image, rotated_image, angle):
    plt.figure(figsize=(8, 4))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    
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
    
    # Predict the rotation angle
    predicted_angle = predict_rotation_angle(model, image_tensor)
    
    print(f"Predicted rotation angle: {predicted_angle:.2f} degrees")
    
    # Rotate the image based on the predicted angle
    rotated_image = rotate_image(original_image, predicted_angle)
    
    # Plot the original and rotated images
    plot_images(original_image, rotated_image, predicted_angle)
    
    return predicted_angle

if __name__ == "__main__":
    # Specify paths to the model and the image
    model_path = 'models/rotation_model.pth'  # Path to your saved model
    image_path = '5GN1876.4_side-2.png'  # Path to the image you want to process
    
    # Run the main function and get the predicted rotation
    rotation = main(model_path, image_path)
