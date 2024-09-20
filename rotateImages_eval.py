# evaluate.py

import torch
from torchvision import transforms
from models.rotation_bbox_model import RotationAndBBoxModel
from utils.preprocessing import ProjectilePointDataset, collate_fn
from utils.plotting import plot_images_with_bbox, visualize_iou, rotate_bbox_corners
import os
import numpy as np
from PIL import Image
from shapely import Polygon
from shapely.validation import explain_validity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path):
    """
    Loads the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        RotationAndBBoxModel: The loaded model.
    """
    model = RotationAndBBoxModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    return model

def preprocess_image(image_path):
    """
    Preprocesses the image for model inference.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (original_image, image_tensor)
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0],  # Normalize RGB channels, leave alpha unchanged
            std=[0.229, 0.224, 0.225, 1.0]
        )
    ])

    # Load the image as RGBA (to keep the alpha channel)
    image = Image.open(image_path).convert('RGBA')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

def predict(model, image_tensor):
    """
    Predicts the rotation angle and bounding box using the model.

    Args:
        model (RotationAndBBoxModel): The trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        tuple: (predicted_angle, predicted_bbox)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        angle_pred = outputs[:, 0].item()  # Predicted angle
        bbox_pred = outputs[:, 1:].cpu().numpy().flatten()  # Predicted bounding box corners (8 values)
    return angle_pred, bbox_pred

def rotate_image(image, angle):
    """
    Rotates the image by the given angle while preserving the alpha channel.

    Args:
        image (PIL.Image.Image): The original image.
        angle (float): Rotation angle in degrees.

    Returns:
        PIL.Image.Image: The rotated image.
    """
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image.convert("RGBA")

def main(model_path, image_path, true_bbox_path=None):
    # Load the model
    model = load_model(model_path)

    # Preprocess the image
    original_image, image_tensor = preprocess_image(image_path)

    # Predict the rotation angle and bounding box
    predicted_angle, predicted_bbox = predict(model, image_tensor)

    print(f"Predicted rotation angle: {predicted_angle:.2f} degrees")
    print(f"Predicted bounding box corners: {predicted_bbox}")

    # Rotate the image based on the predicted angle
    rotated_image = rotate_image(original_image, predicted_angle)

    # Optionally, rotate the predicted bounding box for the rotated image
    rotated_bbox = rotate_bbox_corners(predicted_bbox, predicted_angle, original_image.size, rotated_image.size)

    # Plot the original and rotated images with bounding boxes
    if true_bbox_path:
        true_bbox = load_true_bbox(true_bbox_path)
        plot_images_with_bbox(
            original_image, 
            rotated_image, 
            bbox=predicted_bbox, 
            angle=predicted_angle, 
            rotated_bbox=rotated_bbox, 
            ground_truth_bbox=true_bbox
        )
        # Optionally, visualize IoU
        iou = calculate_iou(predicted_bbox, true_bbox)
        visualize_iou(predicted_bbox, true_bbox)
        print(f"IoU with ground truth: {iou:.4f}")
    else:
        plot_images_with_bbox(
            original_image, 
            rotated_image, 
            bbox=predicted_bbox, 
            angle=predicted_angle
        )

    return predicted_angle, predicted_bbox

def load_true_bbox(true_bbox_path):
    """
    Loads the ground truth bounding box from a file.

    Args:
        true_bbox_path (str): Path to the ground truth bbox file.

    Returns:
        np.ndarray: Ground truth bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    # Example: Load from a text file with 8 comma-separated values
    with open(true_bbox_path, 'r') as f:
        bbox_str = f.read().strip()
        bbox = np.array([float(x) for x in bbox_str.split(',')])
    return bbox

def calculate_iou(pred_bbox, true_bbox):
    """
    Calculate the IoU (Intersection over Union) between two bounding boxes.

    Args:
        pred_bbox (np.ndarray): Predicted bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
        true_bbox (np.ndarray): Ground truth bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        float: IoU value.
    """
    pred_polygon = Polygon(pred_bbox.reshape((4, 2)))
    true_polygon = Polygon(true_bbox.reshape((4, 2)))

    if not pred_polygon.is_valid:
        print(f"Invalid predicted polygon: {explain_validity(pred_polygon)}")
        return 0
    if not true_polygon.is_valid:
        print(f"Invalid true polygon: {explain_validity(true_polygon)}")
        return 0

    intersection_area = pred_polygon.intersection(true_polygon).area
    union_area = pred_polygon.union(true_polygon).area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

if __name__ == "__main__":
    # Specify paths to the model and the image
    model_path = 'models/rotate_model_object.pth'  # Path to your saved model
    image_path = '5GN191.11899_side-2.png'  # Path to the image you want to process
    # true_bbox_path = 'path_to_ground_truth_bbox.txt'  # Optional: Path to ground truth bbox

    # Run the main function and get the predicted rotation and bounding box
    angle, bbox = main(model_path, image_path)  # , true_bbox_path=true_bbox_path
