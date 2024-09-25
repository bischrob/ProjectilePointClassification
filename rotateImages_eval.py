# evaluate.py

import torch
from torchvision import transforms
from models.rotation_bbox_model import RotationBBoxModel  # Use the same model as training
from utils.preprocessing import ProjectilePointDataset, collate_fn
from utils.bbox_utils import cwh_theta_to_corners, convert_to_cwh_theta  # Ensure bbox utility functions are imported
from utils.plotting import plot_images_with_bbox, visualize_iou
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
        RotationBBoxModel: The loaded model.
    """
    model = RotationBBoxModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    return model

def rotate_bbox_corners(corners, angle, original_size, new_size):
    """
    Rotates the bounding box corners based on the given angle.

    Args:
        corners (list): A list of 8 values representing 4 corners [x1, y1, x2, y2, x3, y3, x4, y4].
        angle (float): Rotation angle in degrees.
        original_size (tuple): Original image size (width, height).
        new_size (tuple): New image size (width, height) after rotation.

    Returns:
        list: Rotated bounding box corners in the format [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    if len(corners) != 8:
        raise ValueError("bbox must contain 8 elements representing 4 corners.")
    
    # Convert list to NumPy array and reshape it into a 4x2 array
    corner_points = np.array(corners).reshape((4, 2))

    # Find the center of the original image
    original_center = np.array([original_size[0] / 2, original_size[1] / 2])

    # Find the center of the new image (after rotation)
    new_center = np.array([new_size[0] / 2, new_size[1] / 2])

    # Convert angle to radians and calculate rotation matrix
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate the corner points around the original center
    rotated_corners = []
    for point in corner_points:
        # Translate the point to the origin (original center)
        translated_point = point - original_center

        # Rotate the point
        rotated_point = np.dot(rotation_matrix, translated_point)

        # Translate back to the new center
        rotated_point = rotated_point + new_center
        rotated_corners.append(rotated_point)

    # Convert the rotated corners to a flat list and return
    return np.array(rotated_corners).flatten().tolist()


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
    ])

    # Load the image as RGBA (to keep the alpha channel)
    image = Image.open(image_path).convert('RGBA')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

def predict(model, image_tensor):
    """
    Predicts the rotation angle and bounding box using the model.

    Args:
        model (RotationBBoxModel): The trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.

    Returns:
        tuple: (predicted_angle, predicted_bbox)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)

        # Extract predicted angle and bounding box components
        x_center = outputs[:, 1].item()
        y_center = outputs[:, 2].item()
        width = outputs[:, 3].item()
        height = outputs[:, 4].item()
        normalized_angle_pred = outputs[:, 0].item()  # Predicted angle in [0, 1]
        angle_pred = normalized_angle_pred * 360

        # Convert the CWHθ format to corner points
        bbox_pred = cwh_theta_to_corners(x_center, y_center, width, height, angle_pred)

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

    # Flatten the list of tuples to a list of 8 elements
    flattened_bbox = [coord for point in predicted_bbox for coord in point]

    # Rotate the image based on the predicted angle
    rotated_image = rotate_image(original_image, predicted_angle)

    # Optionally, rotate the predicted bounding box for the rotated image
    rotated_bbox = rotate_bbox_corners(flattened_bbox, predicted_angle, original_image.size, rotated_image.size)

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
    model_path = 'models/rotation_bbox_model.pth'  # Path to your saved model
    image_path = '5GN1876.4_side-2.png'  # Path to the image you want to process

    # Run the main function and get the predicted rotation and bounding box
    angle, bbox = main(model_path, image_path)
