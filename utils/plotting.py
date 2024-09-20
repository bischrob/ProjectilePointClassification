# utils/plotting.py

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import numpy as np

def plot_images_with_bbox(original_image, rotated_image, bbox, angle, rotated_bbox=None, ground_truth_bbox=None):
    """
    Plots the original and rotated images side by side with their respective bounding boxes.

    Args:
        original_image (PIL.Image.Image): The original PIL image.
        rotated_image (PIL.Image.Image): The rotated PIL image.
        bbox (array-like): Predicted bounding box corners for the original image [x1, y1, x2, y2, x3, y3, x4, y4].
        angle (float): The rotation angle applied to the original image.
        rotated_bbox (array-like, optional): Predicted bounding box corners for the rotated image.
        ground_truth_bbox (array-like, optional): Ground truth bounding box corners for comparison [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    if bbox is not None and len(bbox) == 8:
        plot_polygon(bbox, color='red', label='Predicted BBox')
    if ground_truth_bbox is not None and len(ground_truth_bbox) == 8:
        plot_polygon(ground_truth_bbox, color='green', linestyle='--', label='Ground Truth BBox')
    plt.title("Original Image")
    plt.axis('off')
    if bbox is not None or ground_truth_bbox is not None:
        plt.legend()
    
    # Plot Rotated Image
    plt.subplot(1, 2, 2)
    plt.imshow(rotated_image)
    if rotated_bbox is not None and len(rotated_bbox) == 8:
        plot_polygon(rotated_bbox, color='blue', label='Predicted Rotated BBox')
    if ground_truth_bbox is not None and len(ground_truth_bbox) == 8:
        # Optionally, rotate ground truth bbox as well for comparison
        rotated_gt_bbox = rotate_bbox_corners(ground_truth_bbox, angle, original_image.size, rotated_image.size)
        plot_polygon(rotated_gt_bbox, color='green', linestyle='--', label='Rotated Ground Truth BBox')
    plt.title(f"Rotated Image by {angle:.2f}°")
    plt.axis('off')
    if rotated_bbox is not None or ground_truth_bbox is not None:
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_polygon(bbox, color='red', linestyle='-', linewidth=2, label=None):
    """
    Plots a polygon on the current matplotlib axis.

    Args:
        bbox (array-like): Bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
        color (str): Color of the polygon edges.
        linestyle (str): Style of the polygon edges.
        linewidth (float): Width of the polygon edges.
        label (str, optional): Label for the polygon (used in legend).
    """
    if len(bbox) != 8:
        print("Invalid bbox length. Expected 8 values representing 4 corners.")
        return
    
    # Reshape bbox to (4, 2)
    corners = np.array(bbox).reshape((4, 2))
    
    # Create a polygon and check validity
    polygon = Polygon(corners)
    if not polygon.is_valid:
        print(f"Invalid polygon: {explain_validity(polygon)}")
        return
    
    x, y = polygon.exterior.xy
    plt.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def rotate_bbox_corners(bbox, angle, original_size, rotated_size):
    """
    Rotates bounding box corners based on the given angle and adjusts for image resizing.

    Args:
        bbox (array-like): Bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
        angle (float): Rotation angle in degrees.
        original_size (tuple): Size of the original image (width, height).
        rotated_size (tuple): Size of the rotated image (width, height).

    Returns:
        np.ndarray: Rotated bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    if len(bbox) != 8:
        raise ValueError("bbox must contain 8 elements representing 4 corners.")
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Original and rotated image centers
    orig_w, orig_h = original_size
    rot_w, rot_h = rotated_size
    orig_center = np.array([orig_w / 2, orig_h / 2])
    rot_center = np.array([rot_w / 2, rot_h / 2])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    # Extract and rotate each corner
    rotated_corners = []
    for i in range(0, 8, 2):
        corner = np.array([bbox[i], bbox[i+1]])
        # Translate to origin
        translated = corner - orig_center
        # Rotate
        rotated = rotation_matrix.dot(translated)
        # Translate to rotated image center
        rotated += rot_center
        rotated_corners.extend(rotated)
    
    return np.array(rotated_corners)


def visualize_iou(pred_bbox, true_bbox):
    """
    Visualizes the Intersection over Union (IoU) between predicted and ground truth bounding boxes.

    Args:
        pred_bbox (array-like): Predicted bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
        true_bbox (array-like): Ground truth bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    plt.figure(figsize=(6, 6))
    
    # Create polygons
    pred_polygon = Polygon(pred_bbox.reshape((4, 2)))
    true_polygon = Polygon(true_bbox.reshape((4, 2)))
    
    # Validate polygons
    if not pred_polygon.is_valid:
        print(f"Invalid predicted polygon: {explain_validity(pred_polygon)}")
        return
    if not true_polygon.is_valid:
        print(f"Invalid ground truth polygon: {explain_validity(true_polygon)}")
        return
    
    # Plot predicted bbox
    x_pred, y_pred = pred_polygon.exterior.xy
    plt.plot(x_pred, y_pred, color='red', linewidth=2, label='Predicted BBox')
    
    # Plot ground truth bbox
    x_true, y_true = true_polygon.exterior.xy
    plt.plot(x_true, y_true, color='green', linewidth=2, linestyle='--', label='Ground Truth BBox')
    
    # Plot intersection
    intersection = pred_polygon.intersection(true_polygon)
    if not intersection.is_empty:
        x_inter, y_inter = intersection.exterior.xy
        plt.fill(x_inter, y_inter, color='blue', alpha=0.5, label='Intersection')
    
    # Plot union
    union = pred_polygon.union(true_polygon)
    if not union.is_empty:
        x_union, y_union = union.exterior.xy
        plt.fill(x_union, y_union, color='yellow', alpha=0.3, label='Union')
    
    plt.title("IoU Visualization")
    plt.axis('equal')
    plt.legend()
    plt.show()
