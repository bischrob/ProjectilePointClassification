# utils/bbox_utils.py

import numpy as np

def order_points(pts):
    """
    Orders points in the following order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): Array of shape (4, 2).

    Returns:
        np.ndarray: Ordered array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points to identify corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # Top-left has the smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right has the largest sum
    rect[1] = pts[np.argmin(diff)]   # Top-right has the smallest difference
    rect[3] = pts[np.argmax(diff)]   # Bottom-left has the largest difference

    return rect

def clip_coordinates(x, y, image_width, image_height):
    """
    Clips the (x, y) coordinates to ensure they lie within the image boundaries.

    Args:
        x (int): X-coordinate.
        y (int): Y-coordinate.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Clipped (x, y) coordinates.
    """
    x_clipped = max(0, min(x, image_width - 1))
    y_clipped = max(0, min(y, image_height - 1))
    return (x_clipped, y_clipped)

def cwh_theta_to_corners(x_center, y_center, width, height, angle_deg, image_size=(128, 128)):
    """
    Converts Center, Width, Height, and Angle (CWHθ) to corner points, adjusting for image coordinate system.

    Args:
        x_center (float): Normalized x-coordinate of the center.
        y_center (float): Normalized y-coordinate of the center.
        width (float): Normalized width of the bounding box.
        height (float): Normalized height of the bounding box.
        angle_deg (float): Normalized angle [0, 1], representing [0, 360] degrees.
        image_size (tuple): Size of the image (width, height).

    Returns:
        list: List of corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    image_width, image_height = image_size
    angle = angle_deg * 360.0  # Denormalize angle to degrees

    # Negate the angle to account for image coordinate system (y increases downward)
    theta = np.deg2rad(-angle)

    # Denormalize center coordinates
    cx = x_center * image_width
    cy = y_center * image_height

    # Denormalize width and height
    w = width * image_width
    h = height * image_height

    # Compute corner points relative to center
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Rotate corners
    rotated_corners = np.dot(corners, rotation_matrix)

    # Translate corners to image coordinates
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy

    # Order the points consistently
    ordered_corners = order_points(rotated_corners)

    # Convert to list of tuples
    corner_points = [tuple(point) for point in ordered_corners]

    return corner_points

def convert_to_cwh_theta(bbox):
    """
    Converts corner points to center, width, height, and angle (CWHθ),
    adjusting for image coordinate system by negating the angle.

    Args:
        bbox (np.ndarray): Scaled bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        np.ndarray: [x_center, y_center, width, height, angle]
    """
    # Extract corner points
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox

    # Compute center
    x_center = (x1 + x3) / 2
    y_center = (y1 + y3) / 2

    # Compute width and height
    width = np.linalg.norm([x2 - x1, y2 - y1])
    height = np.linalg.norm([x4 - x1, y4 - y1])

    # Compute angle (in degrees)
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle = np.degrees(angle_rad)

    # Adjust angle for image coordinate system by negating it
    angle = (-angle) % 360.0  # Ensures angle is within [0, 360)

    return np.array([x_center, y_center, width, height, angle])

def get_bounding_box(image):
    """
    Calculates the bounding box based on non-transparent pixels (alpha > 0).

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        np.ndarray: Bounding box corners [x_min, y_min, x_max, y_max].
    """
    # Convert image to numpy array
    image_array = np.array(image)
    alpha_channel = image_array[:, :, 3]  # Extract alpha channel
    non_zero_coords = np.argwhere(alpha_channel > 0)  # Get non-zero alpha pixel coordinates

    if non_zero_coords.size > 0:
        y_min, x_min = non_zero_coords.min(axis=0)
        y_max, x_max = non_zero_coords.max(axis=0)
        return np.array([x_min, y_min, x_max, y_max])
    else:
        # If no non-transparent pixels are found, return a default bounding box
        return np.array([0, 0, image.size[0], image.size[1]])
    

def rotate_bounding_box(bbox, angle, image_size):
    """
    Rotates the bounding box corners based on the given angle, adjusted for image coordinates.

    Args:
        bbox (np.ndarray): Original bounding box [x_min, y_min, x_max, y_max].
        angle (float): Rotation angle in degrees.
        image_size (tuple): Original image size (width, height).

    Returns:
        np.ndarray: Rotated bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    width, height = image_size
    angle_rad = np.deg2rad(angle)  # Use positive angle

    # Coordinates of the four corners of the bounding box
    corners = np.array([
        [bbox[0], bbox[1]],  # Top-left
        [bbox[2], bbox[1]],  # Top-right
        [bbox[2], bbox[3]],  # Bottom-right
        [bbox[0], bbox[3]]   # Bottom-left
    ])

    # Find the center of the image
    center = np.array([width / 2, height / 2])

    # Adjust rotation matrix for image coordinates
    rotation_matrix = np.array([
        [np.cos(angle_rad), np.sin(angle_rad)],
        [-np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Rotate each corner
    rotated_corners = []
    for corner in corners:
        # Translate corner to origin (center of the image)
        translated = corner - center
        # Rotate using adjusted rotation matrix
        rotated = rotation_matrix.dot(translated)
        # Translate back
        rotated += center
        rotated_corners.extend(rotated)

    return np.array(rotated_corners)

def scale_bounding_box(rotated_bbox, original_size, new_size):
    """
    Scales the bounding box from the original image size to the new size.

    Args:
        rotated_bbox (np.ndarray): Rotated bounding box corners [x1, y1, x2, y2, x3, y3, x4, y4].
        original_size (tuple): Original image size (width, height).
        new_size (tuple): New image size (width, height).

    Returns:
        np.ndarray: Scaled bounding box corners.
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    x_scale = new_w / orig_w
    y_scale = new_h / orig_h

    scaled_bbox = rotated_bbox.copy()
    scaled_bbox[::2] = scaled_bbox[::2] * x_scale  # Scale x coordinates
    scaled_bbox[1::2] = scaled_bbox[1::2] * y_scale  # Scale y coordinates

    return scaled_bbox
