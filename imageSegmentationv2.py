import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_existing_mask(img_path, mask_dirs):
    """
    Check if the mask exists in any of the provided directories.
    """
    img_base_name = os.path.splitext(os.path.basename(img_path))[0]  # Get the image name without extension
    for mask_dir in mask_dirs:
        for ext in ['.png', '.jpg', '.jpeg']:  # Check for different image extensions
            mask_path = os.path.join(mask_dir, img_base_name + ext)
            if os.path.exists(mask_path):
                print(f"Using existing mask from {mask_path}")
                return Image.open(mask_path)
    return None

# Directories where existing masks might be found
mask_dirs = [
    "ColoradoProjectilePointdatabase/training_masks",
    "ColoradoProjectilePointdatabase/training_masks_tmp"
]

# Load your image (replace with the correct image path)
img_path = r"ColoradoProjectilePointdatabase/originals/5SM3459.2.png"
img = Image.open(img_path).convert("RGB")

# Check if a mask already exists in training_masks or training_masks_tmp
existing_mask = load_existing_mask(img_path, mask_dirs)

if existing_mask is None:
    # No existing mask found, so use the model to generate one

    # Load the U-Net model with a ResNet-34 backbone
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights=None,           # Do not use ImageNet weights, as we're loading fine-tuned weights
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )

    # Load the fine-tuned weights
    model.load_state_dict(torch.load("ColoradoProjectilePointdatabase/resnet34_pointsv3.pth"))

    # Set the model to evaluation mode
    model.eval()

    # Define the necessary transformations
    preprocess = transforms.Compose([
        transforms.Resize((1248, 1248)),  # Resize to the input size expected by the model
        transforms.ToTensor(),            # Convert the image to a tensor
        transforms.Normalize(             # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Apply transformations to the image
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Perform inference using the fine-tuned model
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process the output
    # The output is a logit tensor, so apply a sigmoid for binary segmentation
    output = torch.sigmoid(output).squeeze().cpu().numpy()

    # Threshold the output to get binary mask
    output_mask = (output > 0.5).astype(np.uint8)

    # Convert the mask to an image
    existing_mask = Image.fromarray(output_mask * 255)
else:
    # If the mask already exists, resize it to the appropriate dimensions
    existing_mask = existing_mask.resize((1248, 1248), Image.Resampling.NEAREST)

# Display the original image and segmentation mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(existing_mask, cmap="gray")
plt.title("Segmentation Mask")
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageOps

# Ensure the 'cropped' directory exists
cropped_dir = "ColoradoProjectilePointdatabase/cropped"
os.makedirs(cropped_dir, exist_ok=True)

# Load the original image (replace with your image path)
img = Image.open(r"ColoradoProjectilePointdatabase/originals/5SM3459.2.png").convert("RGB")
original_width, original_height = img.size

# Load the corresponding segmentation mask (already predicted by the model)
# Assume `output_mask` is available from earlier code and is binary
resized_width, resized_height = 1248, 1248

# Resize the image for segmentation (not for final cropping)
img_resized = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

# Resize the segmentation mask to match the resized image dimensions
mask_resized = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((resized_width, resized_height), Image.Resampling.NEAREST)

# Step 1: Find connected components in the resized mask
from scipy.ndimage import label, find_objects
labeled_mask, num_features = label(np.array(mask_resized))
print(f"Number of objects found: {num_features}")

# Step 2: Calculate scaling factors to map the bounding boxes back to the original image
x_scale = original_width / resized_width
y_scale = original_height / resized_height

# Step 3: Get bounding boxes for each connected component
objects = find_objects(labeled_mask)

# Step 4: Apply alpha (transparency) mask on the original image (not resized)
mask_original_size = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((original_width, original_height), Image.Resampling.NEAREST)
mask_array = np.array(mask_original_size)
alpha_channel = np.where(mask_array > 0, 255, 0).astype(np.uint8)  # Create an alpha channel
alpha_image = Image.fromarray(alpha_channel, mode='L')

# Convert the original image to RGBA and apply the alpha channel
img_rgba = img.convert("RGBA")
img_with_alpha = ImageChops.multiply(img_rgba, Image.merge("RGBA", [img_rgba.split()[0], 
                                                                    img_rgba.split()[1], 
                                                                    img_rgba.split()[2], 
                                                                    alpha_image]))

# Step 5: Crop and save the two images based on the bounding boxes, adding a 25-pixel border
border_size = 25  # Define the border size

for i, obj_slice in enumerate(objects):
    # Scale the bounding box back to the original dimensions
    left = max(0, int(obj_slice[1].start * x_scale) - border_size)
    upper = max(0, int(obj_slice[0].start * y_scale) - border_size)
    right = min(original_width, int(obj_slice[1].stop * x_scale) + border_size)
    lower = min(original_height, int(obj_slice[0].stop * y_scale) + border_size)

    # Crop the original image (with alpha channel) using the scaled bounding box
    cropped_img = img_with_alpha.crop((left, upper, right, lower))

    # Remove all pure black pixels (0, 0, 0) by making them fully transparent
    data = np.array(cropped_img)
    tolerance = 10
    transparent_black = (
        (data[:, :, 0] <= tolerance) & 
        (data[:, :, 1] <= tolerance) & 
        (data[:, :, 2] <= tolerance)
    )
    data[transparent_black] = [0, 0, 0, 0]  # Set RGB to 0 and alpha (A) to 0 (transparent)
    cropped_img_no_black = Image.fromarray(data, mode='RGBA')

    # Add a 25-pixel transparent border around the cropped image
    cropped_img_with_border = ImageOps.expand(cropped_img_no_black, border=border_size, fill=(0, 0, 0, 0))  # Transparent border

    # Define the path to save the cropped image
    img_base_name = os.path.splitext(os.path.basename(img_path))[0]
    img_name = os.path.join(cropped_dir, f"{img_base_name}_side-{i+1}.png")

    # Save the cropped image (with border and transparency)
    cropped_img_with_border.save(img_name, format='PNG')
    print(f"Saved: {img_name}")

# Display the original image and segmentation mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_with_alpha)
plt.title("Image with Background Removed")

plt.subplot(1, 2, 2)
plt.imshow(mask_resized, cmap="gray")
plt.title("Resized Segmentation Mask")
plt.show()