import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Load your image (replace with the correct image path)
# img = Image.open(r"ColoradoProjectilePointdatabase\originals\5MT10991.8638.png").convert("RGB")
# img = Image.open(r"ColoradoProjectilePointdatabase\originals\5SH1458_0016.png").convert("RGB")
# img = Image.open(r"ColoradoProjectilePointdatabase\originals\5_MO_0320100_0333.png").convert("RGB")
img = Image.open(r"ColoradoProjectilePointdatabase\originals\5SM3459.2.png").convert("RGB")


# Crop the image to remove rulers
# crop_box = (760, 0, 3250, 2500)  # Adjust these values
# img = img.crop(crop_box)

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

# Display the original image and segmentation mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)

plt.imshow(output_mask, cmap="gray")
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


#### function ####

def crop_image(img_path, probability=0.66):
    import torch
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    import numpy as np
    import os
    from PIL import Image, ImageChops, ImageOps
    from scipy.ndimage import label, find_objects
    
    # Load the segmentation model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights=None,           # Do not use ImageNet weights, as we're loading fine-tuned weights
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )
    model.load_state_dict(torch.load("ColoradoProjectilePointdatabase/resnet34_pointsv3.pth"))
    model.eval()
    
    # Load the image
    img = Image.open(img_path).convert("RGB")
    original_width, original_height = img.size
    resized_width, resized_height = 1248, 1248
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((resized_width, resized_height)), 
        transforms.ToTensor(),        
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Get the model prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process the output mask
    output = torch.sigmoid(output).squeeze().cpu().numpy()
    output_mask = (output > probability).astype(np.uint8)
    
    # Create directories for saving cropped images
    cropped_dir = "ColoradoProjectilePointdatabase/cropped"
    os.makedirs(cropped_dir, exist_ok=True)

    # Resize the original image and mask using LANCZOS interpolation
    img_resized = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    mask_resized = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    # Label and find the objects in the mask
    labeled_mask, num_features = label(np.array(mask_resized))
    print(f"Number of objects found: {num_features}")
    
    # Calculate scaling factors for coordinates
    x_scale = original_width / resized_width
    y_scale = original_height / resized_height
    objects = find_objects(labeled_mask)

    # Resize the mask back to original size using LANCZOS
    mask_original_size = Image.fromarray((output_mask * 255).astype(np.uint8)).resize((original_width, original_height), Image.Resampling.LANCZOS)

    # Create an alpha channel based on the resized mask
    mask_array = np.array(mask_original_size)
    alpha_channel = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_channel, mode='L')

    # Add alpha channel to the original image
    img_rgba = img.convert("RGBA")
    img_with_alpha = ImageChops.multiply(img_rgba, Image.merge("RGBA", [img_rgba.split()[0], 
                                                                        img_rgba.split()[1], 
                                                                        img_rgba.split()[2], 
                                                                        alpha_image]))

    # Set the border size for cropped images
    border_size = 25

    # Iterate through detected objects and crop them
    for i, obj_slice in enumerate(objects[:2]):
        left = max(0, int(obj_slice[1].start * x_scale) - border_size)
        upper = max(0, int(obj_slice[0].start * y_scale) - border_size)
        right = min(original_width, int(obj_slice[1].stop * x_scale) + border_size)
        lower = min(original_height, int(obj_slice[0].stop * y_scale) + border_size)
        
        # Crop the object with the alpha channel and add a border
        cropped_img = img_with_alpha.crop((left, upper, right, lower))
        cropped_img_with_border = ImageOps.expand(cropped_img, border=border_size, fill=(0, 0, 0, 0))
        
        # Save the cropped image
        img_base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_name = os.path.join(cropped_dir, f"{img_base_name}_side-{i+1}.png")
        cropped_img_with_border.save(img_name, format='PNG')
        print(f"Saved: {img_name}")


originals_dir = r"ColoradoProjectilePointdatabase/originals"
image_files = [f for f in os.listdir(originals_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_list = image_files[3501:4000]

for img_file in image_list:
    img_path = os.path.join(originals_dir, img_file)
    crop_image(img_path, probability = .95)


def correct_image(img_path):
    import os
    import shutil
    cropped_dir = "ColoradoProjectilePointdatabase/cropped"
    originals_dir = "ColoradoProjectilePointdatabase/originals"
    training_dataset_tmp_dir = "ColoradoProjectilePointdatabase/training_dataset_tmp"
    training_masks_tmp_dir = "ColoradoProjectilePointdatabase/training_masks_tmp"

    # Extract the filename from the path (without directory)
    filename = os.path.basename(os.path.join(cropped_dir, img_path))

    # Remove '_side-1' or '_side-2' from the filename
    if "_side-1" in filename:
        base_filename = filename.replace("_side-1", "")
    elif "_side-2" in filename:
        base_filename = filename.replace("_side-2", "")
    else:
        print(f"Error: '{filename}' does not contain '_side-1' or '_side-2'")
        return

    # Remove the file extension from the base filename
    base_filename_no_ext = os.path.splitext(base_filename)[0]

    # Search for the matching file in the 'originals' directory (ignoring file extension)
    original_image = None
    for ext in ['.png', '.jpg', '.jpeg']:
        original_path = os.path.join(originals_dir, base_filename_no_ext + ext)
        if os.path.exists(original_path):
            original_image = original_path
            break
    
    if original_image is None:
        print(f"Error: No matching file found for '{base_filename_no_ext}' in the originals directory")
        return

    # Define the destination paths
    destination_dataset_path = os.path.join(training_dataset_tmp_dir, os.path.basename(original_image))
    destination_mask_path = os.path.join(training_masks_tmp_dir, os.path.basename(original_image))

    # Check if the image already exists in the destination directories to avoid overwriting
    if os.path.exists(destination_dataset_path):
        print(f"File already exists in {training_dataset_tmp_dir}. Skipping {original_image}.")
    else:
        shutil.copy2(original_image, destination_dataset_path)  # Copy to training dataset tmp
        print(f"Copied '{original_image}' to {destination_dataset_path}.")

    if os.path.exists(destination_mask_path):
        print(f"File already exists in {training_masks_tmp_dir}. Skipping {original_image}.")
    else:
        shutil.copy2(original_image, destination_mask_path)     # Copy to training mask tmp
        print(f"Copied '{original_image}' to {destination_mask_path}.")


    crop_image(os.path.join("ColoradoProjectilePointdatabase/originals",'5GN1664.7156.png'), probability = .66)
    correct_image('5_FR_0060101_0008_side-2.png')
    