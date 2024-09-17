import os
import random
from PIL import Image

# Directories
originals_dir = "ColoradoProjectilePointdatabase/originals"
main_training_dataset_dir = "ColoradoProjectilePointdatabase/training_dataset"  # Main dataset directory
training_dataset_tmp_dir = "ColoradoProjectilePointdatabase/training_dataset_tmp"  # Temporary directory
training_masks_tmp_dir = "ColoradoProjectilePointdatabase/training_masks_tmp"  # Temporary masks directory

# Ensure the temporary output directories exist
os.makedirs(training_dataset_tmp_dir, exist_ok=True)
os.makedirs(training_masks_tmp_dir, exist_ok=True)

# Get all image files from the originals directory
image_files = [f for f in os.listdir(originals_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Randomly sample 50 images
sampled_images = random.sample(image_files, 25)

# Define the target width
target_width = 2000

for img_file in sampled_images:
    # Check if the image already exists in the main training dataset
    if os.path.exists(os.path.join(main_training_dataset_dir, img_file)):
        print(f"Image {img_file} already exists in the main training dataset. Skipping...")
        continue  # Skip to the next image

    # Load the image from the originals directory
    img_path = os.path.join(originals_dir, img_file)
    img = Image.open(img_path).convert("RGB")

    # crop_box = (760, 0, 3250, 2500)  # Adjust these values for cropping, if necessary
    # img = img.crop(crop_box)  # Crop the image if necessary
    
    # Get the original dimensions
    original_width, original_height = img.size
    
    # Calculate the new height to preserve the aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(target_width * aspect_ratio)
    
    # Resize the image while maintaining the aspect ratio
    resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    # Save the resized image to the temporary training dataset directory
    resized_img.save(os.path.join(training_dataset_tmp_dir, img_file))

    # Save the mask (for now, just save the same image as a mask, adjust this for actual mask processing)
    resized_img.save(os.path.join(training_masks_tmp_dir, img_file))

print(f"Successfully processed and saved 50 images in {training_dataset_tmp_dir} and masks in {training_masks_tmp_dir}")
