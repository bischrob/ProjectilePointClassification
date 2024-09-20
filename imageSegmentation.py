import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageChops, ImageOps
from scipy.ndimage import label, find_objects
import glob
import re
import shutil

def crop_image(img_path, dir_path, probability=0.75):

    # Check if file exists before attempting to delete it
    if os.path.exists(img_path):
        print(f"Processing: {img_path}")
    else:
        print(f"File not found, skipping: {img_path}")
        return None
   
    # Load the segmentation model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the backbone
        encoder_weights=None,           # Do not use ImageNet weights, as we're loading fine-tuned weights
        in_channels=3,                  # Input channels (3 for RGB)
        classes=1                       # Output classes (1 for binary segmentation)
    )
    model.load_state_dict(torch.load("models/resnet34_pointsv5.pth"))
    model.eval()
    
    # Load the image
    img = Image.open(img_path).convert("RGB")
    original_width, original_height = img.size
    resized_width, resized_height = 2048, 2048
    
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
    
    os.makedirs(dir_path, exist_ok=True)

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
        img_name = os.path.join(dir_path, f"{img_base_name}_side-{i+1}.png")
        cropped_img_with_border.save(img_name, format='PNG')
        print(f"Saved: {img_name}")

def correct_image(img_path,
                   cropped_dir = "cropped",
                     originals_dir = "originals",
                       training_dataset_tmp_dir = "training_dataset_tmp",
                         training_masks_tmp_dir = "training_masks_tmp"):
  
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


def remove_side(text):
    # This pattern matches '_side-1' or '_side-2'
    pattern = r'_side-[12]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def delete_images_with_prefix(directory, prefix, extensions=None, dry_run=False, confirm=False):
    """
    Deletes image files in the specified directory that start with the given prefix using glob.

    :param directory: Path to the directory containing the images.
    :param prefix: The prefix string to match at the start of filenames.
    :param extensions: List of image file extensions to consider. If None, defaults to common image extensions.
    :param dry_run: If True, lists the files to be deleted without actually deleting them.
    :param confirm: If True, prompts the user for confirmation before deletion.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Create patterns for each extension
    patterns = [os.path.join(directory, f"{prefix}*{ext}") for ext in extensions]

    # Gather all files matching the patterns
    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(glob.glob(pattern))

    if not files_to_delete:
        print("No matching files found.")
        return

    # Dry run
    if dry_run:
        print("Files to be deleted:")
        for file_path in files_to_delete:
            print(f"- {file_path}")
    else:
        # Confirmation prompt
        if confirm:
            print(f"Are you sure you want to delete {len(files_to_delete)} files? (y/n): ", end='')
            choice = input().lower()
            if choice != 'y':
                print("Deletion canceled.")
                return

        # Proceed with deletion
        deleted_files = 0
        for file_path in files_to_delete:
            # Check if file exists before attempting to delete it
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
            else:
                print(f"File not found, skipping: {file_path}")

        # Summary
        summary = f"Total files deleted: {deleted_files}"
        print(f"\n{summary}")





# originals_dir = r"..\ColoradoProjectilePointdatabase/originals"
# image_files = [f for f in os.listdir(originals_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# image_list = image_files[5001:11220]

# for img_file in image_list:
#     img_path = os.path.join(originals_dir, img_file)
#     crop_image(img_path, probability = .75)

import csv

file_path = '../ColoradoProjectilePointdatabase/bad_points2.txt'
with open(file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')

    for row in reader:
        filename = row[0].strip()
        filename_new = filename + ".png"
        # print(filename_new)
        correct_image(filename_new, 
                       cropped_dir="../ColoradoProjectilePointdatabase/cropped",
                       originals_dir= "../ColoradoProjectilePointdatabase/originals",
                       training_dataset_tmp_dir= "../ColoradoProjectilePointdatabase/training_dataset_tmp",
                       training_masks_tmp_dir= "../ColoradoProjectilePointdatabase/training_masks_tmp")

# fix bad images

file_path = '../ColoradoProjectilePointdatabase/bad_points.txt'
dir = "../ColoradoProjectilePointdatabase/cropped"
dir2 = "../ColoradoProjectilePointdatabase/originals"
with open(file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')

    for row in reader:
        filename = row[0].strip()
        filename = remove_side(filename)
        print(filename)
        delete_images_with_prefix(dir,filename)
        filename_new = filename + ".png"
        # print(filename_new)
        crop_image(os.path.join(dir2,filename_new),dir)