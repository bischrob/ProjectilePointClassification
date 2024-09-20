import os
from PIL import Image, UnidentifiedImageError

def validate_images(image_folder):
    """
    Check all images in the folder to ensure they are valid and in RGBA format.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        None
    """
    valid_images = []
    invalid_images = []
    non_rgba_images = []

    # List all files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:
        print(f"No image files found in the folder: {image_folder}")
        return

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            # Open the image using Pillow
            with Image.open(image_path) as img:
                # Check if the image is in RGBA format
                if img.mode == 'RGBA':
                    valid_images.append(image_file)
                else:
                    non_rgba_images.append(image_file)

        except (UnidentifiedImageError, OSError) as e:
            # Image is invalid or unreadable
            print(f"Error with image {image_file}: {e}")
            invalid_images.append(image_file)

    # Print summary
    print(f"\nSummary of image validation in folder: {image_folder}")
    print(f"Total images checked: {len(image_files)}")
    print(f"Valid RGBA images: {len(valid_images)}")
    print(f"Non-RGBA images: {len(non_rgba_images)}")
    print(f"Invalid/Unreadable images: {len(invalid_images)}")

    if non_rgba_images:
        print("\nNon-RGBA images:")
        for img in non_rgba_images:
            print(f"- {img}")

    if invalid_images:
        print("\nInvalid/Unreadable images:")
        for img in invalid_images:
            print(f"- {img}")

if __name__ == '__main__':
    folder_path = input("Enter the path to the image folder: ")
    validate_images(folder_path)
