# test_data_pipeline.py

from utils.preprocessing import ProjectilePointDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
])

dataset = ProjectilePointDataset(image_folder='../ColoradoProjectilePointdatabase/cropped', transform=transform)
sample = dataset[0]

if sample:
    image, angle, bbox = sample
    print(f"Image shape: {image.shape}")    # Should be [4, 128, 128]
    print(f"Angle tensor: {angle}")         # Should be a single float tensor
    print(f"BBox tensor: {bbox}")           # Should be [8] tensor
else:
    print("Sample is None.")
