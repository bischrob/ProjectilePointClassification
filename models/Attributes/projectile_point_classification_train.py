import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import contextlib
import time
from sklearn.model_selection import train_test_split

def train_and_save_model(class_type, form, include_types, version, num_epochs = 20):
    # Load the Excel file
    data_path = "data/heurist_projectilePoints.csv"
    df = pd.read_csv(data_path)

    if isinstance(form, str):
        form = [form]

    df = df[df['Point Form'].isin(form)].copy()
    # df = df.dropna(subset=['x1']).copy()
    df = df[df[class_type].isin(include_types)].copy()
    df['Media file Name'] = df['objectID'].astype(str) + ".png"

    img_dir = "Attributes/images/ml"
    image_filenames = set(os.listdir(img_dir))
    df_exists = df[df['Media file Name'].isin(image_filenames)].copy()

    counts = df_exists[class_type].value_counts().reset_index()
    counts[f'{class_type}_new'] = counts.apply(lambda row: row[class_type] if row['count'] > 9 else 'other', axis=1)
    counts = counts[[class_type, f'{class_type}_new']]
    df_merged = pd.merge(df_exists, counts, on=class_type, how='left')
    df_merged = df_merged.drop(columns=[class_type])
    df_merged = df_merged.rename(columns={f'{class_type}_new': class_type})
    df_merged = df_merged[~df_merged[class_type].isin(['Unknown', 'other', 'Unclassified'])].copy()
    df_merged = df_merged.dropna(subset=[class_type]).copy()

    # Define the number of rotations for each image to expand the dataset
    ROTATION_ANGLES = [0,45,90,135,180,225,270,315]  # You can adjust rotation angles if needed

    # Create a custom dataset for images and labels with augmentations
    class AugmentedPointClusterDataset(Dataset):
        def __init__(self, df, img_dir, transform=None, rotation_angles=ROTATION_ANGLES):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform
            self.rotation_angles = rotation_angles
            self.label_to_idx = {label: idx for idx, label in enumerate(df[class_type].unique())}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        def __len__(self):
            return len(self.df) * len(self.rotation_angles)

        def __getitem__(self, idx):
            original_idx = idx // len(self.rotation_angles)
            rotation_angle = self.rotation_angles[idx % len(self.rotation_angles)]

            img_name = self.df.iloc[original_idx]["Media file Name"]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = image.rotate(rotation_angle)

            if self.transform:
                image = self.transform(image)
            
            label = self.df.iloc[original_idx][class_type]
            label_idx = self.label_to_idx[label]
            
            return image, label_idx

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Create Dataset and DataLoader with augmentations
    train_df, test_df = train_test_split(df_merged, test_size=0.2, random_state=42)
    train_dataset = AugmentedPointClusterDataset(train_df, img_dir, transform=transform)
    test_dataset = AugmentedPointClusterDataset(test_df, img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Load the EfficientNetV2 model and adjust the classification head
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,  # Change from 3 to 1 channel
        out_channels=model.features[0][0].out_channels,
        kernel_size=model.features[0][0].kernel_size,
        stride=model.features[0][0].stride,
        padding=model.features[0][0].padding,
        bias=False
    )
    num_classes = len(df_merged[class_type].unique())
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # type: ignore
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = num_epochs
    for epoch in range(epochs):
        start_time = time.time()
        
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        elapsed_time = time.time() - start_time
        test_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_accuracy = 100 * correct / total
        test_loss /= len(test_loader)

        epoch_result = f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%,  Test accuracy: {test_accuracy:.2f}%, Time: {elapsed_time:.2f} seconds"
        print(epoch_result)

        # Save the model
        form_string = "_".join(form)
        model_save_path = f"Attributes/data/class_{class_type}_{form_string}_epoch_{epoch + 1}_{version}.pth"
        log_file_path = f"Attributes/data/class_{class_type}_{form_string}_{version}.txt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save the accuracy and loss to a text file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch_result}\n")
        
        scheduler.step()

# usage
data_path = "Attributes/data/heurist_projectilePoints.csv"
df = pd.read_csv(data_path)
df.columns
df['Point Form'].unique()
df['Notch Height'].unique()
class_type = 'Notch Height'
form = 'side-notched'
include_types = ["low","mid","high"]
train_and_save_model(class_type=class_type, form=form, include_types=include_types, version="v2", num_epochs = 10)

df['Base Type'].unique()
class_type = 'Base Type'
form = ['side-notched','triangular']
include_types = ["straight","concave"]
train_and_save_model(class_type=class_type, form=form, include_types=include_types, version="v2", num_epochs = 10)

class_type = 'Point Cluster'
df = df.sort_values(by=class_type).copy()
df[class_type].unique()
form = ['side-notched']
include_types = ['side-notched-1', 'side-notched-2','side-notched-3']
train_and_save_model(class_type=class_type, form=form, include_types=include_types, version="v3", num_epochs = 10)

class_type = 'Point Cluster'
df[class_type].unique()
form = ['triangular']
include_types = ['triangular-1','triangular-2','triangular-3']
train_and_save_model(class_type=class_type, form=form, include_types=include_types, version="v2", num_epochs = 10)