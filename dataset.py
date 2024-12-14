import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class AnimalDataset(Dataset):
    def __init__(self, images_dir, labels_df, transform=None):
        self.images_dir = images_dir
        self.labels = labels_df
        self.transform = transform
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(set(labels_df['class']))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = row['class']
        if self.transform:
            image = self.transform(image)
        # Converte la label in tensore numerico utilizzando class_to_idx
        label_idx = self.class_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

    def get_idx_to_class(self):
        return self.idx_to_class

def create_labels(image_dir, classes):
    data = []
    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(image_dir, class_name)
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                data.append({"filename": os.path.join(class_name, filename), "class": class_name})
    return pd.DataFrame(data)
