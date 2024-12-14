from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import AnimalDataset
from torch.utils.data import DataLoader


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

def split_data(labels, test_size=0.2, random_state=42):
    return train_test_split(labels, test_size=test_size, stratify=labels['class'], random_state=random_state)

def create_dataloaders(image_dir, train_labels, test_labels, train_transform, test_transform, batch_size=32):
    train_dataset = AnimalDataset(images_dir=image_dir, labels_df=train_labels, transform=train_transform)
    test_dataset = AnimalDataset(images_dir=image_dir, labels_df=test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


