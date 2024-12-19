from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import PlaceDataset
from torch.utils.data import DataLoader

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Flipping casuale
        transforms.RandomRotation(degrees=15),  # Rotazione casuale
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Alterazione colori
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

def split_data(labels, val_size, test_size, random_state): #accetta direttamente parametri di configurazione tramite test_size e val_size
    train_val_labels, test_labels = train_test_split(
        labels, test_size=test_size, stratify=labels['class'], random_state=random_state
    )
    train_labels, val_labels = train_test_split(
        train_val_labels, test_size=val_size / (1 - test_size), stratify=train_val_labels['class'], random_state=random_state
    )
    return train_labels, val_labels, test_labels

def create_dataloaders(image_dir, train_labels, val_labels, test_labels, train_transform, test_transform, batch_size):
    train_dataset = PlaceDataset(images_dir=image_dir, labels_df=train_labels, transform=train_transform)
    val_dataset = PlaceDataset(images_dir=image_dir, labels_df=val_labels, transform=test_transform)
    test_dataset = PlaceDataset(images_dir=image_dir, labels_df=test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, test_dataset
