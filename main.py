import torch
from dataset import create_labels
from data_transforms import get_transforms, split_data, create_dataloaders
from modelli import ResNet50Model, EfficientNetModel
from training import train_model
from evaluation import evaluate_model
from visualizzazioni import visualize_predictions
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
image_dir = "archive 4/images"
classes = ["buildings", "mountain","sea", "street"]
labels = create_labels(image_dir, classes)

# Trasformazioni e dataloader
train_transform, test_transform = get_transforms()
train_labels, test_labels = split_data(labels)

train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
    image_dir, train_labels, test_labels, train_transform, test_transform
)

# Inizializzazione dei modelli
num_classes = len(classes)
resnet_model = ResNet50Model(num_classes).to(device)
efficientnet_model = EfficientNetModel(num_classes).to(device)

# Configurazione perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4)
efficientnet_optimizer = optim.Adam(efficientnet_model.parameters(), lr=1e-4)

# Addestramento e valutazione
train_model(resnet_model, train_loader, criterion, resnet_optimizer, device)
train_model(efficientnet_model, train_loader, criterion, efficientnet_optimizer, device)

evaluate_model(resnet_model, test_loader, train_dataset.get_idx_to_class(), device)
evaluate_model(efficientnet_model, test_loader, train_dataset.get_idx_to_class(), device)

visualize_predictions(resnet_model, test_loader, train_dataset.get_idx_to_class(), device)
