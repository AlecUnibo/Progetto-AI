import torch
from dataset import create_labels
from data_transforms import get_transforms, split_data, create_dataloaders
from modelli import ResNet50Model, EfficientNetModel
from training import train_model
from evaluation import evaluate_model
from visualizzazioni import visualize_predictions
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from analyze_dataset import analyze_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
image_dir = "archive 4/images"
classes = ["buildings", "mountain", "sea", "street"]
labels = create_labels(image_dir, classes)

# Analisi del dattaset
analyze_dataset(labels)

# Trasformazioni e dataloader
train_transform, test_transform = get_transforms()
train_labels, val_labels, test_labels = split_data(labels)

train_loader, val_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
    image_dir, train_labels, val_labels, test_labels, train_transform, test_transform
)

# Inizializzazione dei modelli
num_classes = len(classes)
resnet_model = ResNet50Model(num_classes).to(device)
efficientnet_model = EfficientNetModel(num_classes).to(device)

# Configurazione perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()
# Configurazione ottimizzatore e scheduler
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=1e-4)
efficientnet_optimizer = optim.Adam(efficientnet_model.parameters(), lr=1e-4)

resnet_scheduler = StepLR(resnet_optimizer, step_size=5, gamma=0.5)
efficientnet_scheduler = StepLR(efficientnet_optimizer, step_size=5, gamma=0.5)

# Training aggiornato con scheduler
train_model(resnet_model, train_loader, test_loader, criterion, resnet_optimizer, device, model_name="ResNet50")
resnet_scheduler.step()

train_model(efficientnet_model, train_loader, test_loader, criterion, efficientnet_optimizer, device, model_name="EfficientNet")
efficientnet_scheduler.step()

# Valutazione e visualizzazione
evaluate_model(resnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="ResNet50")
evaluate_model(efficientnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="EfficientNet")

# Visualizzazione delle predizioni per ResNet50
visualize_predictions(resnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="ResNet50")

# Visualizzazione delle predizioni per EfficientNet
visualize_predictions(efficientnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="EfficientNet")

