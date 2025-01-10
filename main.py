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
import json
from jsonschema import validate, ValidationError
import numpy as np
import random

# Carica lo schema di validazione
with open("config_schema.json", "r") as schema_file:
    config_schema = json.load(schema_file)

# Carica configurazioni da config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Valida la configurazione
try:
    validate(instance=config, schema=config_schema)
    print("Configurazione valida.")
except ValidationError as e:
    print("Errore di validazione nella configurazione:", e)
    exit(1)

# Imposta il seed per la riproducibilità
seed = config["random_state"]
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Per CUDA
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
image_dir = config["image_dir"]
classes = config["classes"]
labels = create_labels(image_dir, classes)

# Analisi del dataset
analyze_dataset(labels)

# Trasformazioni e dataloader
train_transform, test_transform = get_transforms()
train_labels, val_labels, test_labels = split_data(
    labels,
    val_size=config["val_size"],
    test_size=config["test_size"],
    random_state=config["random_state"]
)

train_loader, val_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
    image_dir,
    train_labels,
    val_labels,
    test_labels,
    train_transform,
    test_transform,
    batch_size=config["batch_size"]
)

# Inizializzazione dei modelli
num_classes = len(classes)
resnet_model = ResNet50Model(num_classes).to(device)
efficientnet_model = EfficientNetModel(num_classes).to(device)

# Configurazione perdita e ottimizzatore
criterion = nn.CrossEntropyLoss()

# Configurazione ottimizzatore e scheduler
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=config["learning_rate"])
efficientnet_optimizer = optim.Adam(efficientnet_model.parameters(), lr=config["learning_rate"])

resnet_scheduler = StepLR(resnet_optimizer, step_size=config["step_size"], gamma=config["gamma"])
efficientnet_scheduler = StepLR(efficientnet_optimizer, step_size=config["step_size"], gamma=config["gamma"])

# Training aggiornato con scheduler
train_model(
    resnet_model,
    train_loader,
    val_loader,
    criterion,
    resnet_optimizer,
    device,
    model_name="ResNet50",
    epochs=config["epochs"],
    patience=config["patience"]
)
resnet_scheduler.step()

train_model(
    efficientnet_model,
    train_loader,
    val_loader,
    criterion,
    efficientnet_optimizer,
    device,
    model_name="EfficientNet",
    epochs=config["epochs"],
    patience=config["patience"]
)
efficientnet_scheduler.step()

# Valutazione e visualizzazione
evaluate_model(resnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="ResNet50")
evaluate_model(efficientnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="EfficientNet")

# Visualizzazione delle predizioni per ResNet50
visualize_predictions(resnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="ResNet50")

# Visualizzazione delle predizioni per EfficientNet
visualize_predictions(efficientnet_model, test_loader, train_dataset.get_idx_to_class(), device, model_name="EfficientNet")
