import numpy as np
import torch
import os

def train_model(model, train_loader, test_loader, criterion, optimizer, device, model_name, epochs=50, patience=5):
    if model_name:
        print(f"\nInizio del training di {model_name}\n")
    else:
        print("\nInizio del training\n")

    model.train()
    best_loss = np.inf
    patience_counter = 0
    best_model_path = f"{model_name}_best.pth"  # Salva il modello migliore
    last_model_path = f"{model_name}_last.pth"  # Salva l'ultimo modello

    for epoch in range(epochs):
        # Training
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
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Salvataggio del miglior modello
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Salva lo stato del miglior modello
            torch.save(best_model_state, best_model_path)  # Salva su disco il miglior modello
            print(f"Salvato miglior modello: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Salva l'ultimo modello dopo il training
    last_model_state = model.state_dict()
    torch.save(last_model_state, last_model_path)
    print(f"Salvato ultimo modello: {last_model_path}")

    # Carica lo stato del miglior modello per restituirlo
    model.load_state_dict(best_model_state)
    return model
