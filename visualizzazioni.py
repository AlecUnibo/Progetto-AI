import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, test_loader, idx_to_class, device, model_name="Model", num_images=20):
    model.eval()
    shown = 0
    plt.figure(figsize=(15, 10))  # Dimensione figura

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(len(images)):
                if shown >= num_images:
                    break

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalizzazione
                img = np.clip(img, 0, 1)

                plt.subplot(4, 5, shown + 1)  # 4 righe e 5 colonne
                plt.imshow(img)
                plt.axis('off')

                # Aggiorna il titolo per includere il nome del modello
                true_label = idx_to_class[labels[i].item()]
                pred_label = idx_to_class[preds[i].item()]
                plt.title(
                    f"{model_name}\nTrue: {true_label}\nPred: {pred_label}",
                    color=("green" if true_label == pred_label else "red")
                )

                shown += 1
                if shown >= num_images:
                    break

    plt.tight_layout()
    plt.show()
