import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_dataset(labels_df):
    
    # Conta il numero di immagini per ogni classe
    class_counts = labels_df['class'].value_counts()

    # Risultati
    print("Distribuzione delle classi:")
    print(class_counts)

    # Grafico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.xlabel("Classi")
    plt.ylabel("Numero di immagini")
    plt.title("Distribuzione delle immagini per classe")
    plt.xticks(rotation=45)
    plt.show()
