import csv
import matplotlib.pyplot as plt
import numpy as np

def read_server_data_from_csv(file_path):
    data = {}
    
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            round_num = int(row['Round'])
            data[round_num] = {
                'Validation Loss': float(row['Test Loss']),
                'Validation Accuracy': float(row['Test Accuracy']) * 100  # Converte accuracy in percentuale
            }
    
    return data

def create_validation_loss_graph(data):
    rounds = sorted(data.keys())  # Prende tutti i round disponibili
    val_loss = [data[round]['Validation Loss'] for round in rounds]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, val_loss, label="Validation Loss", color='red', marker='o')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Validation Loss - Epsilon 18.8')
    plt.gca().invert_yaxis()  # Inverte asse y perché la Loss diminuisce
    
    # Imposta le etichette dei round sull'asse X mantenendo 0-5-10-15-20-25-30
    plt.xticks(np.arange(0, 26, 5))  # Mostra solo i round principali sull'asse X (0, 5, 10, 15, 20, 25, 30)

    plt.tight_layout()
    plt.show()

def create_validation_accuracy_graph(data):
    rounds = sorted(data.keys())  # Prende tutti i round disponibili
    val_acc = [data[round]['Validation Accuracy'] for round in rounds]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, val_acc, label="Validation Accuracy", color='blue', marker='o')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy - Epsilon 18.8')
    plt.ylim(0, 100)  # Limita l'accuratezza tra 0 e 100%
    
    # Imposta le etichette dei round sull'asse X mantenendo 0-5-10-15-20-25-30
    plt.xticks(np.arange(0, 26, 5))  # Mostra solo i round principali sull'asse X (0, 5, 10, 15, 20, 25, 30)

    plt.tight_layout()
    plt.show()

# Leggi i dati dal file CSV
file_path = 'server_metrics.csv'
server_data = read_server_data_from_csv(file_path)

# Crea il grafico per Validation Loss
create_validation_loss_graph(server_data)

# Crea il grafico per Validation Accuracy
create_validation_accuracy_graph(server_data)
