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

def create_validation_loss_graph_with_grid(data):
    rounds = sorted(data.keys())  # Prende tutti i round disponibili
    val_loss = [data[round]['Validation Loss'] for round in rounds]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, val_loss, label="Validation Loss", color='red', marker='o')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('HE Validation Loss over Rounds - PNEUMONIA')
    
    plt.xticks(np.arange(0, 26, 5))  # Mostra solo i round principali sull'asse X (0, 5, 10, 15, 20, 25, 30)
    
    # Aggiunge una griglia
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_validation_accuracy_graph_with_grid(data):
    rounds = sorted(data.keys())  # Prende tutti i round disponibili
    val_acc = [data[round]['Validation Accuracy'] for round in rounds]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, val_acc, label="Validation Accuracy", color='blue', marker='o')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('HE Validation Accuracy over Rounds - PNEUMONIA')
    plt.ylim(0, 100)  # Limita l'accuratezza tra 0 e 100%
    
    plt.xticks(np.arange(0, 26, 5))  # Mostra solo i round principali sull'asse X (0, 5, 10, 15, 20, 25, 30)

    # Aggiunge una griglia
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Leggi i dati dal file CSV
file_path = 'federated_metrics.csv'
server_data = read_server_data_from_csv(file_path)

# Crea il grafico per Validation Loss con la griglia
create_validation_loss_graph_with_grid(server_data)

# Crea il grafico per Validation Accuracy con la griglia
create_validation_accuracy_graph_with_grid(server_data)
