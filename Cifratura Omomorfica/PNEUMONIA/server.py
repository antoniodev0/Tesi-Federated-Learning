import flwr as fl
from flwr.server.strategy import FedAvg
import tenseal as ts
import numpy as np
import pickle
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)
context = ts.context_from(secret_context)

# Modello PneumoniaCNN come nel client
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Funzione di test
def test_model(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return test_loss / len(testloader), accuracy

class HomomorphicFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_file = 'federated_metrics.csv'
        # Scrive solo le colonne richieste: "Round", "Test Loss", "Test Accuracy"
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Test Loss", "Test Accuracy"])

        # Carica il dataset di test
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_dataset = datasets.ImageFolder('DB/test', transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Inizio della misurazione del tempo
        start_time = time.time()

        encrypted_params = []
        weights = []
        for client, fit_res in results:
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                # Decodifica e collega il contesto
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

        # Calcola il totale degli esempi
        total_examples = sum(weights)
        normalized_weights = [w / total_examples for w in weights]

        # Esegui l'aggregazione pesata dei parametri cifrati
        aggregated_params = [param * normalized_weights[0] for param in encrypted_params[0]]
        for client_params, weight in zip(encrypted_params[1:], normalized_weights[1:]):
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i] * weight

        # Fine della misurazione del tempo
        end_time = time.time()
        aggregation_time = end_time - start_time

        # Decifra i parametri aggregati
        decrypted_params = [np.array(param.decrypt()) for param in aggregated_params]

        # Serializza i parametri aggregati
        aggregated_serialized = [param.serialize() for param in aggregated_params]

        # Decifrare e caricare i parametri nel modello
        net = PneumoniaCNN()
        params_dict = zip(net.state_dict().keys(), decrypted_params)
        state_dict = {k: torch.Tensor(v.reshape(net.state_dict()[k].shape)) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

        # Valuta il modello sul dataset di test
        test_loss, test_accuracy = test_model(net, self.test_loader)
        print(f"Round {server_round} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Aggregation Time: {aggregation_time:.4f} seconds")

        # Salva le metriche nel file CSV con le nuove colonne
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, test_loss, test_accuracy, aggregation_time])

        return fl.common.ndarrays_to_parameters(aggregated_serialized), {}


def main():
    strategy = HomomorphicFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
