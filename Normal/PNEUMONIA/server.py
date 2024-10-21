import flwr as fl
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import csv
import time

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

class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_file = 'federated_metrics.csv'
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Test Loss", "Test Accuracy"])

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

        # Estrai i parametri dai risultati dei client
        parameters_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Otteniamo i pesi (num_examples) dai risultati
        weights = [fit_res.num_examples for _, fit_res in results]
        total_examples = sum(weights)

        # Esegui l'aggregazione pesata dei parametri
        weighted_params = [param * (weights[0] / total_examples) for param in parameters_list[0]]
        for client_idx, params in enumerate(parameters_list[1:], start=1):
            for i, param in enumerate(params):
                weighted_params[i] += param * (weights[client_idx] / total_examples)

        # Fine della misurazione del tempo
        end_time = time.time()
        aggregation_time = end_time - start_time

        # Carica i parametri aggregati nel modello
        net = PneumoniaCNN()
        params_dict = zip(net.state_dict().keys(), weighted_params)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

        # Valuta il modello sul dataset di test
        test_loss, test_accuracy = test_model(net, self.test_loader)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

        # Salva le metriche nel file CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, test_loss, test_accuracy])

        return fl.common.ndarrays_to_parameters(weighted_params), {}


def main():
    strategy = CustomFedAvg(
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
