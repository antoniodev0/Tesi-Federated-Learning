import argparse
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine
import warnings
import numpy as np
import time
import csv

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated")

parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_datasets(partition_id, num_partitions=100):
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    partition_size = len(mnist_train) // num_partitions
    partitions = random_split(mnist_train, [partition_size] * num_partitions)
    
    partition = partitions[partition_id]
    
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_subset, val_subset = random_split(partition, [train_size, val_size])
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=32)
    testloader = DataLoader(mnist_test, batch_size=32)
    
    return trainloader, valloader, testloader

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, partition_id, delta=1e-5):
        self.model = model
        self.train_loader = trainloader
        self.val_loader = valloader
        self.partition_id = partition_id
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=2.0,
            max_grad_norm=1.5,
        )
        self.delta = delta
        self.csv_file = f'client_{partition_id}_metrics.csv'
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Train Time", "Evaluate Time", "Epsilon"])
        self.current_round = 0  # Initialize the round counter

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.current_round += 1  # Increment the round counter
        self.set_parameters(parameters)
        self.model.train()
        start_time = time.time()
        for _ in range(5):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        train_time = time.time() - start_time
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        print(f"Training complete for round {self.current_round}. (ε = {epsilon:.2f}, δ = {self.delta})")

        loss_sum = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item() * len(images)
        average_loss = loss_sum / len(self.train_loader.dataset)
        
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_round, train_time, 0, epsilon])  # 0 for evaluate time, will be updated later
        
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": average_loss, "epsilon": epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        start_time = time.time()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        evaluate_time = time.time() - start_time
        accuracy = correct / total
        avg_loss = loss / len(self.val_loader.dataset)
        
        # Update the evaluation time in the CSV file
        with open(self.csv_file, mode='r') as file:
            lines = list(csv.reader(file))
        lines[-1][2] = str(evaluate_time)  # Update the last row's evaluate time
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lines)
        
        print(f"Evaluation complete for round {self.current_round}. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

def main():
    trainloader, valloader, testloader = load_datasets(args.partition_id)
    model = OptimizedCNN()
    client = FlowerClientWithDP(model, trainloader, valloader, args.partition_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
