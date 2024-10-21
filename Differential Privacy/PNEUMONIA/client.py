import argparse
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import flwr as fl
from opacus import PrivacyEngine
import warnings
import numpy as np
import time
import csv
import random

warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated")

parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

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

def load_balanced_datasets(partition_id, num_partitions=100):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.ImageFolder('DB/train', transform=transform)

    # Separate indices by class
    normal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
    pneumonia_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]

    # Ensure each partition has an equal number of samples per class
    samples_per_class = min(len(normal_indices), len(pneumonia_indices)) // num_partitions

    start_idx = partition_id * samples_per_class
    end_idx = start_idx + samples_per_class

    partition_normal = normal_indices[start_idx:end_idx]
    partition_pneumonia = pneumonia_indices[start_idx:end_idx]

    # Combine and shuffle indices
    partition_indices = partition_normal + partition_pneumonia
    random.shuffle(partition_indices)

    # Create subset for this partition
    partition_dataset = Subset(full_dataset, partition_indices)

    # Split into training and validation sets (80% training, 20% validation)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(partition_dataset, [train_size, val_size])

    # Get labels for the training set
    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
    train_targets = torch.tensor(train_targets)

    # Calculate weights for the sampler
    class_sample_count = torch.bincount(train_targets)
    weight = 1. / class_sample_count.float()
    samples_weight = weight[train_targets]

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader

class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, partition_id, num_partitions=5, delta=1e-5):
        self.model = model
        self.partition_id = partition_id
        self.train_loader, self.val_loader = load_balanced_datasets(partition_id, num_partitions)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Initialize PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        # Attach PrivacyEngine to the model and optimizer
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=2.0,  # Control the noise
            max_grad_norm=1.5,     # Gradient clipping
        )

        # Delta for differential privacy
        self.delta = delta

        # Initialize CSV file for logging
        self.csv_file = f'client_{partition_id}_metrics.csv'
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Train Time", "Evaluate Time", "Epsilon"])

        self.current_round = 0  # Initialize round counter

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.current_round += 1  # Increment round counter
        self.set_parameters(parameters)
        self.model.train()
        start_time = time.time()
        for _ in range(3):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        train_time = time.time() - start_time

        # Calculate epsilon after training
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
        print(f"Client {self.partition_id} - Training complete for round {self.current_round}. (ε = {epsilon:.2f}, δ = {self.delta})")

        loss_sum = 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item() * len(images)
        average_loss = loss_sum / len(self.train_loader.dataset)

        # Log training time and epsilon
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_round, train_time, 0, epsilon])  # Evaluate time will be updated later

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

        # Update evaluation time in the CSV file
        with open(self.csv_file, mode='r') as file:
            lines = list(csv.reader(file))
        lines[-1][2] = str(evaluate_time)  # Update the last row's evaluate time
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lines)

        print(f"Client {self.partition_id} - Evaluation complete for round {self.current_round}. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

def main():
    # Initialize the model
    model = PneumoniaCNN()

    # Create the FL client with differential privacy
    client = FlowerClientWithDP(model, args.partition_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
