import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from opacus import PrivacyEngine
import flwr as fl
import csv
import time
from opacus.layers import DPLSTM  # Import DPLSTM for differential privacy
import random

# Parsing arguments for client partition
parser = argparse.ArgumentParser(description='Federated Learning Client with DP')
parser.add_argument('--partition-id', type=int, required=True, help='Partition ID')
args = parser.parse_args()

# LSTM model for Shakespeare with DPLSTM
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = DPLSTM(embed_size, hidden_size, num_layers, batch_first=True)  # Use DPLSTM for DP compatibility
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = self.fc(out)
        return out

# Custom dataset for Shakespeare
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.seq_length = seq_length

        # Use a fixed vocabulary of 256 characters
        self.vocab = [chr(i) for i in range(256)]
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

        # Encode the text using the fixed vocabulary
        self.encoded_text = [self.char_to_idx.get(c, 0) for c in text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        seq = self.encoded_text[idx:idx + self.seq_length]
        target = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        seq_idx = torch.tensor(seq, dtype=torch.long)
        target_idx = torch.tensor(target, dtype=torch.long)
        return seq_idx, target_idx

# Function to load and partition the data among clients
def load_shakespeare_data(partition_id, num_partitions=1000, seq_length=100):
    with open('shakespeare.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    dataset = ShakespeareDataset(text, seq_length)

    # Split the dataset into partitions
    partition_size = len(dataset) // num_partitions
    partitions = [partition_size] * num_partitions
    partitions[-1] += len(dataset) - sum(partitions)  # Adjust the last partition
    partition_dataset = random_split(dataset, partitions)[partition_id]

    # Split into training and validation sets (80% training, 20% validation)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = random_split(partition_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader

# FL client with DP and CSV logging
class FlowerClientWithDP(fl.client.NumPyClient):
    def __init__(self, model, partition_id, delta=1e-5):
        self.model = model
        self.partition_id = partition_id
        self.train_loader, self.val_loader = load_shakespeare_data(partition_id)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Initialize the PrivacyEngine
        self.privacy_engine = PrivacyEngine()

        # Attach the PrivacyEngine to the model and optimizer
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.5,  # Control the noise level
            max_grad_norm=1.5,     # Gradient clipping
        )

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
        for _ in range(3):  # One epoch per round
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        train_time = time.time() - start_time
        epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)

        # Log training time and epsilon
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_round, train_time, 0, epsilon])  # Evaluate time will be updated later

        return self.get_parameters(config), len(self.train_loader.dataset), {"epsilon": epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        start_time = time.time()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)
                loss += self.criterion(outputs, labels).item() * len(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        evaluate_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0
        avg_loss = loss / len(self.val_loader.dataset)

        # Update evaluation time in the CSV file
        with open(self.csv_file, mode='r') as file:
            lines = list(csv.reader(file))
        lines[-1][2] = str(evaluate_time)  # Update the last row's evaluate time
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lines)

        return avg_loss, len(self.val_loader.dataset), {"loss": avg_loss, "accuracy": accuracy}

# Main function to launch the client
def main():
    vocab_size = 256  # Fixed vocabulary size
    model = ShakespeareLSTM(vocab_size=vocab_size, embed_size=64, hidden_size=128, num_layers=2)
    client = FlowerClientWithDP(model, args.partition_id)
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
