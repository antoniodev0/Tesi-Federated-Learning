import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np
from typing import List, Tuple, Optional
from flwr.common import Parameters
from opacus.layers import DPLSTM  # Import DPLSTM from opacus.layers
import time

# Custom dataset for Shakespeare
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
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

# Function to load 10% of the test dataset
def load_shakespeare_test_data(seq_length=100):
    with open('shakespeare.txt', 'r') as file:
        text = file.read()

    dataset = ShakespeareDataset(text, seq_length)

    # Utilizziamo una porzione del dataset per il test
    test_size = int(0.1 * len(dataset))
    test_dataset = torch.utils.data.Subset(dataset, range(test_size))

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_loader, dataset.vocab

# LSTM model for Shakespeare with DPLSTM
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = DPLSTM(embed_size, hidden_size, num_layers, batch_first=True)  # Use DPLSTM
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = self.fc(out)
        return out

# Custom FedAvg strategy with DP evaluation and CSV logging
class FedAvgServerWithDP(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load test dataset and model on the server
        self.test_loader, self.vocab = load_shakespeare_test_data()
        vocab_size = len(self.vocab)  # Fixed vocabulary size
        embed_size = 64
        hidden_size = 128
        num_layers = 2

        self.model = ShakespeareLSTM(vocab_size, embed_size, hidden_size, num_layers)
        self.csv_file = 'server_metrics.csv'

        # Initialize the CSV file with headers
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Test Loss", "Test Accuracy", "Total Epsilon"])

        self.total_epsilon = None  # Variable for epsilon

    def aggregate_fit(self, rnd, results, failures):

        # Call the super method to perform the actual aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        # Extract epsilons from client results
        epsilons = [fit_res.metrics["epsilon"] for _, fit_res in results if "epsilon" in fit_res.metrics]
        if epsilons:
            self.total_epsilon = max(epsilons)

        # Return only the expected two values
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round, parameters):
        # Convert parameters to weights
        weights = self.parameters_to_weights(parameters)
        params_dict = zip(self.model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        # Load parameters into the model
        self.model.load_state_dict(state_dict, strict=True)

        # Evaluate the model on the test set
        test_loss, test_accuracy = self.test(self.model, self.test_loader)
        print(f"Round {server_round} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Total Epsilon: {self.total_epsilon}")

        # Write results to the CSV file, now including aggregation_time
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, test_loss, test_accuracy, self.total_epsilon])

        return test_loss, {"accuracy": test_accuracy}


    def test(self, model, test_loader):
        criterion = nn.CrossEntropyLoss()
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)

                # Appiattiamo l'output e i target
                outputs = outputs.view(-1, outputs.size(2))
                labels = labels.view(-1)

                test_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        return test_loss / len(test_loader), accuracy

    # Methods for parameter conversion
    def parameters_to_weights(self, parameters: Parameters) -> List[np.ndarray]:
        return fl.common.parameters_to_ndarrays(parameters)

    def weights_to_parameters(self, weights: List[np.ndarray]) -> Parameters:
        return fl.common.ndarrays_to_parameters(weights)

# Main function to start the server
def main():
    strategy = FedAvgServerWithDP(
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
