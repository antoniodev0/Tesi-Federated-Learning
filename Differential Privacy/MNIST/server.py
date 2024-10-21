import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import time 

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

class FedAvgServer(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_file = 'server_metrics.csv'
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Test Loss", "Test Accuracy", "Total Epsilon", "Aggregation Time"])
        
        # Initialize model and test data
        self.model = OptimizedCNN()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.total_epsilon = None
        self.aggregation_time = 0.0  # Inizializza il tempo di aggregazione

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Start the timing using time.perf_counter()
        start_time = time.perf_counter()

        # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(weights_results)
        # Convert weights to parameters
        parameters_aggregated = self.weights_to_parameters(aggregated_weights)

        # End the timing
        end_time = time.perf_counter()
        self.aggregation_time = end_time - start_time  # Save the aggregation time

        # Collect epsilons from clients
        epsilons = []
        for _, fit_res in results:
            epsilon = fit_res.metrics.get("epsilon", None)
            if epsilon is not None:
                epsilons.append(epsilon)
        # Compute total epsilon
        self.total_epsilon = max(epsilons) if epsilons else None

        print(f"Round {rnd} - Total Epsilon: {self.total_epsilon}, Aggregation Time: {self.aggregation_time:.6f}s")

        return parameters_aggregated, {}

    def evaluate(self, rnd: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Convert parameters to weights
        weights = self.parameters_to_weights(parameters)
        # Load weights into the model
        params_dict = zip(self.model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # Evaluate the model on the test dataset
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * len(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / total
        accuracy = correct / total

        print(f"Round {rnd} - Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Log metrics to CSV, including the aggregation time with higher precision
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                rnd,
                avg_loss,
                accuracy,
                self.total_epsilon,
                f"{self.aggregation_time:.6f}"
            ])

        return avg_loss, {"accuracy": accuracy}


    def weights_to_parameters(self, weights: List[np.ndarray]) -> Parameters:
        return fl.common.ndarrays_to_parameters(weights)

    def parameters_to_weights(self, parameters: Parameters) -> List[np.ndarray]:
        return fl.common.parameters_to_ndarrays(parameters)

    def aggregate_weights(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: List[np.ndarray] = [
            np.sum(layer_updates, axis=0) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

# Create strategy
strategy = FedAvgServer(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=25),
    strategy=strategy
)
