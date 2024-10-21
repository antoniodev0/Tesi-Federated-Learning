import flwr as fl
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import time 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = datasets.MNIST('data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return testloader

def test(net, testloader):
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
        self.testloader = load_test_data()

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Inizio della misurazione del tempo
        start_time = time.time()

        # Convert parameters into NumPy arrays
        parameters_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Calculate weights for weighted average
        weights = [fit_res.num_examples for _, fit_res in results]
        total_examples = sum(weights)

        # Perform weighted aggregation of parameters
        weighted_params = []
        for i in range(len(parameters_list[0])):
            weighted_param = parameters_list[0][i] * (weights[0] / total_examples)
            for j in range(1, len(parameters_list)):
                weighted_param += parameters_list[j][i] * (weights[j] / total_examples)
            weighted_params.append(weighted_param)

        # Fine della misurazione del tempo
        end_time = time.time()
        aggregation_time = end_time - start_time

        # Set aggregated parameters on a new model and test it
        net = Net()
        params_dict = zip(net.state_dict().keys(), weighted_params)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

        # Evaluate aggregated model
        test_loss, test_accuracy = test(net, self.testloader)
        print(f"Round {server_round} - Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

        # Log metrics in CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([server_round, test_loss, test_accuracy])

        # Return aggregated parameters to send back to clients
        return fl.common.ndarrays_to_parameters(weighted_params), {}

def main():
    # Define the strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    # Start the server with the custom strategy
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
