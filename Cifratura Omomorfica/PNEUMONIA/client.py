import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import tenseal as ts
import pickle
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import random
import csv
import time

# Carica il contesto TenSEAL
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

# Definizione del modello ottimale e semplice
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

# Client FL con TenSEAL
class HomomorphicFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.csv_file = f'client_{cid}_times.csv'
        self.current_round = 0
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Train Time", "Evaluate Time", "Encryption Time", "Decryption Time", "Serialization Time", "Deserialization Time"])

    def get_parameters(self, config):
        params = [param.cpu().detach().numpy() for param in self.net.parameters()]
        
        encryption_start = time.time()
        encrypted_params = [ts.ckks_vector(context, param.flatten()) for param in params]
        encryption_time = time.time() - encryption_start
        
        serialization_start = time.time()
        serialized_params = [param.serialize() for param in encrypted_params]
        serialization_time = time.time() - serialization_start
        
        self.encryption_time = encryption_time
        self.serialization_time = serialization_time
        
        return serialized_params

    def set_parameters(self, parameters):
        deserialization_start = time.time()
        ckks_vectors = [ts.lazy_ckks_vector_from(param.tobytes()) for param in parameters]
        for vec in ckks_vectors:
            vec.link_context(context)
        deserialization_time = time.time() - deserialization_start
        
        decryption_start = time.time()
        params = [np.array(vec.decrypt()) for vec in ckks_vectors]
        decryption_time = time.time() - decryption_start
        
        self.deserialization_time = deserialization_time
        self.decryption_time = decryption_time

        # Rebuild the state_dict
        params_dict = zip(self.net.state_dict().keys(), params)
        state_dict = {k: torch.Tensor(v.reshape(self.net.state_dict()[k].shape)) for k, v in params_dict}
        
        # Load parameters into the model
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.current_round += 1
        self.set_parameters(parameters)
        start_time = time.time()
        train(self.net, self.trainloader, epochs=3)
        train_time = time.time() - start_time
        self.train_time = train_time
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"partition_id": self.cid}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        start_time = time.time()
        val_loss, accuracy = test(self.net, self.valloader)
        evaluate_time = time.time() - start_time
         # Save all times to CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.current_round, 
                self.train_time, 
                evaluate_time, 
                self.encryption_time, 
                self.decryption_time, 
                self.serialization_time, 
                self.deserialization_time
            ])
        
        return float(val_loss), len(self.valloader.dataset), {"val_loss": float(val_loss), "accuracy": float(accuracy)}

# Funzione di training
def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Funzione di valutazione
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return val_loss / len(testloader), accuracy

# Caricamento del dataset di Pneumonia
def load_balanced_datasets(partition_id, num_partitions=100):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.ImageFolder('DB/train', transform=transform)
    
    # Separare gli indici per classe
    normal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
    pneumonia_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]
    
    # Assicurarsi che ogni partizione abbia un numero uguale di immagini per classe
    samples_per_class = min(len(normal_indices), len(pneumonia_indices)) // num_partitions
    
    start_idx = partition_id * samples_per_class
    end_idx = start_idx + samples_per_class
    
    partition_normal = normal_indices[start_idx:end_idx]
    partition_pneumonia = pneumonia_indices[start_idx:end_idx]
    
    # Combinare e mescolare gli indici
    partition_indices = partition_normal + partition_pneumonia
    random.shuffle(partition_indices)
    
    # Creare il subset per questa partizione
    partition_dataset = Subset(full_dataset, partition_indices)
    
    # Dividere in training e validation set (80% training, 20% validation)
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(partition_dataset, [train_size, val_size])
    
    # Ottenere le etichette per il training set
    train_targets = torch.tensor([full_dataset.targets[i] for i in train_dataset.indices])
    
    # Calcolare i pesi per il sampler
    class_sample_count = torch.bincount(train_targets)
    weight = 1. / class_sample_count.float()
    samples_weight = weight[train_targets]
    
    # Creare il WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader


# Funzione principale
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="ID of the partition to use")
    args = parser.parse_args()

    trainloader, valloader = load_balanced_datasets(partition_id=args.partition_id)

    net = PneumoniaCNN()

    fl.client.start_client(
        server_address="localhost:8080",
        client=HomomorphicFlowerClient(str(args.partition_id), net, trainloader, valloader).to_client()
    )

if __name__ == "__main__":
    main()
