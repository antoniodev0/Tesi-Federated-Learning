import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import tenseal as ts
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import time
import csv

# Carica il contesto TenSEAL
with open("secret_context.pkl", "rb") as f:
    secret_context = pickle.load(f)

context = ts.context_from(secret_context)

# Definizione del modello LSTM
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)  # 'out' ha la forma [batch_size, seq_length, hidden_size]
        # Applichiamo il livello fully connected a ogni timestamp (non solo all'ultimo)
        out = self.fc(out)  # 'out' diventa [batch_size, seq_length, vocab_size]
        return out

# Dataset personalizzato per Shakespeare
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        seq = self.text[idx:idx+self.seq_length]
        target = self.text[idx+1:idx+self.seq_length+1]
        seq_idx = torch.tensor([self.char_to_idx[c] for c in seq])
        target_idx = torch.tensor([self.char_to_idx[c] for c in target])
        return seq_idx, target_idx

# Funzione per caricare i dati e dividere tra i client
def load_shakespeare_data(partition_id, num_partitions=1000, seq_length=100):
    with open('shakespeare.txt', 'r') as file:
        text = file.read()

    dataset = ShakespeareDataset(text, seq_length)

    # Dividi il dataset in base alla partizione
    partition_size = len(dataset) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size
    partition_dataset = random_split(dataset, [partition_size, len(dataset) - partition_size])[0]

    # Dividi il dataset in training e validation
    train_size = int(0.8 * len(partition_dataset))
    val_size = len(partition_dataset) - train_size
    train_dataset, val_dataset = random_split(partition_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, dataset.vocab

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

# Funzioni di training e test
def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            
            # Ottenere l'output dal modello
            outputs = net(inputs)  # Output ha dimensione [batch_size, seq_length, vocab_size]
            
            # Appiattiamo l'output e i target per adattarli a CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(2))  # Da [batch_size, seq_length, vocab_size] a [batch_size * seq_length, vocab_size]
            labels = labels.view(-1)  # Da [batch_size, seq_length] a [batch_size * seq_length]
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            
            # Appiattiamo l'output e i target
            outputs = outputs.view(-1, outputs.size(2))
            labels = labels.view(-1)
            
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    return val_loss / len(testloader), accuracy

# Funzione principale
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="ID of the partition to use")
    args = parser.parse_args()

    trainloader, valloader, vocab = load_shakespeare_data(partition_id=args.partition_id)

    vocab_size = len(vocab)
    embed_size = 64
    hidden_size = 128
    num_layers = 2

    net = ShakespeareLSTM(vocab_size, embed_size, hidden_size, num_layers)

    fl.client.start_client(
        server_address="localhost:8080",
        client=HomomorphicFlowerClient(str(args.partition_id), net, trainloader, valloader).to_client()
    )

if __name__ == "__main__":
    main()
