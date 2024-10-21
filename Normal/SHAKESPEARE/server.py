import flwr as fl
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import csv
import time
# Definizione del modello LSTM (uguale a quello del client)
class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(ShakespeareLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = self.fc(out)
        return out

# Dataset personalizzato per Shakespeare (uguale a quello del client)
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

# Funzione per caricare i dati di test sul server
def load_shakespeare_test_data(seq_length=100):
    with open('shakespeare.txt', 'r') as file:
        text = file.read()

    dataset = ShakespeareDataset(text, seq_length)

    # Utilizziamo una porzione del dataset per il test
    test_size = int(0.1 * len(dataset))
    test_dataset = torch.utils.data.Subset(dataset, range(test_size))

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_loader, dataset.vocab

# Funzione di valutazione
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            outputs = outputs.view(-1, outputs.size(2))
            labels = labels.view(-1)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total
    return val_loss / len(testloader), accuracy

class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csv_file = 'federated_metrics.csv'
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Test Loss", "Test Accuracy"])

        # Carica il dataset di test e il modello sul server
        self.test_loader, self.vocab = load_shakespeare_test_data()
        vocab_size = len(self.vocab)
        embed_size = 64
        hidden_size = 128
        num_layers = 2
        self.net = ShakespeareLSTM(vocab_size, embed_size, hidden_size, num_layers)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Inizio della misurazione del tempo
        start_time = time.time()

        # Estrai i parametri dai risultati dei client
        parameters_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # Calcola il totale degli esempi
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
        params_dict = zip(self.net.state_dict().keys(), weighted_params)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

        # Effettua la valutazione sul dataset di test
        test_loss, test_accuracy = test(self.net, self.test_loader)

        print(f"Round {server_round} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Salva i risultati nel file CSV
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
