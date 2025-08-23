import pickle
import torch
from dataset import MyDataset
from model import MyModel
from server import Server
from client import Client
from train import train_federated_model
from evaluate import evaluate_model

def main():
    # Configuration
    train_data_path = 'dataset/Train.pkl'
    test_data_path = 'dataset/Test.pkl'
    num_epochs = 200
    total_clients = 15
    num_clients_per_round = 12
    max_acc = 80.0
    classes = [f'A{i}' for i in range(1, 12)]  # Activity labels A1 to A11
    
    # Load data
    print("Loading data...")
    train_data = pickle.load(open(train_data_path, 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))
    
    # Prepare datasets (flatten sensor data)
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    
    for key, data in train_data.items():
        for sensor_data in data[:-1]:  # Exclude label (last element)
            train_inputs.append(sensor_data)  # 3x140x140
            train_labels.append(data[-1])     # Label
    for key, data in test_data.items():
        for sensor_data in data[:-1]:  # Exclude label
            test_inputs.append(sensor_data)   # 3x140x140
            test_labels.append(data[-1])      # Label
    
    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)
    
    # Verify dataset sizes
    print(f"Training combinations: {len(train_data)}")  # Expected: 329
    print(f"Test combinations: {len(test_data)}")      # Expected: 164
    print(f"Training samples (flattened): {len(train_dataset)}")  # Expected: 1645
    print(f"Test samples (flattened): {len(test_dataset)}")      # Expected: 820
    print(f"Sample shape: {train_dataset[0][0].shape}")          # Expected: (3, 140, 140)
    
    # Initialize model
    print("Initializing model...")
    model = MyModel()
    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize server and clients
    print("Initializing server and clients...")
    server = Server(model, test_dataset, num_clients_per_round)
    clients = []
    # Distribute data across clients (by subject)
    subject_keys = {i: [] for i in range(1, 18) if i not in [5, 9]}
    for key in train_data.keys():
        subject = key[0]
        if subject in subject_keys:
            for sensor_data in train_data[key][:-1]:  # Each sensor
                subject_keys[subject].append((sensor_data, train_data[key][-1]))
    
    for c, (subject, data) in enumerate(subject_keys.items()):
        client_inputs = [d[0] for d in data]
        client_labels = [d[1] for d in data]
        client_dataset = MyDataset(client_inputs, client_labels)
        clients.append(Client(model, {c: client_dataset}, id=c))
    
    # Train
    print("Starting training...")
    best_acc = train_federated_model(server, clients, num_epochs, num_clients_per_round, max_acc)
    
    # Evaluate
    print("Evaluating model...")
    model.load_state_dict(torch.load('model.pth'))
    evaluate_model(model, test_dataset, classes)

if __name__ == "__main__":
    main()