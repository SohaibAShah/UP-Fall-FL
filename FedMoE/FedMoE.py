import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

# ==========================================
# 1. Define the Expert and Gating Networks
# ==========================================

class Expert(nn.Module):
    """
    A simple Expert network (MLP). 
    In a real scenario, this could be a CNN or LSTM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    """
    The Router: Decides which expert to use based on input x.
    Standard FedMoE uses a Linear layer + Softmax.
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Logits for each expert
        return self.gate(x)

# ==========================================
# 2. The FedMoE Model Architecture
# ==========================================

class FedMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(FedMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create the pool of experts
        self.experts = nn.ModuleList([
            Expert(input_dim, 64, num_classes) for _ in range(num_experts)
        ])

        # The Gating Network
        self.router = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass with Sparse Top-k Gating.
        """
        batch_size = x.size(0)
        
        # 1. Get Gating Logits
        gate_logits = self.router(x)
        
        # 2. Select Top-k Experts
        # weights: (batch_size, top_k), indices: (batch_size, top_k)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        
        # 3. Normalize weights (Softmax over the top-k)
        weights = F.softmax(weights, dim=1)

        # 4. Forward pass through selected experts
        # We initialize the output as zeros
        final_output = torch.zeros(batch_size, self.experts[0].net[-1].out_features).to(x.device)
        
        # For simulation efficiency, we loop through experts (Masking approach)
        # In optimized cuda code, you would index selectively.
        
        # Auxiliary loss for load balancing
        # (Encourages usage of all experts over the batch)
        # Calculate probability of selection for load balancing
        gate_probs = F.softmax(gate_logits, dim=1)
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i] # The index of the expert for this rank
            weight = weights[:, i].unsqueeze(1) # The weight for this expert
            
            # Loop through the batch to route to specific experts
            # (Note: This loop is slow in Python but explicit for understanding logic)
            # A vectorized approach typically uses masks.
            
            expert_outputs = torch.stack([
                self.experts[idx](x[b].unsqueeze(0)) for b, idx in enumerate(expert_idx)
            ]).squeeze(1)
            
            final_output += weight * expert_outputs

        return final_output, gate_probs

# ==========================================
# 3. Load Balancing Loss
# ==========================================

def load_balancing_loss(gate_probs, num_experts):
    """
    Penalizes the model if it always picks the same expert.
    Standard in MoE papers (Shazeer et al.).
    """
    # importance = sum of probabilities for each expert across the batch
    importance = gate_probs.sum(0)
    # coefficient of variation squared
    loss = (num_experts * torch.sum(importance**2) / (torch.sum(importance)**2))
    return loss

# ==========================================
# 4. Federated Client Logic
# ==========================================

class Client:
    def __init__(self, client_id, dataset, model):
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs=1):
        self.model.train()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        epoch_loss = 0
        for epoch in range(epochs):
            for x, y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, gate_probs = self.model(x)
                
                # Calculate Losses
                cls_loss = F.cross_entropy(outputs, y)
                aux_loss = load_balancing_loss(gate_probs, self.model.num_experts)
                
                # Total loss
                total_loss = cls_loss + 0.01 * aux_loss # 0.01 is the beta hyperparam
                
                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()
        
        return self.model.state_dict(), epoch_loss / len(dataloader)

# ==========================================
# 5. Server Logic (FedAvg)
# ==========================================

def fed_avg(global_model, client_weights):
    """
    Standard FedAvg aggregation. 
    Averages weights from all clients.
    """
    global_dict = global_model.state_dict()
    
    # Initialize with zeros
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()
        
    # Sum up weights
    for w in client_weights:
        for key in global_dict.keys():
            global_dict[key] += w[key]
            
    # Divide by number of clients
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] / len(client_weights)
        
    global_model.load_state_dict(global_dict)
    return global_model

# ==========================================
# 6. Main Simulation Loop
# ==========================================

if __name__ == "__main__":
    print("--- Starting FedMoE Simulation ---")
    
    # 1. Setup Environment
    INPUT_DIM = 20   # e.g., sensor features
    NUM_CLASSES = 5  # e.g., Fall, Walk, Sit...
    NUM_EXPERTS = 4  # Total experts in pool
    TOP_K = 2        # Sparsity
    NUM_CLIENTS = 5
    ROUNDS = 5
    
    # 2. Create Global Model
    global_model = FedMoE(INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K)
    print(f"Global Model Created with {NUM_EXPERTS} experts (Top-{TOP_K} active).")

    # 3. Create Dummy Data (Synthetic)
    # In your real Phase 1, you load UP-Fall here
    clients = []
    for i in range(NUM_CLIENTS):
        # Random data
        X = torch.randn(100, INPUT_DIM) 
        y = torch.randint(0, NUM_CLASSES, (100,))
        dataset = torch.utils.data.TensorDataset(X, y)
        clients.append(Client(i, dataset, global_model))

    # 4. Training Loop
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")
        local_weights = []
        
        # Client Update
        for client in clients:
            # Sync with global model
            client.model.load_state_dict(global_model.state_dict())
            
            # Local Training
            w, loss = client.train(epochs=2)
            local_weights.append(w)
            
            # Simulating Router Collapse Check
            # If input was zero-vector (missing modality), loss would be high here
            print(f"Client {client.client_id} Loss: {loss:.4f}")
            
        # Server Aggregation
        print("Aggregating models...")
        global_model = fed_avg(global_model, local_weights)
        print("Round Complete.")

    print("\nSimulation Finished. This establishes your Baseline B.")