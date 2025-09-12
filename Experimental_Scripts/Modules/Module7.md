### Module 7 — Personalization: adapters and online ELM heads (tie‑in to [8])
Intuition
- People differ. Keep a global backbone for shared invariants, and let each client own a tiny personalized component that adapts quickly. Online ELM (Extreme Learning Machine) is a simple, fast head for on‑device updates.

Key equations
- Parameter split: θ = (θ_g, φ_i), with θ_g global and φ_i client‑specific.
- Personalized objective: minimize F_i(θ_g, φ_i) + λ||φ_i||^2 subject to secure aggregation of updates to θ_g only.
- ELM head (closed‑form ridge regression): given hidden features H ∈ R^{N×H}, β_i = (H^T H + λI)^{-1} H^T Y.
- Online update (recursive): β_{t+1} = β_t + P_t h_t (y_t − h_t^T β_t), with P_t maintained via Sherman–Morrison.

Mini‑example
- Freeze the multimodal trunk; learn a per‑client 1–2 layer adapter or an ELM head on hidden features. Update φ_i nightly on a few misclassified windows.

Small assignment
- Implement per‑client adapters (few thousand params) and compare to a shared‑only model on worst‑client F1.
- Implement an ELM head and show it recovers performance after a simulated gait change (concept drift).
- Stretch: restrict uploads so only θ_g updates are shared, keeping φ_i private. [8]


Certainly! Here’s a **comprehensive and easy-to-understand explanation** of your Module7.py code, covering its **purpose, structure, key concepts, input/output, and suggestions for improvement**.

---

## **Overall Purpose**

The main goal of this code is to **implement and evaluate personalized federated learning for fall detection** using IMU data. It does this by:
- Training a **shared global model (backbone)** across all clients.
- Allowing each client to have a **personalized component** (adapter or ELM head) that adapts to their unique data.
- Demonstrating how **personalization** improves worst-case client performance and enables fast adaptation to concept drift (e.g., changes in a user's gait).

---

## **Line-by-Line Breakdown**

### **Imports and Setup**

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import logging
import sys
import copy
```
- **Imports** all necessary libraries for data handling, neural networks, federated learning, and logging.

---

### **1. Setup and Helper Functions**

#### **Logging**
```python
def setup_logging(log_dir):
    ...
```
- Sets up logging to both file and console for experiment traceability.

#### **Seeding**
```python
def set_seed(seed=42):
    ...
```
- Ensures reproducibility by setting random seeds for numpy and torch.

---

### **2. Data Loading**

#### **Loading and Partitioning Data**
```python
def load_and_create_clients(data_dir):
    ...
    return clients, test_loader, len(feature_cols)
```
- Loads sensor data, cleans and scales it.
- Splits data by subject: some for training (clients), some for testing (global test set).
- Returns a dictionary of client datasets, a test DataLoader, and the number of features.

#### **Simulating Concept Drift**
```python
def simulate_concept_drift(client_data):
    ...
    return (X_csv_drifted, y)
```
- Simulates a change in a client’s data distribution (e.g., a user’s gait changes) by swapping two feature columns in the second half of the data.

---

### **3. Model Architectures**

#### **Shared Backbone**
```python
class SharedBackbone(nn.Module):
    ...
```
- A 1D-CNN that encodes IMU features into a shared representation (feature vector).

#### **Adapter**
```python
class Adapter(nn.Module):
    ...
```
- A small neural network (bottleneck) that adapts the shared features for each client.

#### **Personalized Adapter Model**
```python
class PersonalizedAdapterModel(nn.Module):
    ...
```
- Combines the shared backbone and the client’s adapter, followed by a classifier.

#### **ELM Head**
```python
class ELMHead:
    ...
```
- Implements an **Extreme Learning Machine**: a fast, closed-form, single-layer neural network head for quick adaptation.
- `train`: Solves for output weights using ridge regression.
- `predict`: Makes predictions using the learned weights.

---

### **4. Federated Learning Components**

#### **Client**
```python
class Client:
    ...
```
- Each client has:
  - Its own data loader.
  - A personal adapter and ELM head.
- `train_adapter`: Trains only the adapter and classifier (not the backbone) on local data.

#### **Model Evaluation**
```python
def evaluate_model(model, loader, device):
    ...
    return f1_score(all_labels, all_preds, pos_label=0, zero_division=0.0)
```
- Evaluates a model on a given dataset and returns the F1-score for the "Fall" class.

---

### **5. Main Simulation Runner**

#### **Federated Learning with Adapters**
```python
def run_fl_simulation_for_adapters(clients_data, test_loader, config, num_features):
    ...
    return min(all_client_f1s) if all_client_f1s else 0.0
```
- Simulates federated learning:
  - Each client trains its adapter locally.
  - The server aggregates (averages) the backbone weights from all clients.
  - After several rounds, evaluates the worst F1-score across all clients (worst-case personalization).

---

### **6. Main Execution Block**

```python
if __name__ == "__main__":
    ...
```
- **Sets up logging and config.**
- **Loads data and creates clients.**
- **Assignment 1:**
  - Runs federated learning with adapters.
  - Compares worst-client F1-score to a shared-only (no personalization) model.
- **Assignment 2:**
  - Demonstrates ELM head adaptation to concept drift:
    - Trains ELM on pre-drift data.
    - Evaluates on post-drift data (performance drops).
    - Retrains ELM on post-drift data (performance recovers).
- **Logs and saves all results.**

---

## **Key Functions/Concepts**

- **Federated Learning:** Clients train locally, share only the global backbone, and keep adapters/ELM heads private.
- **Personalization:** Each client adapts the global model to its own data using a small adapter or ELM head.
- **Concept Drift:** The code simulates a change in client data and shows how the ELM head can quickly adapt.
- **Worst-Client F1:** Focuses on improving the performance for the hardest-to-fit client (fairness).

---

## **Input/Output**

- **Input:**  
  - Sensor data CSV file (with subject, activity, IMU features).
  - Directory path to data.
- **Output:**  
  - Logs of training and evaluation.
  - Worst-client F1-scores for personalized vs. shared-only models.
  - F1-scores before and after ELM adaptation to concept drift.
  - Results saved as CSV/log files in the output directory.

---

## **Potential Improvements**

1. **Efficiency:**
   - Use PyTorch’s `DataLoader` with `num_workers` for faster data loading.
   - Use GPU acceleration for ELM matrix operations if available.

2. **Readability:**
   - Add more docstrings and inline comments for clarity.
   - Modularize code further (e.g., separate data loading, model, and FL logic into different files).

3. **Functionality:**
   - Add support for more complex adapters (e.g., attention, residual connections).
   - Implement online/recursive ELM updates for true streaming adaptation.
   - Allow for more flexible simulation of concept drift (not just column swapping).

4. **Robustness:**
   - Add exception handling for data loading and training.
   - Validate input data shapes/types.

5. **Visualization:**
   - Plot per-client F1-scores for better insight into personalization benefits.
   - Visualize ELM adaptation over time (not just before/after).

6. **Scalability:**
   - Support for more clients, larger datasets, and more communication rounds.

---

## **Summary Table**

| Section                | Purpose                                                      |
|------------------------|-------------------------------------------------------------|
| Data Loading           | Loads and partitions data by client (subject)                |
| Model Architectures    | Defines shared backbone, adapters, and ELM head              |
| FL Simulation          | Runs federated learning with personalized adapters           |
| Evaluation             | Compares personalized vs. shared-only models                |
| Concept Drift          | Shows ELM head can quickly adapt to new client data          |
| Output                 | Logs, F1-scores, and CSVs for analysis                      |

---

**In summary:**  
This code demonstrates how to combine federated learning with per-client personalization (adapters and ELM heads) for robust, fair, and adaptive fall detection. It shows that personalization improves worst-case performance and that ELM heads can rapidly recover from changes in client data, making the system practical for real-world, on-device deployment.