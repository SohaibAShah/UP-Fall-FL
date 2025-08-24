# Module 2 — Federated learning (FedAvg) on non‑IID clients
Intuition

With FL, each device trains locally and shares only model updates. Non‑IID data (different daily routines, mobility aids) makes naive averaging unstable, but FedAvg is the baseline protocol to understand first.
Key equations

Local empirical risk at client k: F_k(θ) = (1/|D_k|) ∑_{(x,y)∈D_k} ℓ(fθ(x), y).
FedAvg local update: θ_k^{t+1} = θ^t − η ∑_{e=1}^{E} ∇θ F_k(θ) (E local epochs).
Server aggregation: θ^{t+1} = ∑{k∈S_t} (|D_k|/∑{j∈S_t}|D_j|) θ_k^{t+1}.
Mini‑example

Simulate 50 clients; give each a different fall:non‑fall ratio (label‑skew). Train 5 local epochs per round; sample 10 clients per round.
Small assignment

Implement FedAvg with subject‑wise partitions as clients.
Compare IID vs non‑IID partitions. Plot accuracy vs rounds and communication (MB).
Stretch: add client sampling probability proportional to recent loss.

# Explanation of the Federated Learning Fall Detection Script

This document provides a comprehensive walkthrough of the Python script designed to simulate and evaluate federated learning (FL) for fall detection. The script compares three scenarios: Non-IID clients, IID clients, and Non-IID clients with an advanced loss-based sampling strategy.

---

## 1. Setup and Helper Functions 🛠️

This initial section prepares the environment by importing necessary libraries and defining utility functions.

* **Imports**: It brings in libraries like `pandas` for data manipulation, `numpy` for numerical operations, `torch` for building the neural network, `sklearn` for performance metrics, and `matplotlib` for plotting.
* **`setup_logging`**: This function configures a logging system to save all console output (like training progress) to a file named `fl_run_log.txt`. This is crucial for keeping a record of each experiment.
* **`set_seed(seed=42)`**: Ensures that the experiments are **reproducible**. By setting a "seed," the random processes (like model initialization and client selection) will be the same every time the code is run.
* **`get_model_size_mb(model)`**: A utility to calculate the size of the model in megabytes. This is used to track the **communication cost** of sending the model between the server and clients.

---

## 2. Data Loading and Client Creation 🧹

This section is responsible for preparing the raw `sensor.csv` data for the federated simulation. It defines how the data is partitioned among the different clients.

* **`load_and_prep_data`**: This function handles the initial data loading and cleaning. It filters out irrelevant subjects and activities and converts the multi-class activity labels into a binary **Fall (0)** vs. **No Fall (1)** format.
* **`create_clients`**: This is a critical function that creates the client data partitions for the two main scenarios:
    * **Non-IID (Realistic Scenario)**: If `iid=False`, the data is partitioned by **subject**. Each client receives all the data for one unique person. Since every individual moves differently, the data distribution is naturally non-uniform (Non-Independent and Identically Distributed).
    * **IID (Control Scenario)**: If `iid=True`, the script first creates all possible training "windows" from the entire dataset. Then, it **shuffles** these intact windows and distributes them evenly among the clients. This ensures every client has a statistically similar mix of data, which is theoretically easier for the standard FedAvg algorithm.
* **Global Test Set**: A single, unified test set is created from unseen subjects to provide a consistent and unbiased evaluation of the global model after each round.

---

## 3. Model Architecture 🧠

The script uses a sophisticated **1D Convolutional Neural Network (CNN)** named `CNN_Attention`.

* **1D Convolutional Layers**: These layers are excellent at automatically detecting patterns in time-series data, such as the sudden changes in acceleration and rotation that characterize a fall.
* **Temporal Attention Mechanism**: Instead of treating all parts of a 2-second data window equally, the attention layer learns to focus on the most critical moments. For a fall, this might be the instant of impact. This allows the model to make more informed decisions by weighing the most relevant information more heavily. 

---

## 4. Federated Learning Components 🌐

These are the core building blocks that simulate the federated learning process.

* **`Client` Class**: This class represents an individual device or person in the FL network.
    * The `train` method simulates local training. The client receives the global model, trains it for several epochs on its own private data, and calculates its performance (loss and accuracy). It then prepares its updated model to be sent back to the server.
* **`server_aggregate` Function**: This function implements the **Federated Averaging (FedAvg)** algorithm. It collects the updated models from the selected clients and computes a weighted average of them. The weight for each client's model is proportional to the size of its local dataset. This aggregated result becomes the new global model for the next round.
* **`evaluate_global_model` Function**: After each round of aggregation, this function assesses the performance of the newly updated global model against the unseen global test set.

---

## 5. Main Simulation Runner 🏃

The `run_federated_simulation` function orchestrates the entire experiment.

1.  **Initialization**: It sets up the global model and initializes all the client objects.
2.  **Communication Rounds**: It loops for a specified number of rounds (e.g., 50). In each round, it performs the following steps:
    * **Client Selection**: The server chooses a subset of clients to participate in the training. This is done in two ways:
        * **Uniform**: Clients are chosen randomly.
        * **Loss-Based (Stretch Goal)**: Clients that had a higher training loss in previous rounds are more likely to be selected. The intuition is that these clients have data that the global model is struggling with, so training on them could be more beneficial.
    * **Local Training**: The selected clients train the model locally.
    * **Server Aggregation**: The server collects the updates and creates the new global model.
    * **Global Evaluation**: The new model is tested, and its accuracy and the cumulative communication cost are recorded.
3.  **History**: The function returns a complete history of the model's performance over all rounds.

---

## 6. Main Execution Block & Plotting 📊

This final section runs the three experiments and visualizes the results.

1.  **Run Experiments**: It calls `run_federated_simulation` three times for:
    * Non-IID with uniform sampling.
    * IID with uniform sampling.
    * Non-IID with loss-based sampling.
2.  **Save Results**: The performance history for each experiment is saved to a `.csv` file, and all console output is saved to `fl_run_log.txt` in the `output_fl` directory.
3.  **Plotting**: It generates and saves two plots:
    * **Accuracy vs. Communication Rounds**: This shows how quickly each method learns.
    * **Accuracy vs. Communication Cost (MB)**: This shows how efficiently each method uses the communication budget.

These plots provide a clear visual comparison of the performance and efficiency of the different federated learning strategies.