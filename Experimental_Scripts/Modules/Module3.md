***

# Module 3: Stabilizing Federated Learning

## Intuition 🧠
In Federated Learning (FL), when clients have non-IID (non-uniform) data, their local model updates can pull the global model in conflicting directions. This can make training unstable. The algorithms in this module introduce "guardrails" to prevent local models from straying too far from the global consensus and to smooth out the server's update process, preventing it from zig-zagging.

---
## Key Equations ✒️
* **FedProx (proximal term)**: Client *k* minimizes its local loss plus a penalty term that keeps its model $\theta$ close to the global model $\theta^t$ it received. The strength of this penalty is controlled by $ \mu $:
    $$\min_{\theta} F_k(\theta) + \frac{\mu}{2} ||\theta - \theta^t||^2$$

* **SCAFFOLD (control variates)**: This method uses control variates on the client ($c_k$) and server ($c$) to correct for "client drift." The client's update step is adjusted using these variates:
    $$\theta \leftarrow \theta - \eta (\nabla F_k(\theta) - c_k + c)$$

* **Server Momentum**: This is another technique (not implemented in this specific script but relevant to stabilization) where the server update incorporates momentum to smooth out the aggregation process.

---
## Assignment Overview 📝

### Mini-Example
* Reuse the Non-IID client setup from the previous module.
* Enable FedProx with different $\mu$ values to observe its effect on convergence stability.

### Tasks
1.  **Run and Compare**: Execute simulations for FedAvg, FedProx, and SCAFFOLD.
2.  **Report Metrics**:
    * **Rounds-to-Target-F1**: How many rounds it takes for each algorithm to reach a target F1-score. This measures convergence speed.
    * **Final Worst-Client F1**: The final F1-score on the most difficult client. This measures fairness and generalization.
3.  **Stretch Goal**: Tune the $\mu$ hyperparameter for each client based on how much its model has drifted from the global model.

---
## Code Explanation

### 1. Setup and Helper Functions 🛠️
This initial section prepares the environment for the simulation.
* **`setup_logging`**: Configures a logger to save all console output to a file (`fl_module3_log.txt`), creating a persistent record of the experiment.
* **`set_seed(seed=42)`**: This is crucial for **reproducibility**. It ensures that all random processes (like client selection) are the same each time the code runs.

### 2. Data Loading and Client Creation 🧹
This section prepares the raw `sensor.csv` data for the federated scenarios.
* **`load_and_prep_data`**: Loads and cleans the dataset, filters out specified subjects, and creates the binary **Fall (0)** vs. **No Fall (1)** labels.
* **`create_clients`**: Partitions the data among clients. For this module, it uses a **Non-IID** setup where each client corresponds to a unique subject, realistically simulating diverse data distributions. It also creates a single, unified test set from unseen subjects for global evaluation.

### 3. Model Architecture 🧠
The script uses the `CNN_Attention` model. This **1D Convolutional Neural Network** is well-suited for time-series data and uses an **attention mechanism** to focus on the most important moments within a data window when making a prediction.

### 4. Federated Learning Components 🌐
These are the building blocks of the FL simulation, now enhanced to support FedProx and SCAFFOLD.

* **`Client` Class**: Represents a single device in the network.
    * The `train` method is now highly flexible. It takes a `config` object that tells it which algorithm to use (`fedavg`, `fedprox`, or `scaffold`).
    * For **FedProx**, it adds the proximal penalty term to the loss function.
    * For **SCAFFOLD**, it maintains a client-side `control_variate` and modifies the gradient updates according to the SCAFFOLD equation. It also calculates and returns the *deltas* (changes) for both the model and the control variate.
* **`evaluate_model`**: This function now calculates the **F1-score** for the "Fall" class, which is a more robust metric than accuracy for imbalanced datasets. It works on a *copy* of the global model to prevent accidentally moving it to the GPU.

### 5. Main Simulation Runner 🏃
The `run_federated_simulation` function orchestrates the entire experiment for a given algorithm.
1.  **Initialization**: It sets up the global model and initializes clients. For SCAFFOLD, it also creates the server-side and client-side control variates, initializing them to zeros.
2.  **Communication Rounds**: It loops for a set number of rounds. In each round:
    * A random subset of clients is selected.
    * The selected clients perform local training based on the specified algorithm.
    * The **server aggregation** logic branches:
        * For FedAvg and FedProx, it performs a weighted average of the clients' full model weights.
        * For SCAFFOLD, it aggregates the *deltas* from the clients to update the global model and the global control variate.
    * The updated global model is evaluated, and its F1-score is logged.
3.  **Final Metrics Calculation**: After all rounds are complete, it calculates the **Rounds-to-Target-F1** and the **Final Worst-Client F1**.

### 6. Main Execution Block & Plotting 📊
This final section runs the three experiments sequentially and presents the results.
1.  **Run Experiments**: It calls `run_federated_simulation` three times, once for each algorithm (FedAvg, FedProx, and SCAFFOLD), collecting the results.
2.  **Display and Save Results**: It prints a final comparison table to the console and saves it to `final_comparison.csv`.
3.  **Plotting**: It generates a plot of the F1-score vs. communication rounds for all three algorithms, making it easy to visually compare their convergence speed and stability. The plot is saved as `f1_vs_rounds.png`.