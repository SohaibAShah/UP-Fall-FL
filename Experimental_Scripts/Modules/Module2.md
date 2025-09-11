Of course. Here is the provided text converted into a clean, well-structured Markdown format.

***

# Module 2: Federated Learning (FedAvg) on Non-IID Clients

## Intuition 💡
In Federated Learning (FL), each device trains a model on its local data and shares only the model updates (not the data itself) with a central server. When clients have **Non-IID** (non-uniform) data, which is common in the real world due to different daily routines or the use of mobility aids, the simple averaging of model updates can become unstable. However, **Federated Averaging (FedAvg)** is the foundational baseline protocol to understand first.

---
## Key Equations 🧠

* **Local Empirical Risk**: The loss function for a single client *k* is the average loss over their local dataset $D_k$:
    $$F_k(\theta) = \frac{1}{|D_k|} \sum_{(x,y)\in D_k} \ell(f_\theta(x), y)$$

* **FedAvg Local Update**: Each client updates its model for *E* local epochs by performing gradient descent on its local data, starting from the current global model $\theta^t$:
    $$\theta_k^{t+1} = \theta^t - \eta \sum_{e=1}^{E} \nabla_\theta F_k(\theta)$$

* **Server Aggregation**: The server creates the new global model $\theta^{t+1}$ by performing a weighted average of the updated local models from the selected clients $S_t$. Each client's contribution is weighted by the size of its local dataset $|D_k|$:
    $$\theta^{t+1} = \sum_{k \in S_t} \frac{|D_k|}{\sum_{j \in S_t} |D_j|} \theta_k^{t+1}$$

---
## Mini-Example 🧪
Simulate 50 clients, where each is given a different ratio of fall-to-non-fall data to create a **label-skew** non-IID scenario. In each communication round, sample 10 clients and have them train for 5 local epochs.

---
## Small Assignment 🎯
* Implement the **FedAvg** algorithm, using subject-wise data partitions to create clients.
* Compare the performance of the model when trained on **IID vs. Non-IID** data partitions.
* Plot the global model's **accuracy vs. communication rounds** and **accuracy vs. communication cost (MB)**.

---
## Stretch Goal 🌟
* Enhance the client selection process by making the sampling probability for each client proportional to its **most recent training loss**.

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

---

# Results


### **Plot 1: Global Model Accuracy vs. Communication Rounds**

**What it shows:**  
- The x-axis is the number of communication rounds (how many times the server aggregates client models).
- The y-axis is the global test accuracy (how well the global model performs on the test set).
- Three lines represent different federated learning strategies:
  - **Non-IID (Uniform Sampling):** Blue, dashed line with circles.
  - **IID (Uniform Sampling):** Orange, solid line with squares.
  - **Non-IID (Loss-Based Sampling):** Green, dotted line with triangles.

**Interpretation:**  
- **IID (Uniform Sampling)** achieves higher and more stable accuracy compared to **Non-IID (Uniform Sampling)**. This is expected, as IID data distribution across clients makes federated learning easier and more consistent.
- **Non-IID (Uniform Sampling)** shows more fluctuation and lower accuracy, reflecting the challenge of learning from heterogeneous (non-IID) data.
- **Non-IID (Loss-Based Sampling)** outperforms both, reaching high accuracy quickly and maintaining it. This strategy prioritizes clients with higher loss, which helps the global model learn from harder or underrepresented data, improving convergence and final accuracy.

---

### **Plot 2: Global Model Accuracy vs. Communication Cost (MB)**

**What it shows:**  
- The x-axis is the cumulative communication cost in megabytes (MB), i.e., the total amount of data exchanged between server and clients.
- The y-axis is again the global test accuracy.
- The same three strategies are compared.

**Interpretation:**  
- **IID (Uniform Sampling)** and **Non-IID (Loss-Based Sampling)** both reach high accuracy with less communication cost compared to **Non-IID (Uniform Sampling)**.
- **Non-IID (Loss-Based Sampling)** is the most communication-efficient: it achieves high accuracy with fewer MBs exchanged, indicating that smarter client selection (based on loss) can reduce communication overhead while improving performance.
- **Non-IID (Uniform Sampling)** is less efficient, requiring more communication to reach similar or lower accuracy.

---

### **Summary Table**

| Strategy                       | Accuracy | Stability | Communication Efficiency |
|---------------------------------|----------|-----------|-------------------------|
| IID (Uniform Sampling)          | High     | Stable    | Good                    |
| Non-IID (Uniform Sampling)      | Lower    | Fluctuates| Poor                    |
| Non-IID (Loss-Based Sampling)   | Highest  | Stable    | Best                    |

---

### **Key Takeaways**

- **IID data** makes federated learning easier and more reliable.
- **Non-IID data** can hurt performance and stability if clients are sampled uniformly.
- **Loss-based client sampling** in Non-IID settings can significantly improve both accuracy and communication efficiency, making federated learning more practical for real-world, heterogeneous data.

---

## **Why These Effects Occur**

### 1. **IID vs. Non-IID Data**
- **IID (Independent and Identically Distributed):**
  - Each client’s data is a good representative sample of the overall data distribution.
  - When models are aggregated, their updates are consistent and reinforce each other, leading to stable and fast convergence.
- **Non-IID:**
  - Each client’s data may be biased or only represent a subset of the global distribution (e.g., some clients see mostly “fall” events, others mostly “no fall”).
  - Model updates can conflict, causing instability and slower or poorer convergence. The global model may not generalize well.

### 2. **Uniform vs. Loss-Based Sampling**
- **Uniform Sampling:**
  - Every client has an equal chance of being selected, regardless of how much their data can help the global model.
  - In Non-IID settings, this can mean the model spends too much time on “easy” or redundant data, and not enough on “hard” or underrepresented data.
- **Loss-Based Sampling:**
  - Clients with higher local loss (i.e., whose data is not well predicted by the current global model) are prioritized.
  - This focuses learning on the most challenging or least-represented data, speeding up convergence and improving generalization.
  - It also reduces communication cost, as fewer rounds are needed to reach high accuracy.

---

## **How to Further Improve Federated Learning**

### **A. Algorithmic Improvements**
- **Personalized FL:**  
  Instead of a single global model, allow some personalization for each client (e.g., FedPer, pFedMe). This helps in highly Non-IID settings.
- **Adaptive Aggregation:**  
  Weight client updates not just by data size, but also by data quality or relevance.
- **Regularization:**  
  Use techniques like FedProx to penalize local models that drift too far from the global model, improving stability.

### **B. Data Handling**
- **Data Augmentation:**  
  Encourage clients to augment their data to better represent the global distribution.
- **Client Clustering:**  
  Group similar clients and aggregate within clusters before global aggregation.

### **C. Communication Efficiency**
- **Model Compression:**  
  Use quantization or sparsification to reduce the size of model updates.
- **Partial Model Updates:**  
  Only send/receive parts of the model that changed significantly.

### **D. Practical Considerations**
- **Fairness:**  
  Ensure that minority clients (with rare data) are not ignored, which can be addressed by loss-based or fairness-aware sampling.
- **Privacy:**  
  Use differential privacy or secure aggregation to protect client data.

---

## **Summary Table**

| Challenge         | Solution Example                | Benefit                        |
|-------------------|--------------------------------|--------------------------------|
| Non-IID Data      | Loss-based sampling, FedProx   | Better accuracy, stability     |
| Communication     | Model compression, smart sampling | Lower bandwidth, faster convergence |
| Personalization   | Local fine-tuning, pFedMe      | Better client-specific results |
| Fairness          | Fair sampling, reweighting     | Avoids bias, improves equity   |

---

## **References for Further Reading**
- [Federated Learning: Challenges, Methods, and Future Directions (Kairouz et al., 2021)](https://arxiv.org/abs/1912.04977)
- [FedProx: Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
- [Personalized Federated Learning: A Meta-Learning Approach](https://arxiv.org/abs/1802.07876)

---