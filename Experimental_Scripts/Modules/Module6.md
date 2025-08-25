# Module 6: Client Selection with Pareto Metrics

## Intuition 💡
In Federated Learning, clients can be unreliable—some may have noisy data, missing sensors, or old information. Simply averaging updates from all clients can harm the model. **Pareto-optimal selection** is a strategy to choose a subset of the "best" clients in each round based on multiple criteria, improving the stability and performance of the global model without requiring extra communication.

---
## Key Concepts 🧠

* **Metric Vector**: For each client *k*, we define a vector of metrics, $s_k$, that describes its quality and state. This can include factors like:
    $$s_k = [\text{loss}_k, \text{recency}_k, \text{completeness}_k, \text{diversity}_k, \text{size}_k]$$

* **Pareto Dominance**: A client *a* **dominates** client *b* if client *a* is no worse than client *b* in all metrics and is strictly better in at least one.

* **Selection**: The server identifies the **Pareto front**, which is the set of all clients that are not dominated by any other client. From this optimal set, the server then subsamples the number of clients needed for the training round. 

---
## Mini-Example 🧪
In each round, the server computes the following for every client: its training loss, when it was last seen, the completeness of its sensor data, a proxy for its data diversity (like feature variance), and its dataset size. Using these metrics, it builds the Pareto front and samples *K* clients to participate.

---
## Small Assignment 🎯

* Integrate Pareto client selection into the federated training pipeline from the previous modules.
* Compare its performance against random client selection, focusing on convergence speed, final F1-score, and the variance of the F1-score across rounds.

---
## Stretch Goal 🌟
Enhance the selection process by adapting the sampling probabilities *within* the Pareto front. Give higher priority to clients with a greater "need" score, such as those with under-represented data modalities.

### Code Explanation

This script implements an advanced federated learning (FL) simulation that compares standard **random client selection** with a more intelligent **Pareto-optimal client selection**. It uses the robust, energy-efficient `GatedResidualFusionModel` from the previous module.

#### 1\. Setup and Data Loading

  * **`setup_logging`, `set_seed`**: These are standard helper functions for creating log files and ensuring reproducible results.
  * **`load_and_create_clients`**: This is the main data pipeline.
      * It loads the IMU sensor data and images from two cameras.
      * It performs an **optimized alignment** to ensure all data sources are perfectly synchronized by timestamp.
      * Crucially, it partitions the data on a **per-subject basis**, creating a realistic Non-IID (non-uniform) federated environment where each client has a unique data distribution.
      * A global test set is created from held-out subjects for unbiased evaluation.

-----

#### 2\. Model Architecture

The script uses the `GatedResidualFusionModel` from Module 5. This is a powerful multimodal model with several key features:

  * **Encoders**: Separate encoders for IMU and image data to extract meaningful features.
  * **IMU-driven Gate**: A lightweight network that uses only the IMU data to decide if the computationally expensive image encoders should be activated.
  * **Modality Dropout**: During training, the model is forced to occasionally work without image data, making it more robust if sensors fail.

-----

#### 3\. Federated Learning Components

  * **`Client` Class**: Represents a single subject in the FL network.
      * It now simulates real-world conditions by having a `completeness` attribute. There's a chance each client is designated as "incomplete," meaning their camera data is zeroed out for the entire simulation.
      * It calculates several **Pareto metrics** after local training: `loss`, `accuracy`, `completeness`, `diversity` (a proxy for data variance), and `data_size`.
  * **`select_pareto_optimal_clients`**: This is the core of the new assignment.
      * It takes the metric vectors from all clients.
      * It identifies the **Pareto Front**—the set of clients that are not "dominated" by any other single client. A client is dominated if another client is better or equal on *all* metrics and strictly better on at least one.
      * It returns this set of non-dominated, optimal clients.
  * **`evaluate_global_model`**: A standard function to test the performance (F1-score) of the global model on the unseen test set.

-----

#### 4\. Main Simulation Runner

The `run_federated_simulation` function orchestrates the experiment.

1.  **Local Training (Pre-selection)**: In a key change for this module, *all* clients first train a copy of the global model locally. This is necessary to generate the performance metrics needed for the selection step.
2.  **Client Selection**: Based on the `selection_method` ('random' or 'pareto'), the server chooses which clients will contribute to the global update.
3.  **Aggregation**: The server takes the model updates *only* from the selected clients and performs a weighted average to create the new global model.
4.  **Evaluation**: The new global model is tested, and its F1-score is logged.

-----

#### 5\. Main Execution Block

This final section runs two full simulations and compares them.

1.  **Run Random Selection**: It runs the simulation with the `selection_method` set to 'random'.
2.  **Run Pareto Selection**: It runs the simulation again with the `selection_method` set to 'pareto'.
3.  **Plotting**: It generates and saves a plot comparing the F1-score convergence of both methods over the communication rounds.

-----

### Output Explanation: Why Pareto Selects Only Client 12

Based on your log output, the reason Pareto selection consistently chooses only **Client 12** is because that client **overwhelmingly dominates** all other clients across the most important metrics, particularly **loss and accuracy**.

Let's analyze the metrics from **Round 6** as an example:

```
--- Round 6/20 ---
> Pre-selection | Client  1: Loss=0.0210, Acc=0.9925
> Pre-selection | Client  2: Loss=0.0369, Acc=0.9865, Completeness=0.5
...
> Pre-selection | Client  7: Loss=0.0045, Acc=0.9984
...
> Pre-selection | Client 12: Loss=0.0011, Acc=0.9997, Completeness=1.0
```

#### What is Pareto Dominance?

A client A "dominates" a client B if A is **better or equal** on all metrics and **strictly better** on at least one. The metrics are:

  * `Loss` (lower is better)
  * `Accuracy` (higher is better)
  * `Completeness` (higher is better)
  * `Diversity` and `Data Size` (higher is better)

#### Analysis of Round 6:

1.  **Client 12 vs. Everyone Else**:

      * **Loss**: Client 12 has a loss of **0.0011**. This is drastically lower than the next best, Client 7 (0.0045), and orders of magnitude better than most other clients.
      * **Accuracy**: Client 12 has an accuracy of **0.9997**, which is the highest of all clients.
      * **Completeness**: Client 12 has a completeness of **1.0**, meaning its data is not simulated as faulty.

2.  **The Result**: Because Client 12 is the **undisputed best** in the two most critical performance metrics (loss and accuracy) and is also optimal in completeness, it dominates every other client. For any other client you pick, Client 12 is strictly better in at least loss and accuracy.

Since **no other client can dominate Client 12**, Client 12 is, by definition, on the Pareto Front. Since **Client 12 dominates all other clients**, no other client can be on the Pareto Front.

Therefore, the Pareto Front consists of a single member: **Client 12**. The selection algorithm correctly identifies this and, as a result, selects only that client for the global model update. This pattern repeats every round because Client 12's data is likely the "easiest" for the model to learn, leading to consistently superior local training performance.

You've made an excellent observation, and it cuts to the heart of the main challenge in federated learning.

The final F1-score is poor because the Pareto selection, while logically correct, is causing the global model to **overfit to a single, unrepresentative client**.

Let's break down exactly why this is happening.

---
### The "Star Student" Analogy 🎓

Imagine a teacher trying to create a final exam study guide (the **global model**) for a class of students (the **clients**). The final exam (the **test set**) covers many different topics.

* **Client 12** is the "star student." This student is a genius at **one specific topic** (e.g., calculus) and aces every quiz on it, achieving a perfect score (low loss, high accuracy).
* The other students are good at other topics (algebra, geometry, etc.), but none are as perfect at their one topic as the star student.
* The **Pareto selector** is like a teacher who decides, "I'll create the study guide by only asking my best-performing student for input."

Round after round, the teacher only gets input from the calculus genius. The resulting study guide becomes incredibly advanced in calculus but contains nothing about algebra or geometry. When the whole class takes the final exam, they fail because the study guide was based on an expert who wasn't representative of the entire curriculum.

---
### What's Happening in the Code 📉

This is exactly what is happening in your simulation:

1.  **Over-specialization**: Client 12 consistently gets the lowest loss and highest accuracy on its *own* local data. This means its data is likely very clean, uniform, or "easy." The local model trains on this data and becomes a specialist in recognizing Client 12's specific patterns.

2.  **Greedy but Flawed Selection**: The Pareto selection algorithm sees Client 12's outstanding local metrics and correctly identifies it as the "dominant" client. It's making a greedy, short-sighted decision based on which client *appears* to be the best in isolation.

3.  **Overfitting to One Client**: Because only Client 12 is ever selected, the server only ever averages in updates from this one specialist. The **global model** doesn't learn from the diverse data of other clients; instead, it becomes a copy of Client 12's over-specialized local model.

4.  **Failure to Generalize**: The global test set contains data from other subjects (e.g., 14, 15, 16, 17) who have different movement patterns. The global model, which has only ever learned from Client 12, has no idea how to handle this new, different data. It fails to generalize, resulting in a very poor F1-score on the test set.

---
### Why Random Selection is Better (in this case) 🎲

Random selection, while less sophisticated, avoids this trap.

By randomly choosing a **diverse group of clients** each round, it forces the global model to learn a little bit from everyone. The updates might be less "perfect" than Client 12's, but they are more varied. This process prevents the model from overfitting to any single client and results in a more robust, generalist model that performs better on the unseen test data.

**In short, your experiment perfectly demonstrates a key FL principle: the diversity of client contributions is often more important for building a strong global model than the isolated performance of a single "best" client.**