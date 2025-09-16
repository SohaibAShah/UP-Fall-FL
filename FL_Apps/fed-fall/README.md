# Federated Learning for Human Fall Detection

This project implements a federated learning system using the [Flower](https://flower.ai/) framework and PyTorch to train a model for fall detection based on wearable sensor data. The goal is to build a privacy-preserving model that can accurately identify fall events without centralizing raw user data.

This document details the project's evolution, from a simple baseline to a robust, optimized system, highlighting key challenges and improvements along the way.

-----

## 🚀 The Journey of Model Improvement

This project followed an iterative development process, tackling common challenges in machine learning and federated learning one by one.

### V1: The Initial Baseline (The "Stuck" Model)

The first version of the model consistently failed to improve, with its accuracy stalling at around 68%.

  * **Problem**: Evaluation accuracy was completely flat after the first round.
  * **Diagnosis**: **Class Imbalance**. The model had simply learned to predict the majority "Non-Fall" class.

*The data analysis confirmed that the "Non-Fall" class was dominant across all clients.*

### V2: Addressing Imbalance (The "Paranoid" Model)

To solve the imbalance, we introduced a weighted loss function and more advanced evaluation metrics.

  * **Code Changes**: Implemented `class_weights` in `CrossEntropyLoss` and added **Precision, Recall, and F1-Score** metrics.
  * **Result**: The model began identifying falls but went too far, resulting in extremely **high Recall** (often near 100%) but very **low Precision** (\~30-40%).
  * **Diagnosis**: The model had become "paranoid"—so afraid of missing a fall that it generated numerous false alarms.

### V3: Tackling Instability with FedAvg

The next step was to stabilize the erratic training process.

  * **Code Changes**: Introduced regularization (`Dropout`, `weight_decay`) and reduced `local-epochs` to `1` to prevent client-side overfitting.
  * **Result**: The model's performance remained very unstable, with all metrics bouncing unpredictably.
  * **Diagnosis**: Standard `FedAvg` was struggling with the **heterogeneous (Non-IID)** nature of the client data.

### V4: Achieving Stable Convergence with FedAdam

The key to solving instability was to switch to a more advanced federated optimization strategy.

  * **Code Change**: Replaced `FedAvg` with **`FedAdam`** in `server_app.py`.
  * **Result**: **Success\!** The training process became incredibly stable, with the evaluation loss showing a consistent, smooth trend.
  * **Diagnosis**: `FedAdam` solved the instability. However, the model's performance, while stable, plateaued, indicating the simple CNN architecture had reached its limit.

*Comparison showing the erratic loss of FedAvg (left) versus the smooth convergence of FedAdam (right).*

### V5: Increasing Model Capacity (The "Timid" Model)

With a stable pipeline, we increased the model's power by switching to a more advanced `Conv-LSTM` architecture.

  * **Code Change**: The `Net` architecture was replaced with a hybrid `ConvLSTMNet` in `task.py`.
  * **Result**: In a short 20-round run, the model became very conservative (very low Recall, higher Precision). The F1-score was low but showed a slow, steady upward trend.
  * **Diagnosis**: The new, larger model was **undertrained**. It needed more time and a slightly faster learning rate to reach its potential.

### V6: The Final Tuned Model (Peak Performance)

The final step was to give the powerful `Conv-LSTM` model the resources it needed to fully train.

  * **Final Configuration**:
    1.  **Increased Training Rounds**: `num-server-rounds` was increased to **50**.
    2.  **Increased Learning Rate**: `lr` was increased to **`0.001`**.
  * **Result**: **Project Success\!** The model's F1-Score steadily climbed, reaching a peak performance of **\~60%**. It settled into an optimal strategy for this problem: **extremely high Recall (\~98%)** and moderate Precision (\~42%). The increasing loss in later rounds confirmed the model had reached its performance ceiling.

-----

## 🏁 Final Model and Conclusion

The final model is a well-trained, specialized fall detector. It is excellent at its primary job—**not missing real fall events**—at the acceptable cost of some false alarms. This high-recall strategy is often the desired outcome for safety-critical applications.

Through a systematic process of diagnosing issues and implementing targeted solutions, this project successfully built a stable, effective federated learning system for a complex, real-world problem.

-----

## ⚙️ Tech Stack

  * **Framework**: Flower 1.8+
  * **ML Library**: PyTorch
  * **Metrics**: Scikit-learn
  * **Data Handling**: Pandas, NumPy
  * **Plotting**: Matplotlib, Seaborn

-----

## 📂 Project Structure

```
fed_fall/
├── partitions/            # Holds the partitioned client data (.pt files)
├── fed-fall/
│   ├── __init__.py
│   ├── server_app.py      # Defines the server logic and strategy
│   ├── client_app.py      # Defines the client logic (train/evaluate)
│   └── task.py            # Contains the model architecture and ML logic
├── pyproject.toml         # Project configuration and dependencies
└── plot_results.py        # Script to parse logs and plot metrics
```

-----

## 📄 Code Explanation

### `pyproject.toml`

The main configuration hub. It defines dependencies, links the server/client apps, and sets crucial hyperparameters for the simulation like `num-server-rounds`, `lr`, and `num-supernodes`.

### `task.py`

The core machine learning module.

  * **`ConvLSTMNet(nn.Module)`**: The final, high-performing model. It uses two `Conv1d` layers to extract features and a two-layer `LSTM` to learn the temporal sequence of those features.
  * **`train()` / `test()`**: The functions for client-side training and evaluation, complete with weighted loss for imbalance and detailed metric calculations (Precision, Recall, F1-Score).

### `server_app.py`

Defines the central server's behavior.

  * **`@app.main()`**: The entry point that orchestrates the FL process.
  * **Strategy Definition**: Uses the **`FedAdam`** strategy to ensure stable convergence with heterogeneous client data.

### `client_app.py`

Defines the client logic.

  * **`@app.train()`**: Receives the global model, trains it on local data, and sends back the updated weights.
  * **`@app.evaluate()`**: Evaluates the global model on local data and returns a rich set of performance metrics to the server.

-----

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install flwr[simulation] torch scikit-learn pandas matplotlib seaborn
    ```
2.  **Prepare Data**:
    Run the `dataset.py` script to generate the client partitions in the `partitions/` directory.
3.  **Start the Simulation**:
    From the root directory, run the Flower app with the final configurations.
    ```bash
    flwr run .
    ```
4.  **Visualize Results**:
    After a run completes, save the terminal output to `training_log.txt` and run the plotting script:
    ```bash
    python plot_results.py
    ```