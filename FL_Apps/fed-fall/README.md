Of course. Here is a full, detailed `README.md` file that documents the entire project, from the initial setup to the iterative improvements we made together. It explains the code, the challenges, and the solutions, serving as a complete guide to your work.

You can copy and paste the content below directly into a `README.md` file in your project's root directory.

-----

# Federated Learning for Human Fall Detection

This project implements a federated learning system using the [Flower](https://flower.ai/) framework and PyTorch to train a model for fall detection based on wearable sensor data. The goal is to build a privacy-preserving model that can accurately identify fall events without centralizing raw user data.

This document details the project's evolution, from a simple baseline to a robust, optimized system, highlighting key challenges and improvements along the way.

-----

## 🚀 The Journey of Model Improvement

This project followed an iterative development process, tackling common challenges in machine learning and federated learning one by one.

### V1: The Initial Baseline (The "Stuck" Model)

The first version of the model consistently failed to improve, with its accuracy stalling at around 68%.

  * **Problem**: Evaluation accuracy was completely flat after the first round.
  * **Diagnosis**: **Class Imbalance**. An analysis of the dataset revealed that the "Non-Fall" class made up \~68% of the test data. The model had simply learned to predict the majority class every time, which is a common failure mode for imbalanced datasets.

*The data analysis confirmed that the "Non-Fall" class was dominant across all clients.*

### V2: Addressing Imbalance with Weighted Loss (The "Paranoid" Model)

To solve the imbalance, we introduced a weighted loss function and more advanced evaluation metrics.

  * **Code Changes**:
      * Modified the `CrossEntropyLoss` in `task.py` to include `class_weights`, penalizing the model more for misclassifying the rare "Fall" class.
      * Added **Precision**, **Recall**, and **F1-Score** to the `test()` function to get a better understanding of performance.
  * **Result**: The model immediately started identifying the minority class, but it went too far. We observed extremely **high Recall** (often near 100%) but very **low Precision** (\~30-40%).
  * **Diagnosis**: The model had become "paranoid"—so afraid of missing a fall that it started labeling many normal activities as falls, leading to numerous false alarms.

### V3: Tackling Instability with FedAvg

The next step was to stabilize the training process, as the metrics were highly erratic from one round to the next.

  * **Code Changes**:
      * Introduced regularization (`Dropout` in the model, `weight_decay` in the optimizer) to prevent overfitting.
      * Reduced `local-epochs` to `1` to stop clients from memorizing their small local datasets.
  * **Result**: The model's performance was still very unstable. All evaluation metrics bounced up and down without a clear trend of improvement.
  * **Diagnosis**: The standard `FedAvg` algorithm was struggling with the **heterogeneous (Non-IID)** nature of the client data. Averaging models trained on vastly different user data was pulling the global model in different directions each round.

### V4: Achieving Stable Convergence with FedAdam

The key to solving instability was to switch to a more advanced federated optimization strategy.

  * **Code Change**:
      * Replaced `FedAvg` with **`FedAdam`** in `server_app.py`. `FedAdam` uses server-side momentum to smooth out the erratic client updates, leading to more stable convergence.
  * **Result**: **Success\!** The training process became incredibly stable. The evaluation loss showed a consistent, smooth downward trend.
  * **Diagnosis**: `FedAdam` successfully solved the instability caused by heterogeneous data. However, the model's performance, while stable, was still low (F1-Score plateaued around 50%), indicating that the simple model architecture had reached its limit.

*Comparison showing the erratic loss of FedAvg (left) versus the smooth convergence of FedAdam (right).*

### V5: Increasing Model Capacity (The "Timid" Model)

With a stable pipeline, we increased the model's power to learn more complex patterns.

  * **Code Change**: The `Net` architecture in `task.py` was made "deeper" by adding a **third convolutional block**.
  * **Result**: The model's behavior flipped. It became very conservative, with very **low Recall** but higher **Precision**. The F1-Score was low but showed a slow, steady upward trend.
  * **Diagnosis**: The new, larger model was **undertrained**. 20 rounds were not enough for it to learn effectively, so it resorted to a "safe" strategy of rarely predicting the fall class.

### Final Step: Unleashing the Model's Potential

The final recommendation is to provide the powerful, deeper model with enough resources to learn properly.

  * **Recommended Changes**:
    1.  **Increase Training Rounds**: In `pyproject.toml`, increase `num-server-rounds` from 20 to **50+** to allow the slow, steady improvement to continue.
    2.  **Increase Learning Rate**: In `pyproject.toml`, increase `lr` from `0.0005` back to **`0.001`** to help the larger model train more efficiently.

This final tuning phase leverages the stable `FedAdam` pipeline and the powerful 3-layer CNN to train a high-performing, generalized model.

-----

## ⚙️ Tech Stack

  * **Framework**: Flower 1.21.0
  * **ML Library**: PyTorch
  * **Metrics**: Scikit-learn
  * **Data Handling**: Pandas, NumPy
  * **Plotting**: Matplotlib, Seaborn

-----

## 📂 Project Structure

```
fed-fall/
├── partitions/            # Holds the partitioned client data (.pt files)
│   ├── client_1.pt
│   ├── ...
│   └── test.pt
├── fed-fall/
│   ├── __init__.py
│   ├── server_app.py      # Defines the server logic and strategy
│   ├── client_app.py      # Defines the client logic (train/evaluate)
│   └── task.py            # Contains the model architecture, data loading, and train/test functions
├── pyproject.toml         # Project configuration, dependencies, and simulation settings
└── plot_results.py        # (Optional) Script to parse logs and plot metrics
```

-----

## 📄 Code Explanation

### `pyproject.toml`

This file is the main configuration hub for the Flower project.

  * `[project]`: Defines project dependencies like `flwr`, `torch`, etc.
  * `[tool.flwr.app.components]`: Links the `serverapp` and `clientapp` objects from our Python files.
  * `[tool.flwr.app.config]`: Sets global hyperparameters for the run, such as the number of rounds, learning rate (`lr`), and local epochs.
  * `[tool.flwr.federations.local-simulation]`: Configures the simulation, including the total number of virtual clients (`num-supernodes`).

### `task.py`

This is the core machine learning module.

  * **`Net(nn.Module)`**: Defines the 1D Convolutional Neural Network (CNN). Our final version uses three convolutional blocks to extract hierarchical features from the time-series sensor data. A `Dropout` layer is included to prevent overfitting.
  * **`load_data()`**: This function is responsible for finding the correct client data partition on disk (e.g., `client_1.pt`), loading it, and preparing `DataLoaders` for PyTorch.
  * **`train()`**: Contains the client-side training loop. It uses a weighted `CrossEntropyLoss` to handle class imbalance and an `Adam` optimizer with `weight_decay` for regularization.
  * **`test()`**: Contains the evaluation logic. It calculates loss and accuracy, and more importantly, uses `scikit-learn` to compute **Precision, Recall, and F1-Score** for the "Fall" class.

### `server_app.py`

This file defines the central server's behavior.

  * **`@app.main()`**: The main function that orchestrates the federated learning process.
  * **Model Initialization**: It initializes a global instance of the `Net` from `task.py`. It's critical that `NUM_FEATURES` is set correctly here.
  * **Strategy Definition**: It defines the aggregation strategy. We evolved from `FedAvg` to **`FedAdam`** to handle data heterogeneity and achieve stable convergence. The strategy is configured with parameters from `pyproject.toml`.
  * **Execution**: It starts the strategy, which runs for the specified number of rounds, and finally saves the trained global model weights to `final_model.pt`.

### `client_app.py`

This file defines what each client does when contacted by the server.

  * **`@app.train()`**: This function is called when a client is selected for training. It receives the global model weights from the server, loads its local data partition, trains the model using the `train()` function from `task.py`, and sends the updated weights and training metrics back.
  * **`@app.evaluate()`**: This function is called for evaluation. It receives the latest global model, evaluates it on its local test data using the `test()` function from `task.py`, and returns the calculated metrics (Loss, Accuracy, Precision, Recall, F1-Score) to the server for aggregation.

-----

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt 
    # Or install manually: flwr[simulation], torch, scikit-learn, pandas
    ```
2.  **Prepare Data**:
    Run the `dataset.py` script (not included here, assumed to exist) to generate the client partitions in the `partitions/` directory.
3.  **Start the Simulation**:
    From the root directory (`waooowooo/`), run the Flower app:
    ```bash
    flwr run .
    ```
4.  **Visualize Results**:
    After the run completes, save the terminal output to `training_log.txt` and run the plotting script:
    ```bash
    python plot_results.py
    ```