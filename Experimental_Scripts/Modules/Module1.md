Of course. Here is the provided information converted into a clean, well-structured Markdown format.

# Module 1: IMU Fall Detection with a Tiny 1D-CNN

This project demonstrates how to build and train a lightweight 1D Convolutional Neural Network (1D-CNN) for fall detection using Inertial Measurement Unit (IMU) sensor data. It starts with the foundational concept of using a single, low-power modality (accelerometer/gyroscope) for binary classification (fall vs. non-fall).

---
## Intuition 💡
The core idea is to start with a simple, always-available sensor like an IMU. We process the raw, continuous signal by slicing it into small, fixed-size "windows" and feed these windows into a lightweight neural network to learn the difference between normal activities and falls.

---
## Key Concepts and Equations 🧠
* **Windowing**: Given a continuous IMU stream $ a(t) $, we form windows $X \in \mathbb{R}^{T \times C}$. Here, $ T $is the window length (e.g., 200 samples for a 2-second window at 100 Hz), and$ C $is the number of channels (e.g., 6 channels for$ a_x, a_y, a_z, g_x, g_y, g_z $).

* **Model ($f_\theta$)**: The neural network model takes a window $X$ and outputs logits $z = f_\theta(X)$. For binary classification, these are converted to probabilities using the sigmoid function: $p = \sigma(z)$.

* **Loss Function (Binary Cross-Entropy)**: The model's error is calculated using the binary cross-entropy loss, which measures the difference between the predicted probability $p$ and the true label $ y $:
    $$L = -[y \log p + (1-y) \log(1-p)]$$

* **1D Convolution**: The core operation of the CNN, where a learned filter (or kernel) $w$ slides across the input signal $x$ to produce a feature map $ y $:
    $$y[n] = \sum_{k=0}^{K-1} w[k] x[n-k] + b$$

---
## Project Assignment Overview 📝
### Mini-Example
* **Data Windows**: Use 2-second windows sampled at 100 Hz, resulting in a window length of $T=200$ samples.
* **Channels**: Use $C=6$ channels (3-axis accelerometer + 3-axis gyroscope).
* **Architecture**: A simple 1D-CNN with two convolutional blocks (kernel size 5), followed by a global average pooling layer and a final sigmoid output layer.

### Tasks
1.  **Preprocessing**: Apply Z-score normalization to each sensor channel on a per-subject basis and create a subject-wise train/test data split.
2.  **Training & Evaluation**: Train the 1D-CNN and report its **accuracy, F1, sensitivity,** and **specificity**. Visualize the results with a confusion matrix.
3.  **Stretch**: Replace the global average pooling layer with a **temporal attention** mechanism and compare the performance metrics of the two models.

---
## Code Implementation Details
### 1. Setup and Helper Functions 🛠️
The script begins by importing essential libraries like `pandas`, `numpy`, `torch`, `sklearn`, and `matplotlib`. Key helper functions are defined:
* **`set_seed(seed=42)`**: Ensures reproducible results by controlling sources of randomness.
* **`calculate_specificity(...)`**: Computes the model's ability to correctly identify "No Fall" cases.
* **`plot_confusion_matrix(...)`**: Creates a visual grid to show correct and incorrect predictions for both "Fall" (0) and "No Fall" (1) classes.

### 2. Data Loading and Preprocessing 🧹
This is a critical stage where the raw `sensor.csv` data is prepared for the model.
* **Load and Clean**: Loads the data and standardizes column names.
* **Filter and Label**: Applies specific exclusion rules and converts multi-class activity data into binary labels. Activities `{2, 3, 4, 5, 6}` are labeled as **Fall (0)**, and all others are **No Fall (1)**.
* **Windowing**: The continuous sensor data is sliced into 2-second windows with a 50% overlap. A window is labeled a "Fall" if any data point within it is a fall.
* **Subject-wise Split**: Data is split by subject to ensure the model is tested on data from completely unseen individuals.
* **Z-Score Normalization**: Scales the data to have a mean of 0 and a standard deviation of 1, which helps optimize the training process.

### 3. Model Architectures 🧠
Two 1D-CNN architectures are implemented to learn patterns from the time-series sensor data.

#### `CNN_GAP` (Global Average Pooling)
This is the baseline model. It uses two `Conv1d` layers to detect features, an `AdaptiveAvgPool1d` layer to summarize these features by taking their average, and a final `Linear` layer for the classification decision.

#### `CNN_Attention` (Temporal Attention)
This advanced model replaces the simple pooling layer with a custom **`TemporalAttention`** layer. Instead of averaging all features equally, the attention mechanism learns to assign more weight to the most critical moments within a window (e.g., the moment of impact during a fall), potentially leading to better predictions. 

### 4. Training and Evaluation Logic 🏋️‍♂️
These functions manage the model's learning and performance assessment.
* **`train_model(...)`**: Orchestrates the training loop. It feeds data to the model, calculates the error (loss) using `BCEWithLogitsLoss`, and uses the `Adam` optimizer to adjust the model's parameters to minimize this error.
* **`evaluate_model(...)`**: Assesses the trained model's performance on the unseen test data by calculating key metrics like Accuracy, F1-Score, Sensitivity (Recall), and Specificity.

### 5. Main Execution Block ▶️
This final section runs the entire experiment. It sets hyperparameters, loads the data, and then creates, trains, and evaluates both the `CNN_GAP` and `CNN_Attention` models, printing a final table to compare their performance.

---
## Getting Started
### Prerequisites
Ensure you have Python installed along with the required libraries: `pandas`, `numpy`, `torch`, `sklearn`, `matplotlib`, and `seaborn`.

### Usage
1.  Place the `sensor.csv` file in the same directory as the script.
2.  Execute the Python script to start the training and evaluation process.

---
## License
This project is licensed under the MIT License.

## Acknowledgments
* Inspired by fall detection research using wearable sensors.
* Deep learning components built with PyTorch.