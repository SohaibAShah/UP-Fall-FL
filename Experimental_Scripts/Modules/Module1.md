Module 1 — IMU Fall Detection with a Tiny 1D-CNN (Foundations)
Intuition
Start with a single, always-available, low-power modality: the IMU (accelerometer/gyroscope). You’ll learn to window raw signals and train a lightweight 1D-CNN for binary classification (fall vs non-fall).
Key Equations

Windowing: Given continuous IMU stream $ a(t) $, form windows $ X \in \mathbb{R}^{T \times C} $ with length $ T $ and channels $ C $ (e.g., $ a_x, a_y, a_z, g_x, g_y, g_z $).
Model $ f_\theta $: Logits $ z = f_\theta(X) $; probabilities $ p = \sigma(z) $ for binary classification.
Loss (Binary Cross-Entropy): $ L = -[y \log p + (1-y) \log(1-p)] $.
Convolution (1D): $ y[n] = \sum_{k=0}^{K-1} w[k] x[n-k] + b $.

Mini-Example
Use 2 s windows at 100 Hz ($ T=200 $), $ C=6 $ channels. Two Conv1D blocks (kernel size 5), global average pooling, and a sigmoid head.
Small Assignment

Preprocess: Z-score each channel per subject; build subject-wise train/test split.
Train the 1D-CNN; report accuracy, F1, sensitivity, specificity. Plot a confusion matrix.
Stretch: Replace global average pooling with temporal attention and compare metrics.

Fall Detection Neural Networks
This repository contains code to train and compare two different types of neural networks for detecting falls using sensor data. The models are implemented in PyTorch and focus on processing time-series data from sensors to classify activities as "Fall" (0) or "No Fall" (1).
Overview
The code is designed to:

Load and preprocess sensor data from a CSV file.
Train two 1D Convolutional Neural Network (CNN) variants: one with Global Average Pooling (GAP) and another with Temporal Attention.
Evaluate model performance using metrics like accuracy, F1-score, sensitivity, and specificity.
Visualize results with confusion matrices and compare the two models.

1. Setup and Helper Functions 🛠️
This initial section imports all the necessary libraries and defines small, reusable functions that will be used throughout the script.

Imports: It brings in libraries like pandas for handling data, numpy for numerical operations, torch for building the neural network, sklearn for performance metrics, and matplotlib/seaborn for plotting graphs.
set_seed(seed=42): This function is crucial for reproducibility. Machine learning involves a lot of randomness (like initial model weights). By setting a "seed," we ensure that every time the code runs, the random numbers generated are the same, leading to the exact same results.
calculate_specificity(...): This calculates how well the model can correctly identify "No Fall" cases. It's an important metric in health applications to avoid false alarms.
plot_confusion_matrix(...): This function creates a visual grid that shows the model's performance. It displays how many "Falls" and "No Falls" were predicted correctly and incorrectly. The labels are explicitly set to ['Fall (0)', 'No Fall (1)'] to match your requirement.

2. Data Loading and Preprocessing 🧹
This is the most critical part for preparing the data. The load_and_preprocess_data function performs several key steps to get the raw sensor.csv file ready for the AI model.

Load and Clean: It loads the data and cleans up the column names to make them easier to work with.
Filter Data: It applies specific exclusion rules (like removing certain subjects or activities) as defined in the original notebook.
Binary Labeling: This is where your new requirement is implemented.

It defines a set of fall_activity_ids ({2, 3, 4, 5, 6}).
It then creates a new column called 'Fall'. If a row's Activity ID is in the fall_activity_ids set, it's labeled 0 (Fall); otherwise, it's labeled 1 (No Fall).


Windowing: Human activities happen over time, not in an instant. This step converts the continuous stream of sensor data into small, overlapping chunks called "windows."

The code uses a window size of 200 samples (equal to 2 seconds of data) with a 50% overlap.
A window is labeled as a Fall (0) if any single data point within that window is a fall. This ensures no fall event is missed.


Subject-wise Split: To ensure the model is tested on completely unseen individuals, the data is split into training and testing sets based on subject IDs. The model trains on one group of subjects and is evaluated on a different group.
Z-Score Normalization: It scales the sensor data so that each feature (e.g., x-axis acceleration) has a mean of 0 and a standard deviation of 1. This helps the model train faster and more effectively. Crucially, it learns the scaling parameters only from the training data to avoid leaking information from the test set.
Final Reshape: The data is reshaped into the format PyTorch expects: (batch_size, channels, window_length).

3. Model Architectures 🧠
This section defines the "brain" of the operation—the two neural network models that will learn to detect falls. Both are 1D Convolutional Neural Networks (1D-CNNs), which are excellent for finding patterns in time-series data like sensor signals.

CNN_GAP (Global Average Pooling)
This is the baseline model.

Two Conv1d layers: These layers act like pattern detectors, scanning the input windows to find features indicative of falls (e.g., sudden changes in acceleration).
AdaptiveAvgPool1d: After patterns are detected, this layer simplifies the information by taking the average of all the features. It's a simple and effective way to summarize what the convolutional layers found.
Linear layer: This is the final decision-making layer. It takes the summarized features and outputs a single value (a logit) that represents the likelihood of a fall.


CNN_Attention (Temporal Attention)
This is the more advanced model. It's identical to the CNN_GAP model, but it replaces the simple pooling layer with a custom TemporalAttention layer.

How Attention Works: Instead of treating all parts of the 2-second window equally (like averaging), the attention mechanism learns to pay more attention to the most important moments within the window. For a fall, this might be the moment of impact. This allows the model to focus on the most relevant information and potentially make better predictions.



4. Training and Evaluation Logic 🏋️‍♂️
These functions control the learning process and measure how well the models perform.

train_model(...): This function orchestrates the training loop.

It iterates through the training data for a set number of epochs (cycles).
In each cycle, it feeds the data to the model, calculates the error (loss) using BCEWithLogitsLoss (a stable loss function for binary tasks), and adjusts the model's internal parameters using the Adam optimizer to reduce that error.


evaluate_model(...): This function assesses the trained model's performance on the unseen test data.

It makes predictions for the entire test set.
It calculates four key metrics:

Accuracy: The overall percentage of correct predictions.
F1-Score (Fall): A balanced measure of the model's ability to correctly identify falls.
Sensitivity (Fall): The percentage of actual falls that the model correctly identified (also known as recall).
Specificity (No Fall): The percentage of actual "No Fall" activities that the model correctly identified.





5. Main Execution Block ▶️
This is the final part that runs the entire experiment from start to finish.

Set Hyperparameters: It defines key settings like the file path, window size, number of training epochs, and learning rate.
Load Data: It calls the load_and_preprocess_data function to prepare the datasets.
Run Experiment 1 (CNN-GAP): It creates, trains, and evaluates the baseline model, then prints its metrics and plots its confusion matrix.
Run Experiment 2 (CNN-Attention): It does the same for the more advanced attention-based model.
Final Comparison: It displays a final summary table that directly compares the performance metrics of both models, making it easy to see which one performed better.

Getting Started

Prerequisites: Ensure you have Python installed with the required libraries (pandas, numpy, torch, sklearn, matplotlib, seaborn).
Data: Place the sensor.csv file in the appropriate directory.
Run the Code: Execute the main script to train and evaluate the models.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by fall detection research in wearable sensors.
Built with PyTorch for deep learning.