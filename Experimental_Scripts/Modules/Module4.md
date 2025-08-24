Module 4: Multimodal Fusion
Intuition 💡
Combining a low-power sensor like an IMU with a richer sensor like an RGB camera can boost accuracy. However, sensors can be noisy or provide redundant information. Advanced fusion techniques, like residual or intermediate fusion, allow a model to learn when and how much to trust each data source, making it more robust.

Key Equations ✒️
Early Fusion: Features from different sensors (ϕ 
IMU
​
 , ϕ 
RGB
​
 ) are extracted and concatenated before being passed to a classifier (φ).

h=φ([ϕ 
IMU
​
 (X 
I
​
 ),ϕ 
RGB
​
 (X 
R
​
 )])
Late Fusion: Each sensor modality has its own classifier, and their final predictions (p 
IMU
​
 , p 
RGB
​
 ) are combined, often with a learned or fixed weight α.

p=αp 
IMU
​
 +(1−α)p 
RGB
​
 
Residual Fusion (Gated): This is a more sophisticated form of intermediate fusion. The features from one modality (e.g., IMU) are treated as a baseline, and the features from another (e.g., RGB) are added selectively, controlled by a learned gate (g).

h=ϕ 
IMU
​
 +σ(g([ϕ 
IMU
​
 ,ϕ 
RGB
​
 ]))⊙Wϕ 
RGB
​
 
Assignment Overview 📝
Mini-Example
Encoders: Use a 1D-CNN for IMU data and a 2D-CNN for RGB frames.

Fusion: Combine the encoders using a gated residual block and classify the result with a shared head.

Tasks
Implement and Compare: Code and train models for early, late, and residual fusion strategies using the same dataset.

Robustness Test: Evaluate the trained models on a "noisy" test set (e.g., with added noise or dark frames) to see which fusion strategy's performance degrades the least.

Stretch Goal: For the residual model, visualize the learned gate values against a metric like motion intensity to see if the model has learned to trust certain sensors more under specific conditions.

Code Explanation
1. Setup and Helper Functions 🛠️
This initial section prepares the environment by importing necessary libraries and defining utility functions.

setup_logging: Configures a logging system to save all console output (like training progress) to a file (fl_module4_log.txt), creating a persistent record of the experiment.

set_seed(seed=42): Ensures that the experiments are reproducible.

add_noise_to_images(...): This function simulates a noisy camera feed by adding random Gaussian noise to the test images. This is used to test the robustness of the fusion models.

2. Data Loading and Preprocessing 🧹
The load_and_process_data function is a critical component that prepares the raw sensor and image data for the models.

Loading: It loads the main sensor.csv file and the .npy files for both camera 1 and camera 2.

Optimized Alignment: This is the most computationally intensive step. To ensure that the sensor reading and images for a given sample correspond to the exact same moment in time, the script performs an efficient alignment:

It gets the unique timestamps from the sensor data and both cameras as Python sets.

It finds the intersection of these three sets, which is much faster than traditional array methods. This gives a master list of timestamps that are present in all three data sources.

It then filters the sensor, camera 1, and camera 2 data using this master list, ensuring all data is perfectly synchronized.

Labeling and Splitting: It converts the multi-class activity labels into a binary Fall (0) vs. No Fall (1) format and splits the data into training (60%), validation (20%), and testing (20%) sets.

Scaling: It applies Z-score normalization to the sensor data and scales the image pixel values to be between 0 and 1.

3. Model Architectures 🧠
The script defines base "encoder" networks for each modality and then combines them into three different fusion models.

IMUEncoder: A simple 1D-CNN that processes the sensor data and outputs a compact feature vector.

ImageEncoder: A simple 2D-CNN that processes an image from one of the cameras and outputs its own feature vector.

Fusion Models
EarlyFusionModel: This model feeds the sensor data and images from both cameras into their respective encoders. It then concatenates the resulting three feature vectors into a single, larger vector before passing it to a final classifier.

LateFusionModel: This model has three completely separate branches. Each branch (IMU, Camera 1, Camera 2) produces its own prediction (logit). These three predictions are then concatenated and fed into a final Linear layer that learns the best way to combine them into a final decision.

ResidualFusionModel: This advanced model implements the gated residual strategy. It treats the IMU features as a baseline. It first combines the features from both cameras, then uses a "gate" (a small neural network with a sigmoid activation) to decide how much of this combined image information to add to the IMU features.

4. Training and Evaluation Logic 🏋️‍♂️
These functions manage the model's learning and performance assessment.

train_model(...): This function orchestrates the training loop. It has been updated to handle three inputs, passing the sensor data and both camera images to the model in each step. It calculates the error (loss) and updates the model's parameters to minimize it. It also evaluates performance on the validation set after each epoch.

evaluate_model(...): This function assesses the final performance of a trained model on a test set. It also takes three inputs and returns a detailed classification report, including accuracy, F1-score, sensitivity, and specificity.

5. Main Execution Block & Plotting 📊
This is the final part of the script that runs the entire experiment.

Setup: It creates an output directory, sets configuration parameters (like learning rate and epochs), and initializes the DataLoaders for the training, validation, and test sets.

Noisy Data Creation: It calls add_noise_to_images to create a "noisy" version of the test set for the robustness check.

Train and Evaluate: It iterates through the three fusion models (EarlyFusion, LateFusion, ResidualFusion). For each model, it:

Trains it using the train_model function.

Evaluates it on the clean test data.

Evaluates it on the noisy test data.

Report Results: It compiles the results into a pandas DataFrame, prints a final comparison table to the console, and saves it to a .csv file. The table clearly shows which model performed best and which was most robust to noise (had the smallest performance drop).

Stretch Goal: It takes the trained ResidualFusionModel and runs an evaluation loop to extract the gate's output values. It then creates and saves a scatter plot of these gate values against the motion intensity (calculated from the gyroscope data), visualizing if the model learned to trust the image data more or less depending on the motion.








