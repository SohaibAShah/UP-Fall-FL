# Module 5: Missing Modalities and an IMU-Driven Gate

## Intuition 💡
In real-world deployments, sensors can fail (e.g., a camera is turned off or blocked). This module focuses on training a model to be robust against missing data from a modality. It also introduces an energy-saving technique where a lightweight sensor (IMU) acts as a "gate" to decide when to activate a more power-hungry sensor (camera).

---
## Key Equations 🧠

* **Modality Dropout**: During training, we randomly create a binary mask, $m$, and multiply it with the input data, $X$, to simulate missing sensors. The model then learns to minimize the expected loss across these different dropout scenarios:
    $$\min_{\theta} \mathbb{E}_m [\ell(f_\theta(X \odot m), y)]$$

* **Energy/Latency Budget Model**: The expected computational cost can be modeled as the cost of the lightweight sensor ($C_{light}$) plus the cost of the heavy sensor ($C_{heavy}$) multiplied by the probability that the gate activates it ($p_{trigger}$):
    $$C = C_{light} + p_{trigger} \cdot C_{heavy}$$

* **Gate as a Classifier**: The gate itself is a simple classifier that uses only the IMU features ($\phi_{IMU}$) to produce a trigger probability, $p_{trigger}$. The more complex camera branch is only activated if this probability exceeds a certain threshold, $\tau$:
    $$p_{trigger} = \sigma(g(\phi_{IMU}(X_I)))$$

---
## Mini-Example 🧪
Train a residual fusion model where, during each training step, there is a 30% chance that the RGB camera data is dropped (zeroed out). When deployed, this model uses a gate that only activates the RGB processing when the IMU data indicates low confidence or a high likelihood of a fall.

---
## Small Assignment 🎯

* Train two models—one with modality dropout and one without. Measure and compare their accuracy on a test set where the RGB data is completely absent.
* Estimate the expected latency or energy savings achieved by the gating mechanism compared to always having the camera on. Simple proxies for cost are acceptable.

---
## Stretch Goal 🌟
Instead of using a fixed threshold $\tau$, develop a method to learn or select a $\tau$ that meets a predefined target energy budget (e.g., "the camera can only be active 25% of the time").

-----

### Code Explanation

This script is designed to build and evaluate an advanced multimodal model for fall detection that is both **robust** to sensor failure and **energy-efficient**. It achieves this by combining several key concepts.

#### 1\. Setup and Data Loading

  * **Setup**: The initial part of the script imports necessary libraries (`torch`, `pandas`, `sklearn`, etc.) and defines helper functions, including `set_seed` for reproducible results.
  * **`load_and_process_data`**: This function is the data pipeline. It loads the IMU sensor data from `sensor.csv` and image data from two cameras. Its most critical task is to perform an **optimized alignment** of these three data sources, ensuring that every IMU reading, camera 1 image, and camera 2 image in a given sample correspond to the exact same moment in time. Finally, it splits the data into training, validation, and test sets.

-----

#### 2\. The GatedResidualFusionModel Architecture 🧠

This is the core of the project. It's a sophisticated model designed to be both accurate and efficient.

  * **Encoders**: It has three "encoder" networks: one `IMUEncoder` (a 1D-CNN) to process sensor data, and two `ImageEncoder`s (2D-CNNs) to process data from the two cameras. These encoders extract meaningful features from each data type.
  * **IMU-driven Gate**: This is a small, separate neural network that looks *only* at the features from the lightweight IMU sensor. Its job is to make a quick decision: "Is this situation ambiguous or potentially a fall?" It outputs a probability (p\_trigger) indicating how confident it is that the cameras are needed.
  * **Two Classifiers ("Heads")**: The model has two decision-making paths:
    1.  An **`imu_only_classifier`** that makes a prediction using only the IMU data. This is the default, low-energy path.
    2.  A **`fused_classifier`** that combines features from the IMU and both cameras. This is the powerful, high-energy path.
  * **Conditional Forward Pass**:
      * **During Training**: The model always calculates both paths. It uses a special technique called **modality dropout**, where it randomly ignores the camera data 30% of the time. This forces the `imu_only_classifier` to become strong on its own.
      * **During Evaluation (Inference)**: The gate makes a hard decision. If its trigger probability is above a set threshold (e.g., 0.5), the model activates the powerful image encoders and uses the `fused_classifier`. If not, it saves energy by keeping the image encoders off and uses the `imu_only_classifier`.

-----

#### 3\. Training and Evaluation

  * **`train_gated_model`**: This function trains the model. Its most important feature is the implementation of **modality dropout**. Before feeding data to the model, it randomly zeroes out the image tensors with a 30% probability, simulating camera failures and forcing the model to learn to be robust.
  * **`evaluate_gated_model`**: This function tests the trained model. It measures the F1-score (a key performance metric for fall detection) and the **trigger rate**—the percentage of time the IMU-driven gate decided to activate the cameras.

-----

#### 4\. Main Execution Block (The Assignments)

This final section runs the experiments defined in the assignment.

  * **Assignment 1: Robustness Test**:
    1.  It first trains a **baseline model** without modality dropout.
    2.  It then trains the **robust model** with 30% modality dropout.
    3.  It evaluates both models on a special test set where the camera data has been completely zeroed out, simulating a total camera failure. This directly tests which model is more robust.
  * **Assignment 2: Energy Savings Test**:
    1.  It takes the trained robust model and evaluates it on a normal test set.
    2.  It calculates the camera trigger rate and uses a simple proxy (`cost_imu = 1`, `cost_image = 10`) to estimate the computational savings achieved by the gating mechanism compared to always having the cameras on.
  * **Stretch Goal: Learning a Threshold (τ)**:
    1.  Instead of using a fixed threshold of 0.5, this part calculates a new threshold `τ` that would cause the cameras to activate on only the top 40% most uncertain samples.
    2.  It then re-evaluates the model with this new, budget-aware threshold to see how performance is affected.


---
# Results

## **Assignment 1: Robustness Evaluation**

### **What was tested?**
- **Baseline Model (No Dropout):**  
  Trained normally, always expects both IMU and camera data.
- **Robust Model (With 30% Dropout):**  
  During training, randomly "hides" (zeros out) the camera images 30% of the time, forcing the model to learn to rely on IMU data alone when needed.

### **How were they evaluated?**
- Both models were tested on data where the camera images were **completely absent** (all zeros).

### **Results:**
```
                Model  F1 on RGB-Absent Data
Baseline (No Dropout)               0.000000
Robust (With Dropout)               0.923077
```

### **What does this mean?**
- **Baseline Model:**  
  Completely fails (F1=0) when camera images are missing, because it never learned to handle this situation.
- **Robust Model:**  
  Performs very well (F1 ≈ 0.92) even when images are missing, because it was trained to handle such cases using only IMU data.

**Takeaway:**  
**Modality dropout** during training makes your model robust to missing sensor modalities.

---

## **Assignment 2: Gating Performance (Threshold=0.5)**

### **What was tested?**
- The robust model was evaluated on normal test data, using a **gating mechanism** to decide when to use the camera images.

### **Key Metrics:**
| Metric                    | Value    |
|---------------------------|----------|
| F1-Score (Fall)           | 0.9807   |
| Camera Trigger Rate       | 57.16%   |
| Cost Before Gating        | 21       |
| Cost After Gating         | 12.43    |
| Savings                   | 40.80%   |

- **F1-Score (Fall):**  
  The model is highly accurate at detecting falls.
- **Camera Trigger Rate:**  
  The model only uses the camera images for about 57% of the test samples; for the rest, it relies on IMU data alone.
- **Cost Before Gating:**  
  If you always use both IMU and both cameras, the "cost" is 21 units per sample (IMU=1, each camera=10).
- **Cost After Gating:**  
  With gating, the average cost drops to 12.43 units per sample.
- **Savings:**  
  About **41% reduction** in computational/energy cost, with almost no loss in accuracy.

**Takeaway:**  
**Gating** allows you to save significant resources by only using expensive sensors (cameras) when needed, while maintaining high accuracy.

---

## **Stretch Goal: Learn Threshold τ for 40% Energy Budget**

### **What was tested?**
- The gating threshold was **automatically set** so that the camera is used for only 40% of the samples (target energy budget).

### **Results:**
- **Learned Threshold τ:** 0.6540
- **Performance:**  
  - F1-Score = 0.9698  
  - Actual Camera Trigger Rate = 39.98%

### **What does this mean?**
- By adjusting the gating threshold, you can **control the trade-off** between energy/cost and accuracy.
- Even when using the camera only 40% of the time, the model still achieves **very high accuracy**.

**Takeaway:**  
You can **tune the gating threshold** to meet a specific resource budget, and the model will adapt, maintaining strong performance.

---

## **Overall Summary**

- **Modality dropout** makes your model robust to missing sensors.
- **Gating** enables smart, dynamic use of expensive sensors, saving resources.
- **You can control the resource/accuracy trade-off** by adjusting the gating threshold.

**In practice:**  
Your system can run efficiently on edge devices, using cameras only when needed, and still accurately detect falls—even if a sensor fails or is missing.


---

# **The Story of Smart, Efficient Fall Detection**

### **1. The Problem**

Imagine you’re building a smart fall detection system for elderly care. You have two types of sensors:
- **IMU (motion sensors):** Always available, cheap, and fast.
- **Cameras:** Expensive (in terms of energy and computation), but provide rich information.

You want your system to:
- **Detect falls accurately.**
- **Save energy by using cameras only when necessary.**
- **Be robust—even if a camera fails or is missing.**

---

### **2. The Hero: GatedResidualFusionModel**

You design a neural network called **GatedResidualFusionModel**. Here’s how it works:

#### **a. Two Brains, One Decision**
- **IMU Encoder:** Processes the motion data and creates a summary (feature vector).
- **Image Encoders:** Each camera image is processed separately, then their features are combined.
- **Fusion:** The IMU and image features are combined, but here’s the twist:  
  The model can decide, for each sample, whether to use the camera images or just the IMU.

#### **b. The Gatekeeper (Gating Mechanism)**
- The model has a **gate**—a little neural network that looks at the IMU features and outputs a value between 0 and 1 (after a sigmoid).
- **Gate value close to 1:** "I trust the images, use them!"
- **Gate value close to 0:** "Just use the IMU, skip the images!"

#### **c. Two Heads Are Better Than One**
- The model has two "heads" (output layers):
  - **Fused Head:** For when both IMU and images are used.
  - **IMU-only Head:** For when only IMU is used.
- The final prediction is a blend of these two heads, weighted by the gate value.

---

### **3. Training: Teaching the Model to Be Robust**

#### **a. Modality Dropout (The Survival Drill)**
- During training, sometimes (30% of the time) you **zero out the images**—pretend the cameras are broken!
- This forces the model to learn to make good predictions **even if the images are missing**.
- The model learns: "Sometimes I have to rely only on IMU, and that’s okay."

#### **b. Training Loop**
- For each batch, you might randomly zero out the images.
- The model always computes both heads, so it learns from both situations.
- The loss is computed as usual, and the model gets better at both "with images" and "without images" cases.

---

### **4. Inference: Making Smart Decisions**

#### **a. The Gate Makes the Call**
- When the model is deployed (inference mode), it looks at the IMU data and the gate decides:
  - If the gate value is **above a threshold** (say, 0.5), use the images.
  - If **below**, skip the images and use only IMU.
- This means the system **dynamically decides** when to spend energy on camera processing.

#### **b. Energy/Latency Savings**
- You can measure how often the camera is used (trigger rate) and calculate the energy/cost savings.

---

### **5. Learning the Best Threshold (τ)**

#### **a. The Energy Budget**
- Suppose you want to use the cameras only 40% of the time (to save energy).
- You run the model on the validation set and collect all the gate values.
- You **sort the gate values** and pick the threshold (τ) so that only the top 40% of samples (with highest gate values) will use the cameras.
- This is done using `np.quantile`.

#### **b. Deploying with the Learned Threshold**
- Now, when the model runs, it uses this learned τ as the cutoff.
- You check: Does the actual camera usage match your budget? How is the accuracy?

---

### **6. The Results**

- **Baseline Model:** Fails when images are missing (never learned to cope).
- **Robust Model (with modality dropout):** Succeeds even if images are missing.
- **Gating:** Lets you save energy by using cameras only when needed, with almost no loss in accuracy.
- **Learned Threshold:** Lets you meet a specific energy budget, with the model adapting its behavior.

---

## **Summary Table**

| Component         | What it does                                         |
|-------------------|-----------------------------------------------------|
| Gating Mechanism  | Decides, per sample, whether to use images or not   |
| Modality Dropout  | Trains model to handle missing images robustly      |
| Threshold τ       | Controls the trade-off between accuracy and energy  |
| Two Output Heads  | One for fused (IMU+image), one for IMU-only         |

---

## **In Short**

Your code builds a **smart, energy-efficient, and robust fall detector** that:
- **Learns to handle missing data** (modality dropout).
- **Dynamically decides** when to use expensive sensors (gating).
- **Lets you control energy/accuracy trade-off** by learning the best threshold.

It’s like training a lifeguard who knows when to call for backup (cameras) and when they can handle things on their own (IMU)—and who can keep working even if the backup is unavailable!