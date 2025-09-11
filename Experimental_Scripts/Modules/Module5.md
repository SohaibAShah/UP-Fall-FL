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

# **The Story: Smarter, Cheaper Fall Detection**

### **1. The Problem**

You want a model that can detect falls using both IMU (sensor) and camera data.  
But:  
- **Cameras are expensive** (energy, computation, privacy).
- Sometimes, **camera data might be missing** (e.g., sensor failure, privacy mode).
- You want a model that is **robust** (works even if cameras are missing) and **efficient** (uses cameras only when needed).

---

### **2. The Hero: GatedResidualFusionModel**

This model is designed to:
- **Fuse IMU and camera data** for best accuracy.
- **Learn when to use the cameras** (gating).
- **Fall back to IMU only** if cameras are missing or not needed.

#### **How does it work?**

```python
class GatedResidualFusionModel(nn.Module):
    ...
    def forward(self, x_csv, x_img1, x_img2, threshold=0.5):
        # Always process the lightweight IMU data
        f_csv = self.imu_encoder(x_csv)
        
        # The gate decides whether to use the cameras
        gate_logit = self.gate(f_csv)
        gate_prob = torch.sigmoid(gate_logit)
```

- The **gate** is a neural network that looks at the IMU features and outputs a probability (`gate_prob`) between 0 and 1.
- This probability represents **how much the model "trusts" the camera data** for this sample.

#### **Training: Soft Gating and Modality Dropout**

During training, the model always computes both "paths":
- **IMU-only path** (for when cameras are missing)
- **Fused path** (IMU + camera)

```python
if self.training:
    # Always compute both paths for gradient flow
    f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
    f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
    fused = f_csv + f_img_combined
    out_fused = self.fused_classifier(fused)
    out_imu_only = self.imu_only_classifier(f_csv)
    # The final prediction is a mix, weighted by the gate
    return gate_prob * out_fused + (1 - gate_prob) * out_imu_only
```

- The output is a **weighted sum**:  
  - If `gate_prob` is high, the model relies more on the fused (IMU+camera) path.
  - If `gate_prob` is low, it relies more on the IMU-only path.

#### **Modality Dropout: Training for Robustness**

In the training loop, you **randomly zero out the camera images** (modality dropout):

```python
if model.training and np.random.rand() < config['dropout_prob']:
    x_img1_b.zero_()
    x_img2_b.zero_()
```

- This forces the model to **learn to handle missing camera data**.
- The model can't always "cheat" by using the camera; it must learn to use IMU when needed.

---

### **3. Inference: Hard Gating**

At test time, the model makes a **hard decision**:  
Should it use the camera for this sample or not?

```python
else:
    use_images = (gate_prob > threshold).float()
    ...
    # Select the output from the correct head based on the gate's decision
    return use_images * out_fused + (1 - use_images) * out_imu_only, gate_prob
```

- If `gate_prob > threshold`, use the fused (IMU+camera) prediction.
- Otherwise, use the IMU-only prediction.
- This means the model **dynamically decides** for each sample whether to "pay the price" of using the camera.

---

### **4. Assignment 1: Robustness to Missing Modality**

You train two models:
- **Baseline:** No modality dropout.
- **Robust:** With 30% modality dropout.

You test both on data where the camera images are **always missing** (all zeros):

```python
X_test_img_absent = torch.zeros_like(torch.from_numpy(data_splits['X_test_img1']))
absent_modality_loader = DataLoader(..., X_test_img_absent, X_test_img_absent, ...)
f1_baseline_absent, _ = evaluate_gated_model(baseline_model, absent_modality_loader, config['device'], threshold=0.0)
f1_robust_absent, _ = evaluate_gated_model(robust_model, absent_modality_loader, config['device'], threshold=0.0)
```

- **Result:**  
  - Baseline fails (F1=0).
  - Robust model succeeds (F1 ≈ 0.92).

---

### **5. Assignment 2: Gating for Efficiency**

You evaluate the robust model on normal test data, using a **threshold of 0.5** for the gate.

```python
f1_gated, trigger_rate = evaluate_gated_model(robust_model, test_loader, config['device'], threshold=0.5)
```

- **trigger_rate:** Fraction of samples where the camera is used.
- **You calculate energy/cost savings** by comparing always-using-camera vs. gating.

---

### **6. Stretch Goal: Learning the Threshold τ**

Suppose you want to **limit camera usage to 40%** of the samples (energy budget).

#### **How do you find the right threshold?**

1. **Collect all gate probabilities** on the validation set:

```python
all_gate_probs = []
with torch.no_grad():
    for x_csv, x_img1, x_img2, y in val_loader:
         _, gate_probs = robust_model(x_csv.to(config['device']), x_img1.to(config['device']), x_img2.to(config['device']), threshold=0.0)
         all_gate_probs.extend(gate_probs.cpu().numpy().flatten())
```

2. **Set τ so that only 40% of samples exceed it:**

```python
learned_threshold = np.quantile(all_gate_probs, 1 - target_trigger_rate)
```

- This means:  
  - Only the top 40% of gate probabilities (most confident in camera) will trigger camera usage.

3. **Evaluate with the learned threshold:**

```python
f1_budget, actual_trigger_rate = evaluate_gated_model(robust_model, test_loader, config['device'], learned_threshold)
```

- **Result:**  
  - Camera is used on ≈40% of samples.
  - F1-score remains high.

---

## **Summary: The Smart, Adaptive Model**

- **During training:**  
  - The model learns to use both IMU and camera, but is forced (via modality dropout) to also learn to work with IMU alone.
  - The gate learns to predict, from IMU features, when the camera will be helpful.

- **During inference:**  
  - The model uses the gate to decide, for each sample, whether to use the camera or not.
  - You can set the threshold τ to control the trade-off between accuracy and resource usage.

- **You get:**  
  - **Robustness:** Works even if cameras are missing.
  - **Efficiency:** Uses cameras only when needed.
  - **Control:** You can tune τ to meet your energy or privacy budget.

---

**This is a modern, practical approach for edge AI and sensor fusion—smart, adaptive, and resource-aware!**