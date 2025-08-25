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

-----

### Output Explanation

Your output shows that the experiments ran successfully. Let's break down what the results mean.

#### Assignment 1: Robustness Comparison

```
--- Robustness Comparison ---
                Model  F1 on RGB-Absent Data
Baseline (No Dropout)               0.000751
Robust (With Dropout)               0.952785
```

  * **Insight**: This is a dramatic and successful result. When tested on data with **no images**, the **baseline model completely failed** (F1-score is near zero), as it had become overly reliant on the cameras during training.
  * In contrast, the **robust model**, which was trained with modality dropout, achieved an excellent **F1-score of 0.95**. This proves that forcing the model to occasionally work without images made it highly effective at using the IMU data alone when needed.

-----

#### Assignment 2: Gating Performance

```
--- Gating Performance (Threshold=0.5) ---
                    Metric  Value
           F1-Score (Fall) 0.9845
       Camera Trigger Rate 91.61%
Cost Before Gating (Proxy)     21
 Cost After Gating (Proxy)  19.32
                   Savings  7.99%
```

  * **Insight**: With a standard 0.5 threshold, the model achieved a very high **F1-score of 0.98**. However, to do so, the gate was quite cautious and decided to activate the cameras **91.61%** of the time.
  * This resulted in a modest **energy/latency saving of 7.99%**. While the model is accurate, it's not yet very efficient.

-----

#### Stretch Goal: Learning a Threshold for a Budget

```
--- Stretch Goal: Learn Threshold τ for 40% Energy Budget ---
Learned Threshold τ = 0.9054 to meet 40% budget.
Performance with learned τ: F1-Score=0.9384, Actual Trigger Rate=40.09%
```

  * **Insight**: This is the most interesting result. To meet the strict energy budget of only using the cameras 40% of the time, the model needed a much higher confidence threshold (`τ = 0.9054`).
  * Even with the cameras off for 60% of the test samples, the model maintained a very strong **F1-score of 0.9384**.
  * This demonstrates a powerful trade-off: you can achieve a significant (**\~57%** reduction in camera usage compared to the 91.61% trigger rate) while only sacrificing a small amount of performance. This is the core benefit of an IMU-driven gate.