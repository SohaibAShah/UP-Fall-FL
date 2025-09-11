# Module 4_1: Multimodal Fusion
---

## **1. Imports and Setup**

- **Standard Libraries:**  
  `os`, `sys`, `copy`, `logging` for file and logging management.
- **Data Science Libraries:**  
  `numpy`, `pandas`, `matplotlib.pyplot` for data handling and visualization.
- **PyTorch:**  
  For deep learning model definition, training, and evaluation.
- **Scikit-learn:**  
  For preprocessing, splitting, and metrics.
- **scipy.stats:**  
  For statistical correlation analysis.

---

## **2. Helper Functions**

### **Logging and Reproducibility**
- `setup_logging(log_dir)`:  
  Sets up logging to both file and console for experiment traceability.
- `set_seed(seed)`:  
  Ensures reproducibility by setting random seeds for numpy and torch.

### **Noise Injection**
- `add_noise_to_images(image_data, noise_level)`:  
  Adds Gaussian noise to image data to simulate sensor noise, then clips values to [0, 1].

---

## **3. Data Loading and Preprocessing**

- `load_and_process_data(data_dir)`:  
  - Loads sensor data (`sensor.csv`) and two camera image arrays.
  - Cleans and aligns timestamps across all modalities.
  - Extracts features and binarizes labels (fall vs. no fall).
  - Splits data into train, validation, and test sets.
  - Scales sensor features and normalizes images.
  - Returns a dictionary of all splits for use in PyTorch DataLoaders.

---

## **4. Model Architectures**

### **IMUEncoder**
- 1D CNN for sensor (IMU) data, outputs a feature vector.

### **ImageEncoder**
- 2D CNN for image data, outputs a feature vector.

### **EarlyFusionModel**
- Encodes each modality, concatenates features, and classifies.

### **LateFusionModel**
- Each modality is classified independently, then their logits are fused and classified.

### **ResidualFusionModel**
- Encodes IMU and both images.
- Fuses image features, computes a **gate value** (learned trust in image modalities).
- Final representation is IMU features plus (gate × image features).
- Classifies the fused representation.
- Can return the gate value for analysis.

---

## **5. Training and Evaluation**

- `train_model(model, train_loader, val_loader, config)`:  
  Standard PyTorch training loop with validation F1-score logging.

- `evaluate_model(model, data_loader, device, return_preds=False)`:  
  Evaluates the model, returns classification report or predictions/probabilities.

---

## **6. Analysis and Experiment Functions**

### **Gate Analysis (for Residual Fusion Model)**
- **Correlation Analysis:**  
  Computes Pearson and Spearman correlation between motion intensity and gate value.
- **Binned Gate Analysis:**  
  Bins motion intensity, computes gate value stats per bin, and plots boxplots.
- **Gate vs. Confidence:**  
  Plots gate value against model prediction confidence.
- **Gate vs. Error:**  
  Compares gate value distributions for correct vs. incorrect predictions.
- **Temporal Gate Analysis:**  
  Plots gate value over time (sample index).
- **Scatter Plot:**  
  Plots gate value vs. motion intensity.
- **Gate Adaptation Under Noisy Images:**  
  Compares gate values for clean vs. noisy images.

---

## **7. Main Execution Block**

1. **Setup:**  
   - Output directory and logging.
   - Training configuration (epochs, learning rate, device).
2. **Data Preparation:**  
   - Loads and splits data.
   - Creates PyTorch DataLoaders for train, validation, test, and noisy test sets.
3. **Model Training and Evaluation:**  
   - Trains and evaluates EarlyFusion, LateFusion, and ResidualFusion models.
   - Evaluates each model on both clean and noisy test data.
   - Logs and saves F1-scores and performance drop due to noise.
4. **Advanced Gate Analysis (Residual Fusion only):**  
   - Extracts gate values and motion intensities from test data.
   - Runs all the analysis/visualization functions described above.
   - Saves all plots and statistics to the output directory.

---

## **What This Code Enables**

- **End-to-end multimodal fusion experiments** for fall detection using IMU and camera data.
- **Robustness analysis**: See how models perform under sensor/image noise.
- **Interpretability**:  
  - Understand how the Residual Fusion model dynamically trusts image vs. IMU data.
  - Analyze how this trust (gate value) relates to motion, prediction confidence, and errors.
- **Visualization and Reporting**:  
  - All results, plots, and statistics are saved for further inspection.

---

## **Summary Table**

| Section         | Purpose                                                      |
|-----------------|-------------------------------------------------------------|
| Data Loading    | Aligns and preprocesses sensor and image data               |
| Model Training  | Trains three fusion models                                  |
| Evaluation      | Tests models on clean and noisy data                        |
| Gate Analysis   | Explains and visualizes how/when the model trusts images    |
| Output          | Saves all results, plots, and logs for reproducibility      |

---

**In short:**  
This script is a comprehensive, reproducible pipeline for multimodal fusion, robustness, and interpretability analysis in fall detection, with a special focus on understanding and visualizing the learned gating mechanism in the Residual Fusion model.