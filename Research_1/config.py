# config.py
import os
# --- Dataset Paths ---
# Base directory for the UP-Fall Dataset
# Assumes 'UP-Fall Dataset' is in the same directory as your Python scripts
DATASET_BASE_DIR = 'UP-Fall Dataset'
SENSOR_DATA_PATH = os.path.join(DATASET_BASE_DIR, 'Imp_sensor.csv')
DOWNLOADED_CAMERA_FILES_DIR = os.path.join(DATASET_BASE_DIR, 'downloaded_camera_files')
# Temporary directory for extracting zip files
TEMP_CAMERA_EXTRACTION_DIR = 'CAMERA_TEMP'

# --- Image Processing Parameters ---
DEFAULT_IMAGE_WIDTH = 32
DEFAULT_IMAGE_HEIGHT = 32

# --- Subject, Activity, Camera Ranges ---
START_SUBJECT = 1
END_SUBJECT = 1
START_ACTIVITY = 1
END_ACTIVITY = 1
START_CAMERA = 1
END_CAMERA = 2 # Changed to include both camera 1 and 2 in a single run if desired

# --- Output File Names ---
IMAGE_FILE_PREFIX = 'image_'
NAME_FILE_PREFIX = 'name_'
LABEL_FILE_PREFIX = 'label_'
NPY_EXTENSION = '.npy'


# --- General Configuration ---
RANDOM_SEED = 42
NUM_CLASSES = 12 # Total number of classes in your dataset

# --- Dataset Paths (Assuming 'UP-Fall Dataset' is in the same directory) ---
DATASET_DIR = 'UP-Fall Dataset'
SENSOR_CSV_PATH = os.path.join(DATASET_DIR, 'Imp_sensor.csv')
IMAGE_CAM1_NPY = os.path.join(DATASET_DIR, 'image_1.npy')
LABEL_CAM1_NPY = os.path.join(DATASET_DIR, 'label_1.npy')
NAME_CAM1_NPY = os.path.join(DATASET_DIR, 'name_1.npy') # Assuming name files are also saved

IMAGE_CAM2_NPY = os.path.join(DATASET_DIR, 'image_2.npy')
LABEL_CAM2_NPY = os.path.join(DATASET_DIR, 'label_2.npy')
NAME_CAM2_NPY = os.path.join(DATASET_DIR, 'name_2.npy') # Assuming name files are also saved

# --- Model Saving Paths ---
SAVED_MODELS_DIR = 'Saved Model/Experiments'
MLP_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'MLP_csv.keras')
XGB_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'XGB_model.sav')
CATBOOST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'Catboost_model.sav')
CNN_IMG1_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'CNN_img1.keras')
CNN_IMG2_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'CNN_img2.keras')
CNN_CONCAT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'model_img12.keras')
CNN_CSV_IMG_CONCAT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'model_concatenate_all.keras')
RF_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'RandomForest_model.sav')
SVM_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'SVM_model.sav')
KNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'KNN_model.sav')


# --- Class Names (from your notebook) ---
CLASS_NAMES = [
    '?????',
    'Falling hands',
    'Falling knees',
    'Falling backwards',
    'Falling sideward',
    'Falling chair',
    'Walking',
    'Standing',
    'Sitting',
    'Picking object',
    'Jumping',
    'Laying'
]