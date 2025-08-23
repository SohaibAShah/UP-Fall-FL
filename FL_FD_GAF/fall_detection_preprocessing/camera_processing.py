import os
import re
import numpy as np
from PIL import Image
import imageio
from zipfile import ZipFile
import shutil
from utils import re_value
import pickle

def process_camera_data(camera_zip_path, temp_camera_path, output_dir='dataset'):
    """Process camera data from zip files and compute average difference images.
    
    Args:
        camera_zip_path (str): Path to directory containing zip files.
        temp_camera_path (str): Path for temporary extracted images.
    
    Returns:
        dict: Camera_data with keys (Subject, Activity, Trial) and values as processed images.
    """

    camera_data_path = os.path.join(temp_camera_path, 'Camera_data.pkl')
    # Check for existing Camera_data.pkl
    if os.path.exists(camera_data_path):
        print(f"Loading Camera data from {camera_data_path}...")
        with open(camera_data_path, 'rb') as f:
            Camera_data = pickle.load(f)
        return Camera_data
    
    count = 0
    Camera_data = {}
    
    # Iterate over subjects, activities, trials, and cameras
    for sub_ in range(1, 18):  # Subjects 1 to 17
        if sub_ in [5, 9]:  # Skip Subjects 5 and 9
            continue
        for act_ in range(1, 12):  # Activities 1 to 11
            for trial_ in range(1, 4):  # Trials 1 to 3
                if sub_ == 8 and act_ == 11 and trial_ in [2, 3]:  # Skip missing trials
                    print(f'Skipping Subject{sub_}Activity{act_}Trial{trial_} (missing data)')
                    continue
                for cam_ in [1, 2]:  # Cameras 1 and 2
                    sub = f'Subject{sub_}'
                    act = f'Activity{act_}'
                    trial = f'Trial{trial_}'
                    cam = f'Camera{cam_}'
                    key = (sub_, act_, trial_)
                    
                    # Extract zip file
                    zip_path = os.path.join(camera_zip_path, f'{sub}{act}{trial}{cam}.zip')
                    extract_dir = os.path.join(temp_camera_path, f'{sub}{act}{trial}{cam}')
                    
                    try:
                        with ZipFile(zip_path, 'r') as zipObj:
                            zipObj.extractall(extract_dir)
                    except Exception as e:
                        print(f"Error extracting {zip_path}: {e}")
                        continue
                    
                    # Process images
                    all_gray_img = []
                    photos_filenames = []
                    for root, _, filenames in os.walk(extract_dir):
                        for filename in sorted(filenames):  # Sort for chronological order
                            if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename, re.IGNORECASE):
                                photos_filenames.append(os.path.join(root, filename))
                                count += 1
                                if count % 5000 == 0:
                                    print('{} : {}'.format(os.path.join(root, filename), count))

                    # Limit to 140 images
                    for j in range(min(140, len(photos_filenames))):
                        try:
                            temp_rgb_arr = imageio.imread(photos_filenames[j])  # RGB array
                            temp_rgb = Image.fromarray(temp_rgb_arr).convert('RGB')  # RGB image
                            tem_gray = temp_rgb.resize((140, 140)).convert('L')  # Grayscale, 140x140
                            all_gray_img.append(tem_gray)
                        except Exception as e:
                            print(f"Error processing {photos_filenames[j]}: {e}")
                            continue
                    
                    # Calculate differences
                    if len(all_gray_img) > 1:
                        delt_list = []
                        for k in range(len(all_gray_img) - 1):
                            delt = np.abs(np.array(all_gray_img[k + 1]).astype(np.int64) - 
                                         np.array(all_gray_img[k]).astype(np.int64))
                            delt_list.append(delt)
                        if delt_list:
                            delt_gray_arr = sum(delt_list) / len(delt_list)  # Average difference
                            arr = re_value(delt_gray_arr)
                            Camera_data[key] = arr
                    
                    # Clean up extracted directory
                    shutil.rmtree(extract_dir, ignore_errors=True)
    # Save Camera_data
    os.makedirs(output_dir, exist_ok=True)

    with open(camera_data_path, 'wb') as f:
        pickle.dump(Camera_data, f)
    print(f"Saved Camera data to {camera_data_path}")

    return Camera_data