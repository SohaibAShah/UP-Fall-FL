# data_loader.py

import pandas as pd
import os
import re
import numpy as np
import cv2
from zipfile import ZipFile
import shutil
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Import configurations
import config


def ResizeImage(IM, DesiredWidth, DesiredHeight):
    """
    Resizes an image to the desired width and height.

    Args:
        IM (numpy.ndarray): The input image (grayscale or color).
        DesiredWidth (int): The target width for the image.
        DesiredHeight (int): The target height for the image.

    Returns:
        numpy.ndarray: The resized image.
    """
    OrigWidth = float(IM.shape[1])
    OrigHeight = float(IM.shape[0])
    Width = DesiredWidth
    Height = DesiredHeight

    if (Width == 0) and (Height == 0):
        return IM

    if Width == 0:
        Width = int((OrigWidth * Height) / OrigHeight)

    if Height == 0:
        Height = int((OrigHeight * Width) / OrigWidth)

    dim = (Width, Height)
    resizedIM = cv2.resize(IM, dim, interpolation=cv2.INTER_NEAREST)
    return resizedIM


def load_img(start_sub, end_sub, start_act, end_act, start_cam, end_cam, DesiredWidth, DesiredHeight):
    """
    Loads and processes images from zipped camera files for specified subjects, activities, and cameras.
    Extracts zip files, resizes images, and handles specific problematic files as per original notebook.

    Args:
        start_sub (int): Starting subject ID.
        end_sub (int): Ending subject ID.
        start_act (int): Starting activity ID.
        end_act (int): Ending activity ID.
        start_cam (int): Starting camera ID.
        end_cam (int): Ending camera ID.
        DesiredWidth (int): Target width for resizing images.
        DesiredHeight (int): Target height for resizing images.

    Returns:
        tuple: A tuple containing:
            - IMG (list): List of processed (resized) images.
            - name_img (list): List of original image file paths.
    """
    IMG = []
    count = 0
    name_img = []

    # Ensure the temporary extraction directory exists
    os.makedirs(config.TEMP_CAMERA_EXTRACTION_DIR, exist_ok=True)

    for sub_ in range(start_sub, end_sub + 1):
        sub = 'Subject' + str(sub_)

        for act_ in range(start_act, end_act + 1):
            act = 'Activity' + str(act_)

            for trial_ in range(1, 3 + 1): # Assuming 3 trials per activity
                trial = 'Trial' + str(trial_)

                # Specific exclusion criteria from the original notebook
                # This explicitly skips Subject8 Activity11 Trial2 and Trial3 for any camera,
                # as well as specific hardcoded problematic paths encountered.
                if (sub_ == 8 and act_ == 11 and (trial_ == 2 or trial_ == 3)):
                    print('----------------------------NULL---------------------------------') #cite: 1
                    continue #cite: 1

                for cam_ in range(start_cam, end_cam + 1):
                    cam = 'Camera' + str(cam_)
                    
                    zip_file_path = os.path.join(config.DOWNLOADED_CAMERA_FILES_DIR, sub + act + trial + cam + '.zip')
                    extract_to_dir = os.path.join(config.TEMP_CAMERA_EXTRACTION_DIR, sub + act + trial + cam)

                    if not os.path.exists(zip_file_path):
                        print(f"Warning: Zip file not found: {zip_file_path}. Skipping.")
                        continue

                    # Extract zip file
                    with ZipFile(zip_file_path, 'r') as zipObj: #cite: 1
                        zipObj.extractall(extract_to_dir) #cite: 1

                    # Walk through extracted files to load images
                    for root, dirnames, filenames in os.walk(extract_to_dir): #cite: 1
                        for filename in filenames: #cite: 1
                            if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename): #cite: 1
                                filepath = os.path.join(root, filename) #cite: 1
                                count += 1 #cite: 1
                                if count % 5000 == 0: #cite: 1
                                    print('{} : {} '.format(filepath, count)) #cite: 1
                                
                                # Specific problematic file check from the original notebook
                                if filepath == 'CAMERA_TEMP/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png': #cite: 1
                                    print('----------------------------NO SHAPE---------------------------------') #cite: 1
                                    continue #cite: 1
                                if filepath == 'CAMERA/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png' :
                                    print('----------------------------NO SHAPE---------------------------------')
                                    continue
                                elif len(filepath) > 70 :
                                    print(' {} : Invalid image'.format(filepath))
                                    continue  #cite: 1
                                
                                # General filepath length check from original, if not specific problematic path
                                elif len(filepath) > 70 :
                                    print(' {} : Invalid image (general length)'.format(filepath))
                                    continue

                                try:
                                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # Load as grayscale (0) as per original. #cite: 1
                                    if img is None:
                                        print(f"Warning: Could not read image {filepath}. Skipping.")
                                        continue
                                    
                                    # Check for empty images (e.g., if imread fails silently)
                                    if img.size == 0:
                                        print(f"Warning: Empty image file {filepath}. Skipping.")
                                        continue

                                    resized = ResizeImage(img, DesiredWidth, DesiredHeight) #cite: 1
                                    IMG.append(resized) #cite: 1
                                    name_img.append(filepath) #cite: 1
                                except Exception as e:
                                    print(f"Error processing image {filepath}: {e}. Skipping.")
                                    continue

                    # Clean up: Remove the extracted directory after processing
                    if os.path.exists(extract_to_dir):
                        shutil.rmtree(extract_to_dir) #cite: 1
    return IMG, name_img #cite: 1


def handle_name(path_name):
    """
    Extracts and formats the timestamp from image file paths.

    Args:
        path_name (list): A list of image file paths.

    Returns:
        list: A list of formatted timestamps.
    """
    img_name = []
    for path in path_name: #cite: 1
        # Extract the relevant part of the filename (timestamp) based on path length
        # This logic is directly translated from the original notebook and is very specific
        if len(path) == 68: #cite: 1
            img_name.append(path[38:64]) #cite: 1
        elif len(path) == 69: #cite: 1
            img_name.append(path[39:65]) #cite: 1
        # The original code's 'else' condition for path[40:66] seems to be for
        # paths of length 70, which were previously marked as 'Invalid image'.
        # This part might need re-evaluation based on actual data paths.
        # For now, keeping the original logic which implies skipping lengths > 70 earlier.
        # If len(path) > 70 images are allowed, this 'else' would capture them
        # if path[40:66] is still the correct slice for timestamp.
        elif len(path) == 70:
             img_name.append(path[40:66])
        else: # Handle unexpected lengths, perhaps log a warning
            print(f"Warning: Unexpected path length for timestamp extraction: {path}")
            # Decide on appropriate action: skip, append a placeholder, or raise an error
            img_name.append(None) # Or path relevant slice if possible from more robust regex

    handle = []
    for name in img_name: #cite: 1
        if name is None: # Skip if timestamp extraction failed
            handle.append(None)
            continue
        # Replace the 13th character (index 13) with ':'
        a1 = name[:13] + ':' + name[14:] # Correct way to replace character at specific index #cite: 1
        # Replace the 16th character (index 16) with ':' in the *new* string a1
        a2 = a1[:16] + ':' + a1[17:] # Correct way to replace character at specific index #cite: 1
        handle.append(a2) #cite: 1
    return handle #cite: 1


def ShowImage(ImageList, nRows=1, nCols=2, WidthSpace=0.00, HeightSpace=0.00):
    """
    Displays a list of images in a grid.

    Args:
        ImageList (list): List of images (numpy arrays) to display.
        nRows (int): Number of rows in the image grid.
        nCols (int): Number of columns in the image grid.
        WidthSpace (float): Horizontal spacing between images.
        HeightSpace (float): Vertical spacing between images.
    """
    gs = gridspec.GridSpec(nRows, nCols) #cite: 1
    gs.update(wspace=WidthSpace, hspace=HeightSpace) # set the spacing between axes. #cite: 1
    plt.figure(figsize=(20, 20)) #cite: 1
    for i in range(len(ImageList)): #cite: 1
        ax1 = plt.subplot(gs[i]) #cite: 1
        ax1.set_xticklabels([]) #cite: 1
        ax1.set_yticklabels([]) #cite: 1
        ax1.set_aspect('equal') #cite: 1
        plt.subplot(nRows, nCols, i + 1) #cite: 1
        image = ImageList[i].copy() #cite: 1
        if len(image.shape) < 3: #cite: 1
            plt.imshow(image, cmap=plt.cm.gray) # Use cmap for grayscale images #cite: 1
        else: #cite: 1
            plt.imshow(image) #cite: 1
        plt.title("Image " + str(i)) #cite: 1
        plt.axis('off') #cite: 1
    plt.show() #cite: 1