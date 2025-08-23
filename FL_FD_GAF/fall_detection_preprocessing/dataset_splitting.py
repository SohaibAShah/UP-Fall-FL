import os
import pickle

def split_and_save_data(GAF_Camera_data, output_dir='dataset'):
    """Split fused data into train and test sets and save as pickle files.
    
    Args:
        GAF_Camera_data (dict): Fused sensor and camera data.
        output_dir (str): Directory to save Train.pkl and Test.pkl.
    
    Returns:
        tuple: (Train_data, Test_data) dictionaries.
    """
    Train_data = {}
    Test_data = {}
    
    label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10}
    
    for key, value in GAF_Camera_data.items():
        activity = int(key[1])
        label = label_map.get(activity, -1)
        if label == -1:
            continue
        value.append(label)
        
        if key[2] in [1, 2]:
            Train_data[key] = value
        elif key[2] == 3:
            Test_data[key] = value
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'Train.pkl'), 'wb') as f:
        pickle.dump(Train_data, f)
    with open(os.path.join(output_dir, 'Test.pkl'), 'wb') as f:
        pickle.dump(Test_data, f)
    
    print(f"Saved {len(Train_data)} training samples and {len(Test_data)} test samples.")
    return Train_data, Test_data