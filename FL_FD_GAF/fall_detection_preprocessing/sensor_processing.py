
import os
import pickle
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField

def process_sensor_data(sensor_path):
    """Process sensor data from CompleteDataSet.csv and apply GAF transformation.
    
    Args:
        sensor_path (str): Path to directory containing CompleteDataSet.csv.
    
    Returns:
        dict: GAF_data with keys (Subject, Activity, Trial) and values as sensor GAFs.
    """
    # Load sensor data
    sensor_file = pd.read_csv(os.path.join(sensor_path, 'CompleteDataSet.csv'), 
                             skiprows=2, header=None)
    
    # Keep columns for accelerometer, angular velocity, and metadata
    keep_columns = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 
                    18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 43, 44, 45]
    
    time = pd.to_datetime(sensor_file.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f')
    values = np.array(sensor_file.iloc[:, keep_columns])
    
    df = pd.DataFrame(values, index=time, 
                      columns=['AnkleAccelerometer_x', 'AnkleAccelerometer_y', 'AnkleAccelerometer_z', 
                               'AnkleAngularVelocity_x', 'AnkleAngularVelocity_y', 'AnkleAngularVelocity_z', 
                               'RightPocketAccelerometer_x', 'RightPocketAccelerometer_y', 'RightPocketAccelerometer_z',
                               'RightPocketAngularVelocity_x', 'RightPocketAngularVelocity_y', 'RightPocketAngularVelocity_z', 
                               'BeltAccelerometer_x', 'BeltAccelerometer_y', 'BeltAccelerometer_z', 
                               'BeltAngularVelocity_x', 'BeltAngularVelocity_y', 'BeltAngularVelocity_z', 
                               'NeckAccelerometer_x', 'NeckAccelerometer_y', 'NeckAccelerometer_z', 
                               'NeckAngularVelocity_x', 'NeckAngularVelocity_y', 'NeckAngularVelocity_z', 
                               'WristAccelerometer_x', 'WristAccelerometer_y', 'WristAccelerometer_z', 
                               'WristAngularVelocity_x', 'WristAngularVelocity_y', 'WristAngularVelocity_z', 
                               'Subject', 
                               'Activity', 
                               'Trial'])
    
    # Drop Subjects 5 and 9 (no camera data)
    df = df.drop(df[df.Subject == 5].index)
    df = df.drop(df[df.Subject == 9].index)
    
    # Group by Subject, Activity, Trial
    S_A_T = df.groupby(['Subject', 'Activity', 'Trial'])
    
    # Create dictionary to save GAF of each sensor
    GAF_data = {}
    
    for key, value in S_A_T:
                
        value = value.iloc[0:140, :]  # Limit to 140 time steps
        gaf = GramianAngularField(method='difference')
        
        # Ankle
        AnkleAccelerometer = np.sqrt(value['AnkleAccelerometer_x']**2 + 
                                    value['AnkleAccelerometer_y']**2 + 
                                    value['AnkleAccelerometer_z']**2)
        gaf_AnkleAccelerometer = gaf.fit_transform(np.array(AnkleAccelerometer).reshape(1, -1))
        AnkleAngularVelocity = np.sqrt(value['AnkleAngularVelocity_x']**2 + 
                                      value['AnkleAngularVelocity_y']**2 + 
                                      value['AnkleAngularVelocity_z']**2)
        gaf_AnkleAngularVelocity = gaf.fit_transform(np.array(AnkleAngularVelocity).reshape(1, -1))
        gaf_Ankle = np.dstack((gaf_AnkleAccelerometer[0], gaf_AnkleAngularVelocity[0]))
        
        # RightPocket
        RightPocketAccelerometer = np.sqrt(value['RightPocketAccelerometer_x']**2 + 
                                          value['RightPocketAccelerometer_y']**2 + 
                                          value['RightPocketAccelerometer_z']**2)
        gaf_RightPocketAccelerometer = gaf.fit_transform(np.array(RightPocketAccelerometer).reshape(1, -1))
        RightPocketAngularVelocity = np.sqrt(value['RightPocketAngularVelocity_x']**2 + 
                                            value['RightPocketAngularVelocity_y']**2 + 
                                            value['RightPocketAngularVelocity_z']**2)
        gaf_RightPocketAngularVelocity = gaf.fit_transform(np.array(RightPocketAngularVelocity).reshape(1, -1))
        gaf_RightPocket = np.dstack((gaf_RightPocketAccelerometer[0], gaf_RightPocketAngularVelocity[0]))
        
        # Belt
        BeltAccelerometer = np.sqrt(value['BeltAccelerometer_x']**2 + 
                                   value['BeltAccelerometer_y']**2 + 
                                   value['BeltAccelerometer_z']**2)
        gaf_BeltAccelerometer = gaf.fit_transform(np.array(BeltAccelerometer).reshape(1, -1))
        BeltAngularVelocity = np.sqrt(value['BeltAngularVelocity_x']**2 + 
                                     value['BeltAngularVelocity_y']**2 + 
                                     value['BeltAngularVelocity_z']**2)
        gaf_BeltAngularVelocity = gaf.fit_transform(np.array(BeltAngularVelocity).reshape(1, -1))
        gaf_Belt = np.dstack((gaf_BeltAccelerometer[0], gaf_BeltAngularVelocity[0]))
        
        # Neck
        NeckAccelerometer = np.sqrt(value['NeckAccelerometer_x']**2 + 
                                   value['NeckAccelerometer_y']**2 + 
                                   value['NeckAccelerometer_z']**2)
        gaf_NeckAccelerometer = gaf.fit_transform(np.array(NeckAccelerometer).reshape(1, -1))
        NeckAngularVelocity = np.sqrt(value['NeckAngularVelocity_x']**2 + 
                                     value['NeckAngularVelocity_y']**2 + 
                                     value['NeckAngularVelocity_z']**2)
        gaf_NeckAngularVelocity = gaf.fit_transform(np.array(NeckAngularVelocity).reshape(1, -1))
        gaf_Neck = np.dstack((gaf_NeckAccelerometer[0], gaf_NeckAngularVelocity[0]))
        
        # Wrist
        WristAccelerometer = np.sqrt(value['WristAccelerometer_x']**2 + 
                                    value['WristAccelerometer_y']**2 + 
                                    value['WristAccelerometer_z']**2)
        gaf_WristAccelerometer = gaf.fit_transform(np.array(WristAccelerometer).reshape(1, -1))
        WristAngularVelocity = np.sqrt(value['WristAngularVelocity_x']**2 + 
                                      value['WristAngularVelocity_y']**2 + 
                                      value['WristAngularVelocity_z']**2)
        gaf_WristAngularVelocity = gaf.fit_transform(np.array(WristAngularVelocity).reshape(1, -1))
        gaf_Wrist = np.dstack((gaf_WristAccelerometer[0], gaf_WristAngularVelocity[0]))
        
        each_gaf = {
            'GAF_Ankle': gaf_Ankle,
            'GAF_RightPocket': gaf_RightPocket,
            'GAF_Belt': gaf_Belt,
            'GAF_Neck': gaf_Neck,
            'GAF_Wrist': gaf_Wrist
        }
        GAF_data[key] = each_gaf

        # Save GAF data to a pickle file
        print(f"Processed GAF for Subject {int(key[0])}, Activity {int(key[1])}, Trial {int(key[2])}")
    with open(os.path.join(sensor_path, 'GAF_sensor_data.pkl'), 'wb') as f:
        pickle.dump(GAF_data, f)
    
    return GAF_data