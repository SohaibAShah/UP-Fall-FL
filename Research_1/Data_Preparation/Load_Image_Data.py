import pandas as pd
import os
import re 
import numpy as np
import cv2
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil

def load_img(start_sub, end_sub ,start_act, end_act,  start_cam ,  end_cam , DesiredWidth = 64, DesiredHeight = 64 ):
    IMG = []
    count = 0
    name_img = []
    for sub_ in range(start_sub, end_sub+1):
        sub = 'Subject' + str(sub_)

        for act_ in range(start_act, end_act + 1) :
            act = 'Activity' + str(act_)

            for trial_ in range(1, 3 + 1):
                trial = 'Trial'+ str(trial_)
                if ( (sub_ == 8 and act_ == 11) and ( trial_ == 2 or trial_ == 3) ) :
                    print('----------------------------NULL---------------------------------')
                    continue 
                for cam_ in range(start_cam , end_cam + 1) :
                    cam = 'Camera'+ str(cam_)

                    with ZipFile('UP-Fall Dataset/downloaded_camera_files//' + sub + act  + trial + cam + '.zip', 'r') as zipObj:
                        zipObj.extractall('CAMERA/' + sub + act  + trial  + cam)

                    for root, dirnames, filenames in os.walk('CAMERA/' + sub + act + trial  +cam ):
                        for filename in filenames:
                            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                                filepath = os.path.join(root, filename)
                                count += 1 
                                if count % 5000 == 0 :
                                    print('{} : {} ' .format(filepath,count))
                                if filepath == 'CAMERA/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png' :
                                    print('----------------------------NO SHAPE---------------------------------')
                                    continue
                                elif len(filepath) > 70 :
                                    print(' {} : Invalid image'.format(filepath))
                                    continue 
                                name_img.append(filepath)
                                img = cv2.imread(filepath, 0) 
                                resized = ResizeImage(img, DesiredWidth, DesiredHeight)
                                IMG.append(resized)
                    shutil.rmtree('CAMERA/'+ sub + act + trial + cam)
    return IMG , name_img


def handle_name(path_name) :
    img_name = []
    for path in path_name :
        if len(path) == 68: 
            img_name.append(path[38:64])
        elif len(path) == 69 :
            img_name.append(path[39:65])
        else :
            img_name.append(path[40:66])
    handle = []
    for name in img_name :
        n1 = 13
        a1 = name.replace(name[n1],':')
        n2 = 16
        a2 = a1.replace(name[n2],':')
        handle.append(a2)
    return handle 


def ShowImage(ImageList, nRows = 1, nCols = 2, WidthSpace = 0.00, HeightSpace = 0.00):
    from matplotlib import pyplot as plt 
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nRows, nCols)     
    gs.update(wspace=WidthSpace, hspace=HeightSpace) # set the spacing between axes.
    plt.figure(figsize=(20,20))
    for i in range(len(ImageList)):
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.subplot(nRows, nCols,i+1)
        image = ImageList[i].copy()
        if (len(image.shape) < 3):
            plt.imshow(image, plt.cm.gray)
        else:
            plt.imshow(image)
        plt.title("Image " + str(i))
        plt.axis('off')
    plt.show()
    
    
def ResizeImage(IM, DesiredWidth, DesiredHeight):
    OrigWidth = float(IM.shape[1])
    OrigHeight = float(IM.shape[0])
    Width = DesiredWidth 
    Height = DesiredHeight

    if((Width == 0) & (Height == 0)):
        return IM
    
    if(Width == 0):
        Width = int((OrigWidth * Height)/OrigHeight)

    if(Height == 0):
        Height = int((OrigHeight * Width)/OrigWidth)

    dim = (Width, Height)
    resizedIM = cv2.resize(IM, dim, interpolation = cv2.INTER_NEAREST) 
    return resizedIM


if __name__ == "__main__":
    SUB = pd.read_csv('UP-Fall Dataset/Imp_sensor.csv')
    times = SUB.iloc[:,0].values
    labels = SUB.iloc[:,-1].values
    Time_Label = pd.DataFrame(labels , index = times)
    print(Time_Label)

    start_sub = 1
    end_sub  = 17
    start_act = 1
    end_act = 11
    DesiredWidth = 32
    DesiredHeight = 32


    img_1, path_1 = load_img(start_sub ,   end_sub,
                start_act , end_act  ,
                1 ,   1 , DesiredWidth ,  DesiredHeight )


    name_1 = handle_name(path_1)

    img_2, path_2 = load_img(start_sub ,   end_sub,
                start_act , end_act  ,
                2 ,   2 , DesiredWidth ,  DesiredHeight )


    name_2 = handle_name(path_2)

    cam = '1'
    image = 'UP-Fall Dataset' + 'Prepared_Data/' + 'image_' + cam +  '.npy'     
    # name = 'Camera + Label' + '/' + size + '/' + 'name_' + cam + '(' + size + ')' + '.npy'     
    name = 'UP-Fall Dataset' + 'Prepared_Data/' + 'name_' + cam +  '.npy'  

    name_1 = handle_name(path_1)

    np.save(image, img_1)
    np.save(name, name_1)

    cam = '2'
    image = 'UP-Fall Dataset' + 'Prepared_Data/' + 'image_' + cam +  '.npy'     
    # name = 'Camera + Label' + '/' + size + '/' + 'name_' + cam + '(' + size + ')' + '.npy'     
    name = 'UP-Fall Dataset' + 'Prepared_Data/' + 'name_' + cam +  '.npy'  

    name_2 = handle_name(path_2)

    np.save(image, img_2)
    np.save(name, name_2)

ind1 = np.arange(0,294678)
red_in1 = ind1[~np.isin(name_1,name_2)]

name_1d =  np.delete(name_1, red_in1[0])
img_1d = np.delete(img_1, red_in1[0], axis = 0)

ind2 = np.arange(0,294678)
red_in2 = ind2[~np.isin(name_2,name_1)]

name_2d =  np.delete(name_2, red_in2[0])
img_2d = np.delete(img_2, red_in2[0], axis = 0)