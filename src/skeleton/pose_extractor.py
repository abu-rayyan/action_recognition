#%%
import os
import sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
import cv2
import numpy as np
from pose_estimation import pose_model
import json
import re
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
from utils import paths

import time
import gc
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import re
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from position_calculations import calculate_max_area, scale_coordinates
import pickle
from PIL import Image

#%%
""" This code just extracts the pose coordinates from the given cropped dataset, which must be divided into classes, 
    and saves the results in a JSON file.
    It also saves the skeleton images within the classes"""
#%%
def generate_json(data,json_path):
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as fp:
        print('Writing JSON file  ',json_path)
        json.dump(data, fp,indent=2)
    print('JSON file generated !')
#%%
# extract_path=os.path.join(os.path.abspath(os.path.join(__file__,"../")),'extracted_frames')
# Define the body keypoints for pose estimation 
key_points=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
CocoPairs = [(0, 1),(0, 2),(1,2),(1,3),(2, 4),(6, 8),
                (8, 10),(5, 7),(5, 6),(5, 11), (7, 9),
    (12, 14), (14, 16), (11, 13),(11, 12),(13, 15),(6, 12)]
pose=pose_model()

#%%
def calculate_coordinates(full_path):
        x, image = data.transforms.presets.rcnn.load_test(full_path,short=1024)
        class_IDs, scores, bounding_boxs = pose.detector(x)
        person=calculate_max_area(bounding_boxs,scores)
        if person is None:
            print('Person not detected')
            return None
        pose_input, upscale_bbox = detector_to_alpha_pose(image, class_IDs, scores, bounding_boxs)
        del x,class_IDs, scores
        gc.collect()
        predicted_heatmap = pose.pose_net(pose_input)
        pred_coords, _ = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
        pred_coords=pred_coords.asnumpy()
        pred_coords=pred_coords[person]
        bounding_boxs=bounding_boxs.asnumpy()
        BB=bounding_boxs[0][person]
        BB,pred_coords=scale_coordinates(full_path,pred_coords,image.shape[0],image.shape[1],BB)
        del predicted_heatmap,upscale_bbox,bounding_boxs,pose_input,image,BB
        gc.collect()
        return pred_coords

#%%
def pose_image(full_path):
    pred_coords=calculate_coordinates(full_path)
    return pred_coords
#%%
def generate_skeleton(image_path,centerOfCircle,skeleton_path_class,img_name):
    im = Image.open(image_path)
    height,width = im.size[:2]
    image=np.zeros([width,height])
    centers = {}
    
    for i in range(len(centerOfCircle)):
        centers[i]=centerOfCircle[i]
        if i>0 and i<5:
            continue
        if i==0:
            radius=6
        else:
            radius=3
        image=cv2.circle(image, tuple(centerOfCircle[i]), radius, (255,255,255), thickness=4, lineType=8, shift=0)
      
    for pair_order, pair in enumerate(CocoPairs):
      if pair_order>0 and pair_order<5:
          continue
      image=cv2.line(image, tuple(centers[pair[0]]),tuple(centers[pair[1]]), (255,255,255), 6)
    cv2.imwrite(os.path.join(skeleton_path_class,img_name), image)
    del image
    gc.collect()
#%%
def generate_image(image_path,centerOfCircle,skeleton_path_class,img_name):
    image=cv2.imread(image_path)
    radius=3
    centers = {}
    
    for i in range(len(centerOfCircle)):
      centers[i]=centerOfCircle[i]
      image=cv2.circle(image, tuple(centerOfCircle[i]), radius, CocoColors[i], thickness=3, lineType=8, shift=0)
      
    for pair_order, pair in enumerate(CocoPairs):
      if pair_order<=3:
        if math.hypot((centers[pair[1]]-centers[pair[0]])[0],(centers[pair[1]]-centers[pair[0]])[1])>30:
          continue
      image=cv2.line(image, tuple(centers[pair[0]]),tuple(centers[pair[1]]), CocoColors[pair_order], 3)
    cv2.imwrite(os.path.join(skeleton_path_class,img_name), image)
    del image
    gc.collect()
#%%
def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
#%%
if __name__=='__main__':
    data_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data')
    file_path=os.path.join(data_path, 'cropped_data')
    # file_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','cropped_data')
    print('Total classes detected : {}'.format(os.listdir(file_path)))
    pose_coordinates={}
    coordinates=[]
    labels=[]
    results_folder=os.path.join(data_path, 'results')
    # json_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','results','pose_filtered')
    json_files_path=os.path.join(results_folder,'pose_filtered')
    skeleton_path=os.path.join(data_path,'skeleton_classes')
    pure_skeleton_path=os.path.join(data_path, 'pure_skeleton_classes')
    
    create_directory(results_folder)
    create_directory(skeleton_path)
    create_directory(pure_skeleton_path)

    for classes in os.listdir(file_path):
        print('For class: {}'.format(classes))
        class_path=os.path.join(file_path,classes)
        
        skeleton_path_class=os.path.join(skeleton_path,classes)
        pure_skeleton_path_class=os.path.join(pure_skeleton_path,classes)
        
        create_directory(pure_skeleton_path_class)
        create_directory(skeleton_path_class)
        
 
        images_list = list(paths.list_images(class_path))
        images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        for img_path in images_list:
            img_name=os.path.split(img_path)[-1]
            print('Extracting pose from ',img_path)
            pred_coords=pose_image(img_path)
            if pred_coords is None:
                continue
            coordinates.append(pred_coords)
            labels.append(int(classes.split('_')[0]))
            key_maps=dict(zip(key_points,(pred_coords.astype(int)).tolist()))
            key_maps['label']=classes
            pose_coordinates[os.path.join(classes,img_name)]=key_maps
            generate_image(img_path,pred_coords,skeleton_path_class,img_name)
            generate_skeleton(img_path,pred_coords,pure_skeleton_path_class,img_name)
    generate_json(pose_coordinates,json_files_path+'_coordinates.json')
