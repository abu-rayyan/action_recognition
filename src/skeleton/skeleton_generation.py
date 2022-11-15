#%%
import os
import sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
from utils.frames_extraction import extract
import cv2
import numpy as np
from pose_estimation import pose_model_normal
import json
import re
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
from utils import paths
from utils.video_generation import video
import time
import gc
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import re
#%%
def generate_json(data,json_path):
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as fp:
        print('Writing JSON file  ',json_path)
        json.dump(data, fp,indent=2)
    print('JSON file generated !')
#%%
extract_path=os.path.join(os.path.abspath(os.path.join(__file__,"../")),'extracted_frames')
# Define the body keypoints for pose estimation 
key_points=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
pose=pose_model_normal()
#%%
def generate_image(image,centerOfCircle):
    radius=3
    centers = {}
    CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    CocoPairs = [(0, 1),(0, 2),(1,2),(1,3),(2, 4),(6, 8),
                  (8, 10),(5, 7),(5, 6),(5, 11), (7, 9),
      (12, 14), (14, 16), (11, 13),(11, 12),(13, 15),(6, 12)]
    for i in range(len(centerOfCircle)):
      centers[i]=centerOfCircle[i]
      image=cv2.circle(image, tuple(centerOfCircle[i]), radius, CocoColors[i], thickness=3, lineType=8, shift=0)
      
    for pair_order, pair in enumerate(CocoPairs):
      if pair_order<=3:
        if math.hypot((centers[pair[1]]-centers[pair[0]])[0],(centers[pair[1]]-centers[pair[0]])[1])>30:
          continue
      image=cv2.line(image, tuple(centers[pair[0]]),tuple(centers[pair[1]]), CocoColors[pair_order], 3)
    return image
#%%
def pose_image(full_path):
    
    img_path=os.path.split(full_path)[1]
    print(img_path)
    BB,pred_coords=pose.calculate_coordinates(full_path)
    # key_maps=dict(zip(key_points,(pred_coords.astype(int)).tolist()))           # Create a dictionary of keypoints
    return BB,pred_coords
#%%
def evaluate_video(file_path,wrist,fps,stick_height,skeleton_path):
    test_file_name=os.path.split(file_path)[-1]
    extract(file_path,extract_path)  
    json_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'results',test_file_name.split('.')[0])
    if os.path.exists(skeleton_path):
        shutil.rmtree(skeleton_path)
    os.makedirs(skeleton_path) 
    wrist=wrist.split('_')[0]
    
    pose_coordinates={}
    images_list = list(paths.list_images(extract_path))
    images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    # images_list=images_list[:6]

    for i,image in enumerate(images_list):
        img_path=os.path.split(images_list[i])[-1]
        BB,pred_coords=pose_image(image)
        # print(pred_coords)
        img=cv2.imread(images_list[i])
        key_maps=dict(zip(key_points,(pred_coords.astype(int)).tolist()))
        pose_coordinates[img_path]=key_maps   
        img=generate_image(img,pred_coords)
        cv2.imwrite(os.path.join(skeleton_path,img_path), img)
        del img
        gc.collect()
    generate_json(pose_coordinates,json_files_path+'_coordinates.json')
#%%
if __name__=='__main__':
    start_time=time.time()
    ext='mp4'
    video_name='IMG_0882_60.mp4'
    #video_name='IMG_0876_60.mp4'
    video_id='1234567'
    path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','testing_videos')
    file_path=os.path.join(path,video_name)
    fps=30
    game_name='softball'
    wrist='left_handed'
    distance_from_camera=10     # (feet)
    stick_height=47             # (inches)
    stick_width=2               # (inches)
    skeleton_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results','Skeleton_{}'.format(video_name.split('.')[0]))
    evaluate_video(file_path,wrist,fps,stick_height,skeleton_path)
    skeleton_video_name='Skeleton_{}.{}'.format(video_name.split('.')[0],ext)
    video_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
    video(skeleton_path,video_path,skeleton_video_name,fps)
    end_time=time.time()
    print('Code execution complete..\n')
    print('Total execution time is : %d minutes and %d seconds'%(((end_time-start_time)-(end_time-start_time)%60)//60,(end_time-start_time)-((end_time-start_time)-(end_time-start_time)%60)))
# %%
