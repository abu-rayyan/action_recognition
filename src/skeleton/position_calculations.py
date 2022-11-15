#%%
import json
import pandas as pd
import numpy as np
import math
import sys
import cv2
import re
import os
import gc
""" Code has been updated in max_velocity and display angle"""
#%%
def load_json(json_file):
    with open(json_file) as f:
        pose = json.load(f)
    return pose
#%%
def extract_points(s):
    sx=[]
    sy=[]
    for i in range(len(s)):
        sx.append(s[i][0])
        sy.append(s[i][1])
    return np.array(sx),np.array(sy)

#%%
def calculate_max_area(bounding_box,scores):
    bounding_box=bounding_box.asnumpy()
    scores=scores.asnumpy()
    area=0
    index=None
    for i in range(np.sum(scores>0.5)):
      x1,y1,x2,y2=bounding_box[0][i]
      width=int(x2-x1)
      height=int(y2-y1)
      if width*height>area:
        area=width*height
        index=i
    return index

#%%
def scale_coordinates(full_path,pred_coords,h,w,BB):
    img=cv2.imread(full_path)
    ha=img.shape[0]/h
    wa=img.shape[1]/w
    del img
    gc.collect()
    pred_coords=(pred_coords*np.array([ha,wa])).astype(int)
    BB=(BB*np.array([ha,wa,ha,wa])).astype(int)
    return BB,pred_coords


#%%
def wrist_calculation(pose,wrist):

    # Extracting wrist coordinates
    df=pd.DataFrame.from_dict(pose,orient='index')
    fr=df.index.values.tolist()
    fr.sort(key=lambda f: int(re.sub('\D', '', f)))
    df=df.reindex(fr)
    if wrist=='right':
      rw=df['right_wrist']
      rwx,rwy=extract_points(rw)    # x and y coordinates of right-wrist
      frame_highest_wrist_y=df.index[rwy.argmin()]
      frame_lowest_wrist_y=df.index[rwy.argmax()]
      print('Frame with highest right wrist : ',frame_highest_wrist_y)
      print('Frame with lowest right wrist : ',frame_lowest_wrist_y)
      return frame_highest_wrist_y,frame_lowest_wrist_y,rw
    elif wrist=='left':
      lw=df['left_wrist']
      lwx,lwy=extract_points(lw)    # x and y coordinates of left-wrist
      frame_highest_wrist_y=df.index[lwy.argmin()]
      frame_lowest_wrist_y=df.index[lwy.argmax()]
      print('Frame with highest left wrist : ',frame_highest_wrist_y)
      print('Frame with lowest left wrist : ',frame_lowest_wrist_y)
      return frame_highest_wrist_y,frame_lowest_wrist_y,lw
    

# %%
def com_calculation(pose):
    # Extracting wrist coordinates
    df=pd.DataFrame.from_dict(pose)
    rhx,rhy=df['right_hip']   # x and y coordinates of right-hip
    lhx,lhy=df['left_hip']    # x and y coordinates of left-hip
    com_x=(rhx+lhx)//2
    com_y=((rhy+lhy)//2)-10
    return int(com_x),int(com_y)

# %%
def display_angle(pose,wrist):
    df=pd.DataFrame.from_dict(pose)
    if wrist=='right':
        rhx,rhy=df['right_hip']
        rsx,rsy=df['right_shoulder']
    else:
        rhx,rhy=df['left_hip']
        rsx,rsy=df['left_shoulder']
    vector1=(rhx-rsx,rhy-rsy)
    vector2=(rsx-rsx,rhy-rsy)
    lean_angle=math.degrees(angle(vector1, vector2))
    if rhx>rsx:
        lean='backward'
    else:
        lean='forward'
    return lean_angle,lean
#%%  
def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))
# %%
def arm_displacement(w,frame_capture):
    wx,wy=extract_points(w)
    frames={}
    w=w.reset_index()
    for f in frame_capture.keys():
        index=int(w[w.where(w['index']==f).notnull().any(axis=1)].index.values)
        value=frame_capture[f]
        mean=np.mean(np.diff(wy[index-3:index+3]))
        if mean>0 and not np.isnan(mean):
            text='Assistive'
        elif mean<0 and not np.isnan(mean):
            text='Resistive'
        else:
            text='no value'
        frames[f]=[value,text]
    return frames,text
#%%
def max_velocity(centers_dict,inches_in_pixel,fps):
    df=pd.DataFrame.from_dict(centers_dict,orient='index',columns=['x','y'])
    df['displacement']=abs(df['x']-df['x'].shift(1))
    df=df.dropna(axis=0)
    frame_max_velocity=df.displacement.idxmax()
    max_displacement=(df.displacement.max())*inches_in_pixel
    max_velocity=max_displacement*fps
    return frame_max_velocity,max_velocity,max_displacement

#%%