#%%
import json
import pandas as pd
import numpy as np
import pickle
import math
import sys
import cv2
import re
import os
import gc
#%%
""" This code requires that the pose has already been generated and saved as JSON file in 'json_files_path' """
#%%
def load_json(json_file):
    with open(json_file) as f:
        pose = json.load(f)
    return pose
#%%
def load_dataframe(pose,wrist):
    df=pd.DataFrame.from_dict(pose,orient='index')
    # df=df.reset_index()
    # if wrist=='right':
    #     columns_to_drop=['left_eye','right_eye','left_ear','right_ear','left_shoulder','left_elbow','left_wrist','left_hip']
    # elif wrist=='left':
    #     columns_to_drop=['left_eye','right_eye','left_ear','right_ear','right_shoulder','right_elbow','right_wrist','right_hip']
    columns_to_drop=['left_eye','right_eye','left_ear','right_ear']
    df=df.drop(columns=columns_to_drop)
    label=df['label']
    df=df.drop(columns='label')
    # df=df.head()
    col=df.columns
    index=df.index
    df_x=pd.DataFrame()
    df_y=pd.DataFrame()
    for i in range(len(col)):
        df_x[col[i]+'-x']=df[col[i]].map(lambda x: x[0])
        df_y[col[i]+'-y']=df[col[i]].map(lambda x: x[1])
    return df_x,df_y,label
#%%
def calculate_max_min(df):
    max_val=df.to_numpy().max()
    min_val=df.to_numpy().min()
    return max_val, min_val
#%%
def min_max_scaling(df,max_val,min_val):
    df=(df-min_val)/(max_val-min_val)
    return df
#%%
def calculate_distance(df_x,df_y):
    mat_x=df_x.values.T
    mat_y=df_y.values.T
    df=pd.DataFrame(index=df_x.index)
    df_angle=pd.DataFrame(index=df_x.index)
    for i in range(len(mat_x)):
        for j in range(len(mat_x)):
            if i >=j:
                continue
            else:
                # print(i,j)
                d_x= mat_x[i]-mat_x[j]
                d_y= mat_y[i]-mat_y[j]
                lean_angles=angle(d_x,d_y)
                df[df_x.columns[i].split('-')[0]+'-'+df_x.columns[j].split('-')[0]]=np.sqrt(np.square(d_x) + np.square(d_y))
                df_angle[df_x.columns[i].split('-')[0]+'_to_'+df_x.columns[j].split('-')[0]+'_angle']=lean_angles
                
    return df,df_angle
#%%  
def angle(d_x,d_y):
    vector1=np.vstack((d_x, d_y)).T
    vector2=np.vstack((d_x,d_y-d_y)).T
    inner_product = np.sum(vector1*vector2,axis=1)
    len1=np.sqrt(np.square(d_x) + np.square(d_y))
    d_x=np.where(d_x==0,10,d_x)
    len2=np.sqrt(np.square(d_x) + np.square(d_y-d_y))
    a=np.arccos(inner_product/(len1*len2))
    ind=len1==0     # if the two points lie exactly on each other, i.e. len1=0
    a[ind]=-1       # set the angle value to -1
    return a
#%%
def extract_points(s):
    sx=[]
    sy=[]
    for i in range(len(s)):
        sx.append(s[i][0])
        sy.append(s[i][1])
    return np.array(sx),np.array(sy)
#%%
def concatenate_train(df_x,df_y,df_distance,df_angle,label):
    
    # for both training and testing
    max_val,min_val=calculate_max_min(df_x)
    df_x=min_max_scaling(df_x,max_val,min_val)

    max_val,min_val=calculate_max_min(df_y)
    df_y=min_max_scaling(df_y,max_val,min_val)

    max_val,min_val=calculate_max_min(df_distance)
    df_distance=min_max_scaling(df_distance,max_val,min_val)

    if label is None:
        df=pd.concat([df_x,df_y,df_distance,df_angle],axis=1)
    else:
        df=pd.concat([df_x,df_y,df_distance,df_angle,label],axis=1)
    return df
    
#%% for testing data only
def concatenate(df_x,df_y,df_distance,df_angle,max_dfx,min_dfx,max_dfy,min_dfy,max_distance,min_distance):
    # max_dfx,min_dfx=calculate_max_min(df_x)
    df_x=min_max_scaling(df_x,max_dfx,min_dfx)

    # max_dfy,min_dfy=calculate_max_min(df_y)
    df_y=min_max_scaling(df_y,max_dfy,min_dfy)

    # max_distance,min_distance=calculate_max_min(df_distance)
    df_distance=min_max_scaling(df_distance,max_distance,min_distance)
    df=pd.concat([df_x,df_y,df_distance,df_angle],axis=1)
    return df
  
    
#%%
if __name__=='__main__':

    json_file_name='pose_filtered_coordinates.json'		# the name should correspond to the actual json file in the results folder
    csv_file_name='pose_filtered_dataframe.csv'		# the csv file will be generated at the end
    json_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','results')
    json_file=os.path.join(json_files_path,json_file_name)
    pose=load_json(json_file)
    wrist='right'
    df_x,df_y,label=load_dataframe(pose,wrist)
    df_distance,df_angle=calculate_distance(df_x,df_y)
    df=concatenate_train(df_x,df_y,df_distance,df_angle,label)
    print(df)
    csv_file=os.path.join(json_files_path,csv_file_name)
    df.to_csv(csv_file)
    print('csv file written in ',csv_file)
# %%
