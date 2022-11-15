#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__,"../")))
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
import cv2
import json
import re
import time
import gc
import pandas as pd
import numpy as np
import shutil
import pickle
from scipy import stats
import concurrent.futures
import multiprocessing
from multiprocessing import Process
from position_calculations import display_angle,wrist_calculation,arm_displacement,max_velocity,ball_velocity,find_displacement
from pose_estimation import pose_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from skeleton.pose_processor import load_dataframe,calculate_distance, concatenate,calculate_max_min
from utils import paths
from utils.frames_extraction import extract
from utils.generate_labeled_video import label_frames

#%%


def load_json(json_file):
    with open(json_file) as f:
        pose = json.load(f)
    return pose

#%%
def generate_json(data,json_path):
    if os.path.exists(json_path):
        os.remove(json_path)
    with open(json_path, 'w') as fp:
        print('Writing JSON file  ',json_path)
        json.dump(data, fp,indent=2)
    print('JSON file generated !')
#%%
extract_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'extracted_frames')
labeled_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'detected_frames')
if os.path.exists(labeled_path):
        shutil.rmtree(labeled_path)
os.makedirs(labeled_path)
model_name='random_forest_pose_model_26_Apr.sav'
#model_name='random_forest_pose_model.sav'
model_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models',model_name)
# Define the body keypoints for pose estimation 
key_points=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

csv_file_name='pose_filtered_dataframe.csv'
json_file_name='pose_filtered_coordinates.json'
results_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(results_path,csv_file_name)
json_file=os.path.join(results_path,json_file_name)
pose_train=load_json(json_file)
if not os.path.exists(csv_file):
    print('csv file not found.')
    sys.exit()
df=pd.read_csv(csv_file,index_col=0)
classes=df['label'].unique()
classes.sort()
print(classes)
model = pickle.load(open(model_path, 'rb'))
# classes=['0_set_position','1_forward_motion','2_lead_foot_plant','3_ball_release']
thresh=16

#%%
def process_dataframe(frame_dict,df2,df_hand):
    df=pd.DataFrame.from_dict(frame_dict,orient='index',columns=['label'])
    check=np.ones(len(classes))*-1
    df["category"] = df["label"].astype('category')
    df["category"] = df["category"].cat.codes

    data=df.values
    ws=2
    for i in range(ws,len(data)-ws):
        data_list=[]
        string_list=[]
        data_list=[int(data[i+j,1]) for j in range(-ws,ws+1)]
        string_list=[str(data[i+j,0]) for j in range(-ws,ws+1)]
        data[i,1]=int(stats.mode(data_list)[0])
        data[i,0]=stats.mode(string_list)[0].tolist()[0]

    for i in range(len(data)):
        lab=data[i,1]
        if check[int(lab)]==-1:
            check[int(lab)]=i
    idx=df_hand['y'].argmin()
    # idx=df2.argmax()
    check[2]=np.where(df2[idx:]<thresh)[0][0]+idx
    
    for i in range(len(check)):
        if i==len(check)-1:
            data[int(check[i]):]=data[int(check[i])]
        else:
            data[int(check[i]):int(check[i+1])]=data[int(check[i])]
    df_new=pd.DataFrame(data,index=df.index,columns=['labels','category'])
    df_new=df_new.drop(columns=['category'])
    #df=df.drop(columns=['category'])
    #return df
    return df_new
#%%
def pose_image(full_path):
    
    img_path=os.path.split(full_path)[1]
    print(img_path)
    pose=pose_model()
    BB,pred_coords,index,ball_score=pose.calculate_coordinates(full_path)
    
    
    pred_coords=np.array(pred_coords)
    pred_coords[:,0]=pred_coords[:,0]-BB[0][0]
    pred_coords[:,1]=pred_coords[:,1]-BB[0][1]
    
    
    #print(BB)
    #print(pred_coords)
    del pose
    gc.collect()
    key_maps=dict(zip(key_points,(pred_coords.astype(int)).tolist()))           # Create a dictionary of keypoints
    return BB,key_maps,index

#%%
def evaluate_video(file_path,wrist,fps):
    test_file_name=os.path.split(file_path)[-1]
    extract(file_path,extract_path)  
    if not os.path.exists(model_path):
        print('Action Recognition model not found. Please download the model using the script')
        sys.exit()

    json_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results',test_file_name.split('.')[0])
     
    wrist=wrist.split('_')[0]
    
    
    train_dfx,train_dfy,_=load_dataframe(pose_train,wrist)
    train_distance,_=calculate_distance(train_dfx,train_dfy)
    max_dfx,min_dfx=calculate_max_min(train_dfx)
    max_dfy,min_dfy=calculate_max_min(train_dfy)
    max_distance,min_distance=calculate_max_min(train_distance)
    
    pose_coordinates={}
    frame_status=np.ones(len(classes))*-1
    frame_capture={}
    output={}
    centers_dict={}
    ball_center={}
    ball=[]
    bounding_boxes=[]
    frame_dict={}
    if wrist=='right':
    		ankle='left_ankle'
    else:
    		ankle='right_ankle'
    images_list = list(paths.list_images(extract_path))
    images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    #images_list=images_list[:4]
    # for i,image in enumerate(images_list):
    #     BB,key_maps,index=pose_image(image)
    #     bounding_boxes.append(BB)
    total_workers=multiprocessing.cpu_count()
    wf=0.75
    print('Total processors : ', int(total_workers*wf))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(total_workers*wf)) as executor:
        print('Starting ProcessPoolExecutor for pose estimation')
        results=executor.map(pose_image,images_list)
    print('ProcessPoolExecutor completed successfully for pose estimation')
    for i,result in enumerate(results):
        img_path=os.path.split(images_list[i])[-1]
        print(img_path)
        # bounding_boxes[img_path]=list(result[0])
        BB=result[0]
        key_maps=result[1]
        index=result[2]
        bounding_boxes.append(list(BB[0]))
        # if index[1]>-1:  # Ball exists
        #         x1b,y1b,x2b,y2b=BB[1]
        #         print(BB[1])
        #         x=x1b+((x2b-x1b)//2)
        #         y=y1b+((y2b-y1b)//2)
        #         ball_center[img_path]=x,y
        #         ball.append([x,y])
        #         img=cv2.imread(images_list[i])
        #         cv2.circle(img,(x,y),1,(0,0,255), 2)
        #         cv2.imwrite(os.path.join(labeled_path,img_path), img)
        # else:
        #         ball.append(-1)

        pose_coordinates[img_path]=key_maps
        key_maps['label']=np.nan
        poses={}
        poses[img_path]=key_maps   
        df_x,df_y,_=load_dataframe(poses,wrist)
        df_distance,df_angle=calculate_distance(df_x,df_y)
        
        df_pose=concatenate(df_x,df_y,df_distance,df_angle,max_dfx,min_dfx,max_dfy,min_dfy,max_distance,min_distance)
        
        
        # print(df_pose)
        predict_label=model.predict(df_pose.values)
        position_text=list(classes[predict_label])
        frame_dict[img_path]=position_text
    #generate_json(pose_coordinates,json_files_path+'_coordinates.json')
    #generate_json(frame_dict,json_files_path+'_actions_random_forest.json')
    res = [val[ankle] for key, val in pose_coordinates.items() if ankle in val]
    poses=dict(zip(pose_coordinates.keys(),res))
    df_ankle=find_displacement(poses,ankle)

    res = [val[wrist+'_wrist'] for key, val in pose_coordinates.items() if wrist+'_wrist' in val]
    poses=dict(zip(pose_coordinates.keys(),res))
    df_hand=pd.DataFrame.from_dict(poses,orient='index',columns=['x','y'])
    df=process_dataframe(frame_dict,df_ankle,df_hand)
    
    df_final=pd.concat([df,df_ankle], axis=1)
    df_final.to_csv(json_files_path+'_actions_random_forest.csv')
    label_frames(test_file_name,extract_path,labeled_path,results_path,df,fps=20)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=int(total_workers*wf)) as executor:
    #     print('Starting ProcessPoolExecutor for action recognition')
    #     results=executor.map(process_image,images_list,bounding_boxes,sticks)
    # print('ProcessPoolExecutor completed successfully for action recognition')
    # for i,result in enumerate(results):
    #     img_path=os.path.split(images_list[i])[-1]
    #     print(img_path)
    #     center,position_text,predict_label=result[0],result[1],result[2]
    #     if predict_label==[]:
    #         print('--Person not found in ',img_path)
    #         continue
    #     elif (center[0]==[] or center[1]==[]):
    #         print('Segmentation error: Center not found in ',img_path)
    #         continue
    #     else:                    
    #         centers_dict[img_path]=np.add(bounding_boxes[i][0:2],center).astype(int).tolist()
    #         
    #     if frame_status[int(predict_label)]==-1:
    #             frame_status[int(predict_label)]=int(img_path.split('.')[0].split('_')[1])
    #             frame_capture[img_path]=position_text
    #             print(frame_status)
    #             print(frame_capture)
    #             lean_angle,lean=display_angle(pose_coordinates[img_path],wrist)

    #             if lean=='forward':
    #                 output['forward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=lean_angle
    #                 output['backward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=np.nan
    #             else:
    #                 output['backward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=lean_angle
    #                 output['forward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=np.nan

    # generate_json(centers_dict,json_files_path+'_center_of_mass.json')
    
    # _,_,w=wrist_calculation(pose_coordinates,wrist=wrist)
    # frame_capture,_=arm_displacement(w,frame_capture)
    # frame_max_velocity, max_v,max_d=max_velocity(centers_dict,inches_in_pixel,fps)
    # if frame_max_velocity in frame_capture.keys():
    #     keys=frame_capture[frame_max_velocity]
    #     frame_capture[frame_max_velocity]=['Frame of maximum velocity',str(max_v)+' in/sec',str(max_d)+' inches',keys]
    # else:
    #     frame_capture[frame_max_velocity]=['Frame of maximum velocity',str(max_v)+' in/sec',str(max_d)+' inches']
    # generate_json(frame_capture,json_files_path+'.json')
    # lean_angle,lean=display_angle(pose_coordinates[frame_max_velocity],wrist)
    # if lean=='forward':
    #     output["forward_lean_angle_max_velocity"]=lean_angle
    #     output["backward_lean_angle_max_velocity"]=np.nan
    # else:
    #     output["backward_lean_angle_max_velocity"]=lean_angle
    #     output["forward_lean_angle_max_velocity"]=np.nan
    # _,text=arm_displacement(w,{frame_max_velocity:frame_capture[frame_max_velocity]})
    # output["arm_position_at_max_velocity"]=text
    # output["top_velocity"]=str(max_v)+' inches per second'
    # frame_classes=dict(zip(classes,frame_status))

    # if frame_classes[classes[4]]!=-1 and frame_classes[classes[1]]!=-1:
    #     total_pitch_frames=abs(frame_classes[classes[4]]-frame_classes[classes[1]])
    #     output["pitch_duration"]=total_pitch_frames/fps
    # else:
    #     output["pitch_duration"]='Action not detected'
    # if frame_classes[classes[3]]!=-1 and frame_classes[classes[2]]!=-1:
    #     stride_velocity_frames=abs(frame_classes[classes[3]]-frame_classes[classes[2]])
    #     stride_time=stride_velocity_frames/fps
    #     diff_x=(centers_dict['frame_{}.jpg'.format(int(frame_classes[classes[3]]))][0]-centers_dict['frame_{}.jpg'.format(int(frame_classes[classes[2]]))][0])*inches_in_pixel
    #     output["Stride velocity"]=abs(diff_x)/stride_time
    # else:
    #     output["Stride velocity"]='Action not detected'
    # output["shoulder_hip_joint_frames_file"]=json_files_path
    
    # print(ball_center)
    # print(ball)
    # # Ball Velocity
    # if frame_classes[classes[4]]!=-1:
    #     release_frame=int(frame_classes[classes[4]])
    #     print(release_frame)
    #     ball_coordinates=[[i,x] for i,x in enumerate(ball[release_frame:]) if x!=-1]
    #     print(ball_coordinates)
    #     velocity_x,release_angle=ball_velocity(ball_coordinates,inches_in_pixel,fps)
    #     print(velocity_x)
    #     print(release_angle)
    #     output['Ball Velocity']=velocity_x
    #     output['Ball Angle']=release_angle
    # generate_json(output,json_files_path+'_output.json')
        

 

#%%
if __name__=='__main__':
    start_time=time.time()
    #video_name='IMG_0875_60.mp4'
    video_name='IMG_6891.mov'
    video_id='1234567'
    path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','testing_videos')
    file_path=os.path.join(path,video_name)
    fps=60
    game_name='softball'
    wrist='right_handed'
    distance_from_camera=10     # (feet)
    evaluate_video(file_path,wrist,fps)

    end_time=time.time()
    print('Total execution time is : %d minutes and %d seconds'%(((end_time-start_time)-(end_time-start_time)%60)//60,(end_time-start_time)-((end_time-start_time)-	 (end_time-start_time)%60)))



#%% Frames Extraction


