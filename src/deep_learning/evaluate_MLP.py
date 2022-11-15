#%%
import os
import sys
sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(__file__,"../..")))
from utils.frames_extraction import extract
import cv2
import numpy as np
from position_calculations import display_angle,wrist_calculation,arm_displacement,max_velocity,ball_velocity
from pose_estimation import pose_model
from action_recognition import action_model
from image_segmentation import segmentation,detect_stick
from skeleton.pose_processor import load_dataframe,calculate_distance, concatenate
import json
import re
import pandas as pd
from utils import paths
# from evaluation import classes,key_points
import time
import gc
import concurrent.futures
import multiprocessing
from multiprocessing import Process
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
cropped_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'detected_frames')
if os.path.exists(cropped_path):
        shutil.rmtree(cropped_path)
os.makedirs(cropped_path)
model_name='MLP_Pose_Model'
model_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models',model_name)
# Define the body keypoints for pose estimation 
key_points=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
csv_file_name='all_pose_dataframe.csv'
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
df=pd.read_csv(csv_file,index_col=0)
classes=df['label'].unique()
classes.sort()
print(classes)
model=action_model(model_path)
# classes=['0_set_position','1_forward_motion','2_lead_foot_plant','3_ball_release']
#%%
def process_image(full_path,BB,stick_height):
    img_path=os.path.split(full_path)[1]
    x1,y1,x2,y2=BB
    print(img_path)
    #print(x1,y1,x2,y2)
    #print(stick_height)
    img=cv2.imread(full_path)
    image=img[y1:y2,x1:x2]
    if image.shape[0]==0 or image.shape[1]==0:  
        print('--Person not found in ',img_path)
        center,position_text,predict_label=[],[],[]
        del img,image
        gc.collect()
        return center,position_text,predict_label
    
    position_text,predict_label=model.predict_action(image,classes)
    del model       # Sajid told about this
    gc.collect()
    segment=segmentation(stick_height)
    cX,cY=segment.person_segment(image)
    del segment,image     
    gc.collect()
    center=[int(cX),int(cY)]
    # print('For {}, predicted position is : {}'.format(img_path,position_text))
    return center,position_text,predict_label
#%%
def pose_image(full_path):
    
    img_path=os.path.split(full_path)[1]
    print(img_path)
    pose=pose_model()
    BB,pred_coords,index=pose.calculate_coordinates(full_path)
    del pose
    gc.collect()
    key_maps=dict(zip(key_points,(pred_coords.astype(int)).tolist()))           # Create a dictionary of keypoints
    return BB,key_maps,index
#%%
# def load_dataframe(pose,wrist):
    # # print(pose)
    # df=pd.DataFrame.from_dict(pose,orient='index')
    # print(df)
    # if wrist=='right':
    #     columns_to_drop=['left_eye','right_eye','left_ear','right_ear','left_shoulder','left_elbow','left_wrist','left_hip']
    # elif wrist=='left':
    #     columns_to_drop=['left_eye','right_eye','left_ear','right_ear','right_shoulder','right_elbow','right_wrist','right_hip']
    # df=df.drop(columns=columns_to_drop)
    # col=df.columns
    # index=df.index
    # df_x=pd.DataFrame(columns=col,index=index)
    # df_y=pd.DataFrame(columns=col,index=index)
    # for i in range(len(col)):
    #     df_x[df.columns[i]]=df[df.columns[i]].map(lambda x: x[0])
    #     df_y[df.columns[i]]=df[df.columns[i]].map(lambda x: x[1])
    # return df_x,df_y
#%%
def evaluate_video(file_path,wrist,fps,stick_height):
    test_file_name=os.path.split(file_path)[-1]
    extract(file_path,extract_path)  
    if not os.path.exists(model_path):
        print('Action Recognition model not found. Please download the model using the script')
        sys.exit()

    json_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results',test_file_name.split('.')[0])
     
    wrist=wrist.split('_')[0]
    
    pose_coordinates={}
    frame_status=np.ones(len(classes))*-1
    frame_capture={}
    output={}
    centers_dict={}
    ball_center={}
    ball=[]
    bounding_boxes=[]
    frame_dict={}
    images_list = list(paths.list_images(extract_path))
    images_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    #images_list=images_list[:4]
    # for i,image in enumerate(images_list):
    #     BB,pred_coords,index=pose_image(image)
    #     bounding_boxes.append(BB)
    total_workers=multiprocessing.cpu_count()
    wf=0.5
    print('Total processors : ', int(total_workers*wf))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(total_workers*wf)) as executor:
        print('Starting ProcessPoolExecutor for pose estimation')
        results=executor.map(pose_image,images_list)
    print('ProcessPoolExecutor completed successfully for pose estimation')
    check=-1
    for i,result in enumerate(results):
        img_path=os.path.split(images_list[i])[-1]
        print(img_path)
        # bounding_boxes[img_path]=list(result[0])
        BB=result[0]
        key_maps=result[1]
        index=result[2]
        bounding_boxes.append(list(BB[0]))
        if index[1]>-1:  # Ball exists
                x1b,y1b,x2b,y2b=BB[1]
                print(BB[1])
                x=x1b+((x2b-x1b)//2)
                y=y1b+((y2b-y1b)//2)
                ball_center[img_path]=x,y
                ball.append([x,y])
                img=cv2.imread(images_list[i])
                cv2.circle(img,(x,y),1,(0,0,255), 2)
                cv2.imwrite(os.path.join(cropped_path,img_path), img)
        else:
                ball.append(-1)
        if check==-1:
            img=cv2.imread(images_list[i])
            inches_in_pixel=detect_stick(img,stick_height)   
            if inches_in_pixel is not None:
                    check=1
        pose_coordinates[img_path]=key_maps
        key_maps['label']=np.nan
        poses={}
        poses[img_path]=key_maps   
        df_x,df_y,_=load_dataframe(poses,wrist)
        df_distance,df_angle=calculate_distance(df_x,df_y)
        df_pose=concatenate(df_x,df_y,df_distance,df_angle,None)
        # print(df_pose)
        position_text,predict_label=model.predict_action(df_pose.values,classes)
        print(position_text)
        frame_dict[img_path]=position_text
    generate_json(pose_coordinates,json_files_path+'_coordinates.json')
    generate_json(frame_dict,json_files_path+'_actions.json')

    sticks=[stick_height]*len(images_list)
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
    start=time.time()
    # video_name='shortened_IMG_0548.mp4'
    video_name='IMG_6891.mov'
    video_id='1234567'
    path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','testing_videos')
    file_path=os.path.join(path,video_name)
    fps=15
    game_name='softball'
    wrist='right_handed'
    distance_from_camera=10     # (feet)
    stick_height=47             # (inches)
    stick_width=2               # (inches)
    evaluate_video(file_path,wrist,fps,stick_height)

    end=time.time()
    print('Time taken MultiProcessing:', end-start)
# %%
