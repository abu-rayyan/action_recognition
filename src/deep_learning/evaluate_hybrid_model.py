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
from pose_estimation import pose_model_normal
from action_recognition import action_model_normal
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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
image_type='cropped'
model_name="Hybrid_Pose_Model_{}_class".format(image_type)
model_path=os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models',model_name)
# Define the body keypoints for pose estimation 
key_points=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
csv_file_name='all_pose_dataframe_{}_class.csv'.format(image_type)
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
df=pd.read_csv(csv_file,index_col=0)
classes=df['label'].unique()
classes.sort()
print(classes)
pose=pose_model_normal()
model=action_model_normal(model_path)
# classes=['0_set_position','1_forward_motion','2_lead_foot_plant','3_ball_release']
#%%
def process_image(full_path,BB,pose_data):
    img_path=os.path.split(full_path)[1]
    x1,y1,x2,y2=BB
    print(img_path)
    #print(x1,y1,x2,y2)
    #print(stick_height)
    img=cv2.imread(full_path)
    image=img[y1:y2,x1:x2]
    if image.shape[0]==0 or image.shape[1]==0:  
        print('--Person not found in ',img_path)
        predict=np.ones(len(classes))*-1
        del img,image
        gc.collect()
        return predict
    predict=model.predict_hybrid_action(pose_data,image)
    return predict
#%%
def pose_image(full_path):
    
    img_path=os.path.split(full_path)[1]
    print(img_path)
    
    BB,pred_coords,index=pose.calculate_coordinates(full_path)
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
    pose_data=[]

    for images in images_list:
        img_path=os.path.split(images)[-1]
        BB,key_maps,index=pose_image(images)
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
        #         cv2.imwrite(os.path.join(cropped_path,img_path), img)
        # else:
        #         ball.append(-1)
        
        pose_coordinates[img_path]=key_maps
        key_maps['label']=np.nan
        poses={}
        poses[img_path]=key_maps   
        df_x,df_y,_=load_dataframe(poses,wrist)
        df_distance,df_angle=calculate_distance(df_x,df_y)
        df_pose=concatenate(df_x,df_y,df_distance,df_angle,None)
        # print(df_pose)
        pose_data.append(df_pose.values)
        # position_text,predict_label=model.predict_action(df_pose.values,classes)
        # print(position_text)
            
    generate_json(pose_coordinates,json_files_path+'_coordinates.json')
    # print(pose_data)
    results_action=[]
    for img_path,BB,pose in zip(images_list,bounding_boxes,pose_data):
            predict=process_image(img_path,BB,pose)
            results_action.append(predict)
    results_action=np.array(results_action)
    results_action=np.squeeze(results_action,axis=1)
    results=results_action.max(axis=0)
    result_frames=results_action.argmax(axis=0)
    images_list= [images_list[i] for i in result_frames]
    for i,result in enumerate(images_list):
        img_path=os.path.split(images_list[i])[-1]
        print(img_path)
        # if predict_label==[]:
        #     print('--Person not found in ',img_path)
        #     continue
        position_text=classes[i]
        frame_dict[img_path]=position_text
        frame_status[i]=int(img_path.split('.')[0].split('_')[1])
        frame_capture[img_path]=position_text
        print(frame_status)
        print(frame_capture)
        lean_angle,lean=display_angle(pose_coordinates[img_path],wrist)

        if lean=='forward':
            output['forward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=lean_angle
            output['backward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=np.nan
        else:
            output['backward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=lean_angle
            output['forward_lean_angle_'+'_'.join(position_text.split('_')[1:])]=np.nan

    # generate_json(centers_dict,json_files_path+'_center_of_mass.json')
    generate_json(frame_dict,json_files_path+'_actions.json')
    df=pd.DataFrame(results_action,columns=classes)
    csv_file=os.path.join(json_files_path+'_'+model_name+'_prediction.csv')
    df.to_csv(csv_file)









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
    video_name='IMG_0882_60.mp4'
    #video_name='IMG_0876_60.mp4'
    video_id='1234567'
    path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','testing_videos')
    file_path=os.path.join(path,video_name)
    fps=15
    game_name='softball'
    wrist='right_handed'
    distance_from_camera=10     # (feet)
    evaluate_video(file_path,wrist,fps)

    end_time=time.time()
    print('Total execution time is : %d minutes and %d seconds'%(((end_time-start_time)-(end_time-start_time)%60)//60,(end_time-start_time)-((end_time-start_time)-	 (end_time-start_time)%60)))
