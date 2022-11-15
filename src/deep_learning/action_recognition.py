#%%
from tensorflow import keras
import sys
import cv2
import numpy as np
import os
import gc
#%%
class action_model:
    def __init__(self,model_path):
        if not os.path.exists(model_path):
            print('Error: Model does not exist. Please train the model first')
            sys.exit()
        self.model=keras.models.load_model(model_path)
        # self.height,self.width=self.model.input.shape[1:3]
    def predict_action(self,data,pose_labels):
        predict=self.model.predict(data)
        print(predict)
        predict_label=predict.argmax(axis=1)
        position_text=pose_labels[int(predict_label)]
        return position_text,predict_label
    def predict_hybrid_action(self,pose_data,image):
        height,width=self.model.input[1].shape[1:3]
        image=cv2.resize(image,(width,height), interpolation = cv2.INTER_AREA)
        image=np.expand_dims(image,axis=0)
        print(f'data size is {pose_data.shape}')
        print(f'image size is {image.shape}')
        predict=self.model.predict([pose_data,image])
        print(predict)
        del image, self.model
        gc.collect()
        return predict
# %%
class action_model_normal:
    def __init__(self,model_path):
        if not os.path.exists(model_path):
            print('Error: Model does not exist. Please train the model first')
            sys.exit()
        self.model=keras.models.load_model(model_path)
        # self.height,self.width=self.model.input[1].shape[1:3]
    def predict_action(self,data,pose_labels):
        predict=self.model.predict(data)
        print(predict)
        predict_label=predict.argmax(axis=1)
        position_text=pose_labels[int(predict_label)]
        return position_text,predict_label
    def predict_hybrid_action(self,pose_data,image):
        height,width=self.model.input[1].shape[1:3]
        image=cv2.resize(image,(width,height), interpolation = cv2.INTER_AREA)
        image=np.expand_dims(image,axis=0)
        print(f'data size is {pose_data.shape}')
        print(f'image size is {image.shape}')
        predict=self.model.predict([pose_data,image])
        print(predict)
        del image
        gc.collect()
        return predict
