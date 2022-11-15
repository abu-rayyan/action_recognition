
""" https://cv.gluon.ai/_modules/gluoncv/model_zoo/rcnn/faster_rcnn/predefined_models.html   """
#%%
import cv2
import numpy as np
import os
import sys
import shutil
import gluoncv
import gc
from gluoncv import model_zoo, data, utils
sys.path.append(os.path.abspath(os.path.join(__file__,"../")))
#%%
""" Crops the largest human figure in each frame using detection algorithm.
    'file_path' should point to the main directory containing the dataset,
    within different classes folder.
    'target_path' would create a new folder, with cropped images saved into
    their actual classes."""
#%%
def get_model():
    #detector = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    detector = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=True)
    detector.reset_class(["person"], reuse_weights=['person'])
    detector.nms_thresh=0.5
    detector.post_nms=100
    return detector
#%%
detector=get_model()
pad_factor=0.2
#%%
def get_bounding_box(full_path,h,w):

    x, trans_image = data.transforms.presets.rcnn.load_test(full_path,short=512)
    print ("***********************************************")
    print ("trans_img.shape[0]=:",trans_image.shape[0] )
    print ("trans_img.shape[1]=:",trans_image.shape[0] )
    print ("***********************************************")
    class_IDs, scores, bounding_boxs = detector(x)
    ret_value=calculate_max_area(bounding_boxs,scores)
    if ret_value is None:
        return
    else:
        _,BB=ret_value
        
    # _,BB=calculate_max_area(bounding_boxs,scores)
    
    ha=h/trans_image.shape[0]
    wa=w/trans_image.shape[1]
    x1,y1,x2,y2=(BB*np.array([ha,wa,ha,wa])).astype(int)
    return x1,y1,x2,y2
#%%
def calculate_max_area(bounding_box,scores):
    
    bounding_box=bounding_box.asnumpy()
    scores=scores.asnumpy()
    area=0
    print ("------------------------------------------")
    print ("got new value")
    print ("bounding box is:", bounding_box )
    # print ("scores is:", scores)
    print ("scores sum is:", np.sum(scores>0.3))
    print ("scores shape is:", scores.shape)
    print ("------------------------------------------")
    count=0
    scores_sum=np.sum(scores>0.5)
    if scores_sum<1:
        return
    for i in range(scores_sum):
      print ("==================================================")
      print ("value of i is:", i)
      print ("value of count is:", count)
      count=count+1
      print ("bounding box is:", bounding_box[0][i])
      try:
          x1,y1,x2,y2=bounding_box[0][i]
          print ("I AM inside try")
          width=int(x2-x1)
          height=int(y2-y1)
          if width*height>area:
            area=width*height
            print ("updated area is:", area)
            index=i
            print ("value of index is:", index) 
            print ("area is:", area)       
        

      except:
          print ("I am inside except")
          x1,y1,x2,y2=bounding_box[0]
          
          width=int(x2-x1)
          height=int(y2-y1)
          if width*height>area:
            area=width*height
            print ("updated area is:", area)
            index=i
            print ("value of index is:", index) 
            print ("area is:", area)        
        
    return index,bounding_box[0][index]


           
      
#%%
def crop(img_path):
    img=cv2.imread(img_path)
    bb_coordinates=get_bounding_box(img_path,img.shape[0],img.shape[1])
    
    if bb_coordinates is None:
        return
    else:
        x1,y1,x2,y2=bb_coordinates
        width=int((x2-x1)*pad_factor)
        height=int((y2-y1)*pad_factor)
        # return img[y1:y2,x1:x2]
        return img[np.max([0,y1-height]):y2+height,np.max([0,x1-width]):x2+width,:]


#%%
if __name__=='__main__':
    # file_path=os.path.join(os.path.abspath(os.path.join(__file__,"../")),'images_data')
    data_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data')
    full_images_path=os.path.join(data_path,'images_data')
    # file_path=('full_classes')
    cropped_images_path=os.path.join(data_path,'cropped_data')
    print(full_images_path)
    print(cropped_images_path)
    if os.path.exists(cropped_images_path):
         shutil.rmtree(cropped_images_path)
    os.makedirs(cropped_images_path)
    print('Total classes detected : {}'.format(os.listdir(full_images_path)))
    for classes in os.listdir(full_images_path):
        print('For class: {}'.format(classes))
        class_path=os.path.join(full_images_path,classes)
        if os.path.exists(os.path.join(cropped_images_path,classes)):
          shutil.rmtree(os.path.join(cropped_images_path,classes))
        os.makedirs(os.path.join(cropped_images_path,classes))

        for img_path in os.listdir(class_path):
            print('Cropping ',os.path.join(class_path,img_path))
            cropped_image=crop(os.path.join(class_path,img_path))
            if cropped_image is not None:
              cv2.imwrite(os.path.join(os.path.join(cropped_images_path,classes),'{}'.format(img_path)),cropped_image)
              del cropped_image
              gc.collect()