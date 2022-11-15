#%%
import os
import sys
import cv2
import imutils
sys.path.append(os.path.abspath(os.path.join(__file__,"/..")))
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
import mxnet as mx
import numpy as np
from position_calculations import calculate_max_area, scale_coordinates
from gluoncv.utils.viz import get_color_pallete
#%%

#%%
class segmentation:
    def __init__(self,stick_height_inches):
        self.net=self.get_models()
        self.class_index=15
        self.thresh=6
        self.stick_height=stick_height_inches
        self.ext='.png'

    def get_models(self):
        net = model_zoo.get_model('deeplab_resnet101_coco', pretrained=True)
        return net
    def person_segment(self,image):
        img = mx.ndarray.array(image)
        img = data.transforms.presets.segmentation.test_transform(img,ctx = mx.cpu(0))
        output = self.net.predict(img)
        try:
            predict = mx.nd.squeeze(output[:,self.class_index]).asnumpy()
        except mx.MXNetError:
            cX, cY=[],[]
            return cX, cY
        p=np.where(predict<self.thresh,0,1)
        p=p.astype('uint8')
        cnts = cv2.findContours(p.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        mask = np.ones(p.shape[:2], dtype="uint8") * 255
        # loop over the contours
        max_area_cnt=max(cnts,key=cv2.contourArea)
        max_area=cv2.contourArea(max_area_cnt)
        for c in cnts:
            # if the contour is bad, draw it on the mask
            cnt_area=cv2.contourArea(c)
            if cnt_area<max_area:
                cv2.drawContours(mask, [c], -1, 0, -1)
            # remove the contours from the image and show the resulting images
        img = cv2.bitwise_and(p,p, mask=mask)
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts)>1:
            print('====MORE THAN ONE CONTOUR DETECTED====')
        for c in cnts:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        return cX, cY

def convert_to_binary(img):
        imgLAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB);
        l,a,b = cv2.split(imgLAB)
        ret,binary_im = cv2.threshold(a,140,255,cv2.THRESH_BINARY)
        return binary_im

def detect_stick(img,stick_height):
        
        stick_candidates= []
        stick_height_inches = 47
        binary_im= convert_to_binary(img)

        contours,hierarchy = cv2.findContours(binary_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        check=-1
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w>7: # h/w of reference contour is 8.85
                stick_height_pixels= h
                print('height of stick detected (in number of pixels): ',stick_height_pixels)
                stick_candidates.append(cnt)
                check=1
        if check==-1:
            return None
        #stick_height_pixels
        inches_in_pixel=stick_height/stick_height_pixels

        print('number of detected sticks:',len(stick_candidates))
        return inches_in_pixel

# %%
