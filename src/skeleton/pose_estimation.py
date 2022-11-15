#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__,"/..")))
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from position_calculations import calculate_max_area, scale_coordinates
import gc
#%%
class pose_model:
    def __init__(self):
        self.pose_net,self.detector=self.get_models()

    def get_models(self):
        pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
        detector = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
        detector.reset_class(["person"], reuse_weights=['person'])
        detector.nms_thresh=0.35
        detector.post_nms=2
        return pose_net,detector

#     def calculate_coordinates(self,full_path):
#         x, image = data.transforms.presets.rcnn.load_test(full_path,short=1024)
#         class_IDs, scores, bounding_boxs = self.detector(x)
#         person=calculate_max_area(bounding_boxs,scores)
#         pose_input, upscale_bbox = detector_to_alpha_pose(image, class_IDs, scores, bounding_boxs)
#         del x,class_IDs, scores,self.detector
#         gc.collect()
#         predicted_heatmap = self.pose_net(pose_input)
#         pred_coords, _ = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
#         pred_coords=pred_coords.asnumpy()
#         pred_coords=pred_coords[person]
#         bounding_boxs=bounding_boxs.asnumpy()
#         BB=bounding_boxs[0][person]
#         BB,pred_coords=scale_coordinates(full_path,pred_coords,image.shape[0],image.shape[1],BB)
#         del predicted_heatmap,upscale_bbox,self.pose_net,bounding_boxs,pose_input,image
#         gc.collect()
#         return BB,pred_coords
# # %%
# class pose_model_normal:
#     def __init__(self):
#         self.pose_net,self.detector=self.get_models()

#     def get_models(self):
#         pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
#         detector = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
#         detector.reset_class(["person"], reuse_weights=['person'])
#         detector.nms_thresh=0.35
#         detector.post_nms=2
#         return pose_net,detector

#     def calculate_coordinates(self,full_path):
#         x, image = data.transforms.presets.rcnn.load_test(full_path,short=1024)
#         class_IDs, scores, bounding_boxs = self.detector(x)
#         person=calculate_max_area(bounding_boxs,scores)
#         pose_input, upscale_bbox = detector_to_alpha_pose(image, class_IDs, scores, bounding_boxs)
#         del x,class_IDs, scores
#         gc.collect()
#         predicted_heatmap = self.pose_net(pose_input)
#         pred_coords, _ = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
#         pred_coords=pred_coords.asnumpy()
#         pred_coords=pred_coords[person]
#         bounding_boxs=bounding_boxs.asnumpy()
#         BB=bounding_boxs[0][person]
#         BB,pred_coords=scale_coordinates(full_path,pred_coords,image.shape[0],image.shape[1],BB)
#         del predicted_heatmap,upscale_bbox,bounding_boxs,pose_input,image
#         gc.collect()
#         return BB,pred_coords

