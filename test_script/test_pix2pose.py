import os,sys
from math import radians
import csv
import cv2
from skimage.transform import resize

from matplotlib import pyplot as plt
import time
import random
import numpy as np
import transforms3d as tf3d

import json

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

if(len(sys.argv)!=4):
    print("python3 test_script/test_pix2pose.py [gpu_id] [cfg file] [dataset_name]")
    sys.exit()
    
gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import tensorflow as tf
from bop_toolkit_lib import inout
from tools import bop_io
from pix2pose_util import data_io as dataio
from pix2pose_model import ae_model as ae
from pix2pose_model import recognition_custom as recog
from pix2pose_util.common_util import get_bbox_from_mask

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

cfg_fn =sys.argv[2]
cfg = inout.load_json(cfg_fn)

#from tools.mask_rcnn_util import BopInferenceConfig
from skimage.transform import resize

# Load the JSON data from the file
print()
det_path = "/dataset/deform_dataset/test_img/scene_gt_bb_dummy.json"
print(det_path)
print()

with open(det_path, 'r') as json_file:
    data = json.load(json_file)

score_type = cfg["score_type"]
#1-scores from a 2D detetion pipeline is used (used for the paper)
#2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, sued for the BOP challenge)

task_type = cfg["task_type"]
#1-Output all results for target object in the given scene
#2-ViVo task (2019 BOP challenge format, take the top-n instances)
cand_factor =float(cfg['cand_factor'])

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

dataset=sys.argv[3]

bop_dir,test_dir,model_plys,\
model_info,model_ids,rgb_files,\
depth_files,mask_files,mask_visib_files,gts,\
cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True,train=False)

im_width,im_height =cam_param_global['im_size'] 
cam_K = cam_param_global['K']
model_params =inout.load_json(os.path.join(bop_dir+"/models_xyz/",cfg['norm_factor_fn']))

img_type='rgb'

if("target_obj" in cfg.keys()):
    target_obj = cfg['target_obj']
    remove_obj_id=[]
    incl_obj_id=[]
    for m_id,model_id in enumerate(model_ids):
        if(model_id not in target_obj):
            remove_obj_id.append(m_id)
        else:
            incl_obj_id.append(m_id)
    for m_id in sorted(remove_obj_id,reverse=True):
        print("Remove a model:",model_ids[m_id])
        del model_plys[m_id]
        del model_info['obj_{:06}'.format(model_ids[m_id])]
        
    model_ids = model_ids[incl_obj_id]

print(model_ids)
    

print("Camera info-----------------")
print(im_width,im_height)
print(cam_K)
print("-----------------")

'''
standard estimation parameter for pix2pose
'''

th_outlier=cfg['outlier_th']
dynamic_th = True
if(type(th_outlier[0])==list):
    print("Individual outlier thresholds are applied for each object")
    dynamic_th=False
    th_outliers = np.squeeze(np.array(th_outlier))
th_inlier=cfg['inlier_th']
th_ransac=3

dummy_run=False

'''
Load pix2pose inference weights
'''
load_partial=False
obj_pix2pose=[]
obj_names=[]
image_dummy=np.zeros((im_height,im_width,3),np.uint8)
if( 'backbone' in cfg.keys()):
    backbone = cfg['backbone']
else:
    backbone = 'paper'
for m_id,model_id in enumerate(model_ids):
    model_param = model_params['{}'.format(model_id)]
    obj_param=bop_io.get_model_params(model_param)
    #new_model_id = 404*(model_id-1) + 1 
    weight_dir = bop_dir+"/pix2pose_weights_no_bg/{:02d}".format(model_id)
    if(backbone=='resnet50'):
        weight_fn = os.path.join(weight_dir,"inference_resnet_model.hdf5")
        if not(os.path.exists(weight_fn)):
            weight_fn = os.path.join(weight_dir,"inference_resnet50.hdf5")
    else:
        weight_fn = os.path.join(weight_dir,"inference.hdf5")
    print("load pix2pose weight for obj_{} from".format(model_id),weight_fn)
    if not(dynamic_th):
        th_outlier = [th_outliers[m_id]] #provid a fixed outlier value
        print("Set outlier threshold to ",th_outlier[0])    
    recog_temp = recog.pix2pose(weight_fn,camK= cam_K,
                                res_x=im_width,res_y=im_height,obj_param=obj_param,
                                th_ransac=th_ransac,th_outlier=th_outlier,
                                th_inlier=th_inlier,backbone=backbone)
    print(obj_param)
    exit(1)
    obj_pix2pose.append(recog_temp)    
    obj_names.append(model_id)

rgb_path = "/dataset/deform_dataset/test_img/000003.jpg"

image_t = inout.load_im(rgb_path)

#print(obj_pix2pose)
#print(data)

cat_list = ["chips", "juice", "paste", "pringles", "shampoo", "teabox", "pastry"]

for obj in data:
    obj_id = obj["obj_id"]
    bbox = obj["bbox_est"]

    new_obj_id = int(np.ceil(obj_id / 404)) - 1

    print(cat_list[new_obj_id])

    image_crop = image_t[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]

    plt.imshow(image_crop)
    plt.title(cat_list[new_obj_id])
    plt.show()

    roi = np.array([bbox[1],bbox[0],bbox[1]+bbox[3],bbox[0]+bbox[2]])

    img_pred,mask_pred,rot_pred,tra_pred,frac_inlier,bbox_t = obj_pix2pose[new_obj_id].est_pose(image_t,roi.astype(np.int)) 

    #print(img_pred)

    #plt.imshow(img_pred)
    #plt.title(cat_list[new_obj_id])
    #plt.show()

    print(mask_pred)
    print(rot_pred)
    print(tra_pred)
    print(frac_inlier)
    print(bbox_t)






