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

img_folder = "test_img" #test_img
img_number = 3 #3

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def get_3d_points(points_2d, ply, im_width, im_height, mask_pred):
    p2d = []
    p3d = []
    rgb_3d = ply['colors'].astype(int)
    for i, rgb in enumerate(points_2d):
        a = np.all(rgb_3d <= rgb+5, axis=1)
        b = np.all(rgb_3d >= rgb-5, axis=1)

        index = np.where(a*b)
        if len(index[0]) > 0:
            v = int(mask_pred[i] % im_width)
            u = int(mask_pred[i] / im_width)
            p2d.append((u, v))
            p3d.append((ply['pts'][index[0][0]][0], ply['pts'][index[0][0]][1], ply['pts'][index[0][0]][2]))

    return (np.array(p2d)).astype('float32'), (np.array(p3d)).astype('float32')


def get_3d_box_points(vertices, model_scale):    
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    pts=[]
    pts.append([x_min,y_min,z_min])#0
    pts.append([x_min,y_min,z_max])#1        
    pts.append([x_min,y_max,z_min])#2
    pts.append([x_min,y_max,z_max])#3        
    pts.append([x_max,y_min,z_min])#4
    pts.append([x_max,y_min,z_max])#5                
    pts.append([x_max,y_max,z_min])#6
    pts.append([x_max,y_max,z_max])#7
    if(x_max>1): #assume, this is mm scale
        return np.array(pts)*model_scale
    else:
        return np.array(pts)

def draw_3d_poses(obj_box,tf,img, camK):
    lines=[[0,1],[0,2],[0,4],[1,5],[1,3],[2,6],[2,3],[3,7],
            [4,6],[4,5],[5,7],[6,7]]   
    direc= [2,1,0,0,1,0,2,0,1,2,1,2]
    proj_2d = np.zeros((8,2),dtype=np.int)
    tf_pts = (np.matmul(tf[:3,:3],obj_box.T)+tf[:3,3,np.newaxis]).T
    max_z = np.max(tf_pts[:,2])
    min_z = np.min(tf_pts[:,2])
    z_diff = max_z-min_z
    z_mean = (max_z+min_z)/2
    proj_2d[:,0] = tf_pts[:,0]/tf_pts[:,2]*camK[0,0]+camK[0,2]
    proj_2d[:,1] = tf_pts[:,1]/tf_pts[:,2]*camK[1,1]+camK[1,2]    
    for l_id in range(len(lines)):        
        line = lines[l_id]
        dr= direc[l_id]
        mean_z_line =( tf_pts[line[0],2] +tf_pts[line[1],2])/2
        color_amp = (z_mean-mean_z_line)/z_diff*255
        color = np.zeros((3),dtype=np.uint8)
        color[dr] = min(128+color_amp,255)
        if(color[dr]<10):
            continue
        cv2.line(img,(proj_2d[line[0],0],proj_2d[line[0],1]),
                    (proj_2d[line[1],0],proj_2d[line[1],1]),
                    (int(color[0]),int(color[1]),int(color[2])),2)
    
    pt_colors=[[255,255,255],[255,0,0],[0,255,0],[0,0,255]]
    for pt_id,color in zip([0,4,2,1],pt_colors): #origin, x,y,z, points
        pt =proj_2d [pt_id]
        cv2.circle(img,(int(pt[0]),int(pt[1])),1,(color[0],color[1],color[2]),5)
    return img

def get_bb(dir_path, scene_id, cat_list, score):
    scene_gt_bb_dummy = {}

    with open(os.path.join(dir_path, "scene_gt_info.json"), 'r') as json_file:
        scene_gt_info = json.load(json_file)

    with open(os.path.join(dir_path, "scene_gt.json"), 'r') as json_file:
        scene_gt = json.load(json_file)

    for rgb in os.listdir(os.path.join(dir_path, "rgb")):
        scene_list = []
        image_id = int(rgb.split(".")[0])
        for index, obj in enumerate(scene_gt["{}".format(image_id)]):
            obj_id = obj['obj_id']
            new_obj_id = 404*(int(np.ceil(obj_id/404))-1)+1
            try:
                bbox = scene_gt_info["{}".format(image_id)][index]['bbox_obj']
            except:
                continue

            scene = {"obj_id": new_obj_id, "bbox_est": bbox, "score": score}
            scene_list.append(scene)

        if len(scene_list) > 0:
            scene_gt_bb_dummy["{}/{}".format(scene_id, image_id)] = scene_list

    return scene_gt_bb_dummy

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

cfg_fn =sys.argv[2]
cfg = inout.load_json(cfg_fn)

#from tools.mask_rcnn_util import BopInferenceConfig
from skimage.transform import resize

score_type = cfg["score_type"]
#1-scores from a 2D detetion pipeline is used (used for the paper)
#2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, sued for the BOP challenge)

task_type = cfg["task_type"]
#1-Output all results for target object in the given scene
#2-ViVo task (2019 BOP challenge format, take the top-n instances)
cand_factor =float(cfg['cand_factor'])

model_scale = cfg['model_scale']

if(int(cfg['icp'])==1):
    icp = True
else:
    icp = False

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
    weight_dir = bop_dir+"/p_0.2_pix2pose_weights_no_bg/{:02d}".format(model_id)
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

    obj_pix2pose.append(recog_temp)    
    obj_names.append(model_id)

cat_list = ["chips", "juice", "paste", "pringles", "shampoo", "teabox", "pastry"]

for scene_number in sorted(os.listdir(os.path.join(bop_dir,'train_pbr'))):
    if int(scene_number) < 33:
        continue
    results = []
    with open(os.path.join(bop_dir,'train_pbr',scene_number,'scene_gt.json'), 'r') as json_file:
        scene_gt = json.load(json_file)

    scene_bb = get_bb(os.path.join(bop_dir,'train_pbr',scene_number), scene_number, cat_list, 1.0)

    for image_name in sorted(os.listdir(os.path.join(bop_dir,'train_pbr',scene_number,'rgb'))):
        print("{} / {}".format(scene_number, image_name))
        rgb_path = os.path.join(bop_dir,'train_pbr',scene_number,'rgb',image_name)
        #rgb_path = "/dataset/deform_dataset/{}/{:06}.jpg".format(img_folder, img_number)
        image_t = inout.load_im(rgb_path)
        image_number = int(image_name.split('.')[0])

        result_scores=[]
        result_poses=[]
        result_ids=[]
        result_bbox=[]
        score = 1
        pred_score = 0
        max_pred = -1
        obj_bboxes=[]
        img_pose=np.asarray(np.copy(image_t))
        rotation_vectors = []
        translation_vectors = []

        for obj in scene_bb['{}/{}'.format(scene_number, image_number)]:
            obj_id = obj["obj_id"]
            bbox = obj["bbox_est"]

            new_obj_id = int(np.ceil(obj_id / 404)) - 1

            roi = np.array([bbox[1],bbox[0],bbox[1]+bbox[3],bbox[0]+bbox[2]])

            img_pred,mask_pred,rot_pred,tra_pred,frac_inlier,bbox_t = obj_pix2pose[new_obj_id].est_pose(image_t,roi.astype(np.int)) 
            try:
                tra_pred = np.flip(tra_pred, 0)
                rot_pred = np.flip(rot_pred, (0, 1))
            except:
                print("a")

            np.arange(0, im_width*im_height)
            mask_from_detect = np.reshape(np.zeros(im_height*im_width),(im_height,im_width))
            mask_from_detect[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
            union_mask = np.logical_or(mask_from_detect,mask_pred)
            union = np.sum(union_mask)
            if(union==0):
                mask_iou=0
                score = 0
            else:
                mask_iou = np.sum(np.logical_and(mask_from_detect,mask_pred))/union
                pred_score = score*frac_inlier*mask_iou*1000

            if(frac_inlier==-1):
                    continue        
                
            tra_pred = tra_pred*model_scale #mm to m
            
            if tra_pred[2] > max_pred and tra_pred[2] < 5:
                max_pred = tra_pred[2]

            if(tra_pred[2]<0.1 or tra_pred[2]>5): #0.1
                continue

            pred_tf=np.eye(4)
            pred_tf[:3,:3]=rot_pred
            pred_tf[:3,3]=tra_pred
            result_scores.append(pred_score)
            result_poses.append(pred_tf)
            result_ids.append(new_obj_id)
            result_bbox.append(roi)

        #render detection results
        #render pose estimation results      
        #for o_id, tf,score,roi, bbox_3d in zip(result_ids,result_poses,result_scores,result_bbox, obj_bboxes):
        for o_id, tf,score,roi in zip(result_ids,result_poses,result_scores,result_bbox):
            #img_pose=np.asarray(np.copy(image_t))
            #img_tf=np.asarray(np.copy(image_t))

            tf_real = np.copy(tf)

            for scene_info in scene_gt["{}".format(int(scene_number))]:
                scene_oid = int(np.ceil(int(scene_info['obj_id']) / 404)) - 1
                if scene_oid == o_id:
                    tf_real[:3,:3] = np.asarray(scene_info['cam_R_m2c']).reshape((3,3))
                    tf_real[:3, 3] = np.asarray(scene_info['cam_t_m2c'])/1000

                    # #print(score)
                    if score > 0:
                        r, _ = cv2.Rodrigues(tf[:3,:3].dot(tf_real[:3,:3].T))
                        rotation_error_from_identity = np.linalg.norm(r)
                        MSE = np.square(np.subtract(tf_real[:3,3],tf[:3,3])).mean() 
                        
                        results.append({"scene":scene_number, 
                                            "img": image_number,
                                            "obj":cat_list[o_id],
                                            "score": score, 
                                            "rot_error":rotation_error_from_identity*180/np.pi,
                                            "trans_error":MSE})


    # Writing to sample.json
    with open(os.path.join(bop_dir,'result_error','{}.json'.format(scene_number)), "w") as outfile:
        json.dump(results, outfile)







