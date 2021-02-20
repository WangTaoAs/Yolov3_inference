# YoLo-V3 Re-implement
# Author: @WT
# Time: 2021.02.12
# E-mail: taowang@stu.pku.edu.cn

from __future__ import division

import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def load_classes(names_file):
    namesfile = open(names_file, 'r')
    names = namesfile.read().split('\n')[:-1]
    return names

def prep_img(img, in_dim):
    img = (letterbox_image(img, (in_dim, in_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy() # 先将 BGR->RGB， 然后将(h*w*c)->(c*h*w)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # (c*h*w)(numpy) -> (1*c*h*w)(tensor) 
    return img

            
def bbox_iou(box1, box2):

    x1,y1,x2,y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    xx1, yy1, xx2, yy2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    area_box1 = (y2 - y1 + 1) * (x2 - x1 + 1)
    area_box2 = (yy2 - yy1 + 1) * (xx2 - xx1 + 1)
    
    inter_x1 = torch.max(x1, xx1)
    inter_y1 = torch.max(y1, yy1)
    inter_x2 = torch.min(x2, xx2)
    inter_y2 = torch.min(y2, yy2)

    interarea = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    iou = interarea / (area_box1 + area_box2 - interarea)

    return iou            


def unique(tensor):
    ts_unique = tensor.cpu().numpy()
    np_unique = np.unique(ts_unique)
    ts_back_unique = torch.from_numpy(np_unique)

    tensor_ret = tensor.new(ts_back_unique.shape)
    tensor_ret.copy_(ts_back_unique)
    return tensor_ret


def predict_transform(x, in_dim, anchors, num_calsses, CUDA = True):

    batch_size = x.size(0)
    stride = in_dim // x.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_calsses
    num_anchors = len(anchors)

    x = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    x = x.transpose(1,2).contiguous()
    x = x.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    x[:,:,0] = torch.sigmoid(x[:,:,0])
    x[:,:,1] = torch.sigmoid(x[:,:,1])
    x[:,:,4] = torch.sigmoid(x[:,:,4])

    grids = np.arange(grid_size)
    a,b = np.meshgrid(grids, grids)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset),dim=1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    x[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    x[:,:,2:4] = torch.exp(x[:,:,2:4]) * anchors
    x[:,:,5:5+num_calsses] = torch.sigmoid((x[:,:,5:5 + num_calsses]))
    x[:,:,:4] *= stride

    return x

def get_result(x, confidence, num_classes, nms_threshold=0.4):

    mask = (x[:,:,4]>confidence).float().unsqueeze(2)
    x = x*mask

    # In order to calculate IOU, a pair of corner point is better so we tansform the center(x,y) and (height, width) to coner points

    corner_point = x.new(x.shape)
    corner_point[:,:,0] = (x[:,:,0] - x[:,:,2]/2)
    corner_point[:,:,1] = (x[:,:,1] - x[:,:,3]/2)
    corner_point[:,:,2] = (x[:,:,0] + x[:,:,2]/2)
    corner_point[:,:,3] = (x[:,:,1] + x[:,:,3]/2)
    x[:,:,:4] = corner_point[:,:,:4]

    batch_size = x.size(0)

    flag = False

    for ind in range(batch_size):
        pred = x[ind]     # IMG Tensor

        max_conf, max_conf_score = torch.max(pred[:,5:5+num_classes], 1)    # 找到5-80的最大conf，并得到序号，该序号就是类别
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (pred[:,:5], max_conf, max_conf_score)    # 前5个是(0->4)是坐标信息，后面是conf，以及conf的索引，重新叠放
        pred = torch.cat(seq, 1)

        non_zero_ind = torch.nonzero(pred[:,4])
        try:
            pred_ = pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if pred_.shape[0] == 0:
            continue

        img_classes = unique(pred_[:,-1])   # 获取类别序号

        # NMS
        for cls in img_classes:
            cls_mask = pred_*(pred_[:,-1] == cls).float().unsqueeze(1) # 在pred_最后一列中 寻找与cls相同（相同与1相乘，其余为0，然后相乘回去得到一个向量，并扩充1维）
            cls_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()  # 去cls_mask的倒数第二维（因为扩充了一维）中寻找非0项，返回序号
            img_pre_classes = pred_[cls_mask_ind].view(-1,7)

            # 将每个bbox按照img_pre_classes[:,:.4]（即objectness score）排序
            conf_sort_index = torch.sort(img_pre_classes[:,4], descending=True)[1]
            img_pre_classes = img_pre_classes[conf_sort_index]
            bbox_id = img_pre_classes.size(0) # num of bbox bbox的个数

            for i in range(bbox_id):
                try:
                    ious = bbox_iou(img_pre_classes[i].unsqueeze(0), img_pre_classes[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break
            
                iou_mask = (ious < nms_threshold).float().unsqueeze(1)
                img_pre_classes[i+1:] *= iou_mask   # 小于阈值的保留，大于阈值的丢弃

                non_zero_ind = torch.nonzero(img_pre_classes[:,4]).squeeze()
                img_pre_classes = img_pre_classes[non_zero_ind].view(-1,7)
            
            batch_ind = img_pre_classes.new(img_pre_classes.size(0), 1).fill_(ind)
            seq = batch_ind, img_pre_classes

            if not flag:
                output = torch.cat(seq, 1)
                flag = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0



def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

