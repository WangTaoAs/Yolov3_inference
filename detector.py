# YoLo-V3 Re-implement
# Author: @WT
# Time: 2021.02.18
# E-mail: taowang@stu.pku.edu.cn

from __future__ import division

import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from torch.autograd import Variable

def arg_parse():
    
    parser = argparse.ArgumentParser(description='Yolov3 Detection Module')

    parser.add_argument('--images', dest='images', help='Image / Directory containing images to perform detection upon', default='imgs', type=str)
    parser.add_argument('--det', dest='det', help='Image / Directory to store detection to', default='det', type=str)
    parser.add_argument('--bs', dest='bs', help='Batch_size', default=1)
    parser.add_argument('--lr', dest='lr', help='learning_rate', default=0.001)
    parser.add_argument('--confidence', dest='confidence',help='Object Confidence to filter prediction', default=0.5)
    parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS Thresh', default=0.4)
    parser.add_argument('--cfg', dest='cfgfile', help='Config file', default='./yolov3.cfg',type=str)
    parser.add_argument('--weights', dest='weightsfile', help='weightsfile',default='weights/yolov3.weights',type=str)
    parser.add_argument('--resolution', dest='resolution', help='Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',default='608',type=str)
    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_calsses = 80
classes = load_classes('data/coco.names') # a list

print('Loading Network......')
model = Darknet(args.cfgfile)    # build model
model.load_weights(args.weightsfile) # load weights
print('Network successfully loaded......')

model.net_info['height'] = args.resolution
model.net_info['learning_rate'] = args.lr
input_dim = int(model.net_info['height'])
assert input_dim % 32 == 0 
assert input_dim > 32

if CUDA:
    model.cuda()

model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print('No file or directory with the name {}'.format(images))
    exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]  # OPENCV read an image as Numpy array, need to convert to Tensor

im_batches = list(map(prep_img, loaded_ims, [input_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

# 处理batch size
leftover = 0
if len(im_dim_list) % batch_size:   # 整除 则不标记， 不整除标记为1
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover  # 整除刚好为num_batch, 不整除 +1
    im_batches = [torch.cat((im_batches[i*batch_size : min((i+1)*batch_size, 
                        len(im_batches))])) for i in range(num_batches)] # 从 i*bs -> (i+1)*bs

write = 0
if CUDA:
    im_dim_list =im_dim_list.cuda()

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    start = time.time() # 开始时间
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():   # 在inference 时用torch.no_grad可以加速，减少不必要的运算
        prediction = model(Variable(batch), CUDA)
    
    prediction = get_result(prediction, confidence, num_calsses, nms_threshold=nms_thresh)

    end = time.time()    #读取一个batch时间
    # print state
    if type(prediction) == int:

        for img_num, iamge in enumerate(imlist[i*batch_size : min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + img_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split('/')[-1], (end-start)/batch_size))
            print('{0:20s} {1:s}'.format('Object Detected:', ''))
            print('--------------------------------------------')
        continue  
    prediction[:,0] += i*batch_size # 没太懂
        
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))
    
    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print('No detections were made')
    exit()


im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(608/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (input_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (input_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i,[1,3]] = torch.clamp(output[i,[1,3]], 0.0, im_dim_list[i,0])
    output[i,[2,4]] = torch.clamp(output[i,[2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()
class_load = time.time()

colors = pkl.load(open("pallete", "rb"))
draw_time = time.time()
# 画框
def draw(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = '{0}'.format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    text = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + text[0] + 3, c1[1] + text[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0],c1[1]+text[1]+4), cv2.FONT_HERSHEY_PLAIN,1,[225,255,255],1)
    return img


list(map(lambda x: draw(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))
list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print('SUMMARY')
print('-----------------------------------------------------------')
print("{:25s}: {}".format("Task", "Time consume (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw_time))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()










