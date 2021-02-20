# YoLo-V3 Re-implement
# Author: @WT
# Time: 2021.02.12
# E-mail: taowang@stu.pku.edu.cn

from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *


def get_img_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (608,608))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

def parse_cfg(config_file):
    """
    Takes a configuration file
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list    
    """
    cfg = open(config_file, 'r')    # read cfg_file
    lines = cfg.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0] # get rid of empty lines
    lines = [x for x in lines if x[0] != '#'] # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks   #blocks 用于存 网络各种参数

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    According blocks to create Net
    """
    # index = 0   # indexing blocks
    net_info = blocks[0]    # get information about net
    module_list = nn.ModuleList()
    prev_filters = 3    # init_kernel_channel_num
    out_put_filters = []
    # creat 

    for index, x in enumerate(blocks[1:]):    # traverse blocks to get convolutional layer informations (like kernel_size, padding, activative_function)
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation'] # find activate function
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            kernel = int(x['filters'])
            padding = int(x['pad'])
            stride = int(x['stride'])
            kernel_size = int(x['size'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # build conv layer
            conv_layer = nn.Conv2d(prev_filters, kernel, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv_layer)

            # biuld BN layer
            if batch_normalize:
                bn = nn.BatchNorm2d(kernel)
                module.add_module('bn_{0}'.format(index), bn)
            
            # build pooling layer
            if activation == "leaky":
                act_fun = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), act_fun)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module('upsample_{}'.format(index), upsample)
        
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # start route
            start = int(x['layers'][0]) # find start layer
            # try find end layer
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)

            if end < 0:
                kernel = out_put_filters[index + start] + out_put_filters[index + end]
            else:
                kernel = out_put_filters[index + start]
        
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        
        elif x['type'] == 'yolo':

            # detect layer
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]    # according mask to find right anchors

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)
       
        module_list.append(module)
        prev_filters = kernel
        out_put_filters.append(kernel)
    
    return (net_info, module_list) # module_list 是构建的网络


# cfg_file = r'C:\Users\msi-pc\Desktop\yolo_v3\yolov3.cfg'
# info, module = create_modules(parse_cfg(cfg_file))
# # print(info)
# # print((parse_cfg(cfg_file))[1:])

# print(module)

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet ,self).__init__()
        self.blocks = parse_cfg(cfg_file)   # create blocks from cfg
        self.net_info, self.modlue_list = create_modules(self.blocks)   # build net
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        output = {}
        if_detect = 0

        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.modlue_list[i](x)
            
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(layer) for layer in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = output[i + (layers[0])]
                
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    route_map1 = output[i + layers[0]]
                    route_map2 = output[i + layers[1]]
                    x = torch.cat((route_map1, route_map2), 1)
            
            elif module_type == 'shortcut':
                from_layers = int(module['from'])
                x = output[i + from_layers] + output[i-1]
            
            elif module_type == 'yolo':
                anchors = self.modlue_list[i][0].anchors

                in_dim = int(self.net_info['height'])
                # print(in_dim)

                num_calsses = int(module['classes'])

                x = x.data
                x = predict_transform(x, in_dim, anchors, num_calsses, CUDA)
                if not if_detect:
                    detections = x
                    if_detect = 1
                else:
                    detections = torch.cat((detections, x), dim=1)

            output[i] = x #每次遍历直接将 得到的结果x 放入字典
        
        return detections

    def load_weights(self, checkpointfile):
        
        checkpoint = open(checkpointfile, 'rb')
        header = np.fromfile(checkpoint, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(checkpoint, dtype=np.float32)
        
        pos = 0

        for i in range(len(self.modlue_list)):
            module_type = self.blocks[i+1]['type'] # 从第二个block开始遍历，第一个是net_info

            if module_type == 'convolutional':
                model = self.modlue_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                print(conv)

                if (batch_normalize):
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()

                    # Load weights 按顺序load bn层参数（bias->weight->running_mean->running_var）
                    bn_biases = torch.from_numpy(weights[pos:pos + num_bn_biases])
                    pos += num_bn_biases

                    bn_weights = torch.from_numpy(weights[pos:pos + num_bn_biases])
                    pos += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[pos:pos + num_bn_biases])
                    pos += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[pos:pos + num_bn_biases])
                    pos += num_bn_biases

                    # change shape of data to the same of model data

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean.data)
                    bn_running_var = bn_running_var.view_as(bn.running_var.data)

                    # load data into the model

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                # if batch_normalize != 0:
                    # conv = model[0]
                    # load conv bias if bn doesn't exist
                    # get num of conv_bias
                    num_conv_biases = conv.bias.numel()
                    # load conv bias
                    conv_bias = torch.from_numpy(weights[pos:pos + num_conv_biases])
                    pos = pos + num_conv_biases

                    conv_bias = conv_bias.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_bias)
                
                # load conv weight
                num_conv_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[pos:pos + num_conv_weights])
                pos = pos + num_conv_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



## try net
# cfg_file = r'C:\Users\msi-pc\Desktop\yolo_v3\yolov3.cfg'
# # cfg = parse_cfg(cfg_file)
# img = get_img_input()
# # print(img.shape)
# darnet = Darknet(cfg_file)
# darnet.load_weights(r'C:\Users\msi-pc\Desktop\yolo_v3\weights\yolov3.weights')
# out = darnet(img, CUDA=False)
# print(out)
# print(out.size())
# # print(out[:,:,4])





        

                

                

                








            


            





    