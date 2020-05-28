"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import random

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False ,action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

print(image_list)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def chinese_num(line):
    total = 0
    for u_char in line:
        if (u_char >= u'\u2f00' and u_char<=u'\u2fd5') or (u_char >= u'\u4e00' and u_char<=u'\u9fa5'):
            total += 1
    return total

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    #Cords_list = []

    #for i,box in enumerate(boxes):
    #    Cords = craft_utils.getVerticalCord(box,score_link,link_threshold,i)
    #    Cords = craft_utils.adjustResultCoordinates(Cords, ratio_w, ratio_h)
    #    Cords_list.append(Cords)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    #print(boxes, polys, ret_score_text)

    #return boxes, polys, ret_score_text ,Cords_list
    return boxes, polys, ret_score_text


def find_one_line(array_dic):
    #list_1 = list_1_array[0]
    final_res = {}
    flag = 0
    for index in sorted(txt_result.keys()):
        if index in txt_result:
            list_1 = txt_result[index]
            final_res[index] = []
            #txt_result.pop(index)
            y0 = (list_1[0][1] + list_1[0][7]) / 2
            w = abs(list_1[0][7] - list_1[0][1])
            for index_t in sorted(txt_result.keys()):
                item = txt_result[index_t]
                y = (item[0][1] + item[0][7]) / 2
                if abs(y - y0) < w/2:
                    final_res[index].append(item)
                    txt_result.pop(index_t)
    print(final_res)
    return final_res


def sorted_by_y(array):
    y_list = []
    for i in range(len(array)):
        y_list.append(int(array[i][0][0]))

    index_list = np.argsort(y_list)
    return index_list


if __name__ == '__main__':
    # load net
    #res = open('res.txt','w',encoding='utf8')
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))



    if args.cuda:
        net = net.cuda()
        
        input_batch = 1
        input_channel = 3
        input_h = 448
        input_w = 448
        output_batch = input_batch
        output_h = input_h / 2
        output_w = input_w / 2
        inputc = torch.randn(input_batch, input_channel, \
            input_h, input_w, device='cuda')

        outputc = net(inputc.cuda())

        dynamic_axes = {'inputc': {0: 'input_batch', 1: 'input_channel', 2: "input_h", 3: 'input_w'},'outputc': {0: 'output_batch', 1: "output_h", 2: 'output_w'}}

        output_names = ["output1","output2"]
        input_names = ["input"]
        torch.onnx.export(
                        net,
                        inputc,
                        'craft.onnx',
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )
        
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()


    
    
