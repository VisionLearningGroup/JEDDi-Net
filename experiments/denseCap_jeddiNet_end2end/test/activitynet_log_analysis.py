# --------------------------------------------------------
# JEDDi-Net
# Copyright (c) 2018 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
 

import sys, os, errno
import numpy as np
import csv
import json
import copy

assert len(sys.argv) == 2, "Usage: python log_analysis.py <test_log>"
logfile = sys.argv[1]


def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_segments(data, thresh):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'sentence' : "", 'timestamp': [0, 0]}
    for l in data:
      # video name and sliding window length
      if "fg_name :" in l:
         vid = 'v_'+l.split('/')[-1]

      # frame index, time, stride
      elif "frames :" in l:
         start_frame=int(l.split()[4])
         end_frame=int(l.split()[5])
         stride = int(l.split()[6].split(']')[0])

      elif "activity:" in l:
         # label = int(l.split()[1])
         # tmp['label'] = label
         find_next = True

      elif "im_detect" in l:
         return vid, segments

      elif find_next:
         left_frame = float(l.split(',')[0][1:])*stride + start_frame
         right_frame = float(l.split(',')[1])*stride + start_frame
         if (left_frame < end_frame) and (right_frame <= end_frame):
           left  = left_frame / 25.0  # change according to the fps that you use
           right = right_frame / 25.0
           if len(l.split(', \''))==1:
               sentence = l.split(', \"')[1].split(']')[0][:-1]
               print sentence
           else:
               sentence = l.split(', \'')[1].split(']')[0][:-1]
               print sentence

               
           tmp1 = copy.deepcopy(tmp)
           tmp1['sentence'] = sentence
           tmp1['timestamp'] = [left, right]
           segments.append(tmp1)
         elif (left_frame < end_frame) and (right_frame > end_frame):
             if (end_frame-left_frame)*1.0/(right_frame-left_frame)>=0:
                 right_frame = end_frame
                 left  = left_frame / 25.0
                 right = right_frame / 25.0
                 if len(l.split(', \''))==1:
                     sentence = l.split(', \"')[1].split(']')[0][:-1]
                     print sentence
                 else:
                     sentence = l.split(', \'')[1].split(']')[0][:-1]
                     print sentence
                 tmp1 = copy.deepcopy(tmp)
                 tmp1['sentence'] = sentence
                 tmp1['timestamp'] = [left, right]
                 segments.append(tmp1)

def analysis_log(logfile, thresh):
  with open(logfile, 'r') as f:
    lines = f.read().splitlines()
  predict_data = []
  res = {}
  for l in lines:
    if "gt_classes :" in l:
      predict_data = []
    predict_data.append(l)
    if "im_detect:" in l:
      vid, segments = get_segments(predict_data, thresh)
      if vid not in res:
        res[vid] = []
      res[vid] += segments
  return res

segmentations = analysis_log(logfile, thresh = 0.0)



res = {'version': 'VERSION 1.0', 
       'external_data': {'used': True, 'details': 'C3D pre-trained on sport-1M training set'},
       'results': {}}
for vid, vinfo in segmentations.iteritems():
  res['results'][vid] = vinfo


with open('results.json', 'w') as outfile:
  json.dump(res, outfile)
