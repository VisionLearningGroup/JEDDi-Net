# --------------------------------------------------------
# JEDDi-Net
# Copyright (c) 2018 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import cPickle
import subprocess
import numpy as np
#import cv2
from util import *

#USE_FLIPPED = True


FPS = 25
LENGTH = 768
min_length = 3 # frame num (filter out one second)
overlap_thresh = 0.7 
STEP = LENGTH / 4
# frames are converted in 25 fps
WINS = [LENGTH * 8]
max_words = 30

vocab_out_path = './vocabulary.txt'
TRAIN_META_FILE = 'gt/train.json'
train_data = json.load(open(TRAIN_META_FILE))

#pre-processing

train_caption =[]
for vid in train_data.keys():
  vinfo = train_data[vid]
  for k in vinfo['sentences']:
    train_caption.append(split_sentence(k))

print '\nthe total number of captions are:',len(train_caption)

#init-vocabulary
vocab,vocab_inverted = init_vocabulary(train_caption, min_count=5)  
dump_vocabulary(vocab_inverted, vocab_out_path)



path = '............/datasets/activityNet/frames'
print('Generate Training Segments')
train_segment = generate_segment(path, 'training',train_data,vocab,max_words)

VAL_META_FILE_1 = 'gt/val_1.json'
val_data_1 = json.load(open(VAL_META_FILE_1))
print('Generate Validation Segments 1')
val_segment_1 = generate_segment(path,'validation',val_data_1,vocab,max_words)

VAL_META_FILE_2 = 'gt/val_2.json'
val_data_2 = json.load(open(VAL_META_FILE_2))
print('Generate Validation Segments 2')
val_segment_2 = generate_segment(path,'validation',val_data_2,vocab,max_words)

val_data = copy.deepcopy(val_data_1)
for key, item in val_data.iteritems():
  if key in val_data_2:
    assert item['duration'] == val_data_2[key]['duration']
    for k in {'sentences', 'timestamps'}:
      item[k] += val_data_2[key][k]

print('Generate Combined Validation Segments')
val_segment = generate_segment(path,'validation',val_data,vocab,max_words)


def generate_roi(rois, rois_lstm, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = ( rois[:,:2] - start ) / stride
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]+1
  #tmp['gt_classes'] = rois[:,2]
  tmp['gt_classes'] = np.ones(rois.shape[0])
  tmp['input_sentence'] = rois_lstm[:,0]
  tmp['cont_sentence'] = rois_lstm[:,1]
  tmp['target_sentence'] = rois_lstm[:,2]
  tmp['max_classes'] = np.ones(rois.shape[0])
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = path + '/'+split+'/' + video
  tmp['fg_name'] = path + '/'+split+'/' + video
  if not os.path.isfile(tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
    print  tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'
    raise
  return tmp

def generate_roidb(split, segment):
  max_num_seg = 0
  VIDEO_PATH = '%s/%s/' % (path,split)
  #VIDEO_PATH = 'frames/'
  video_list = set(os.listdir(VIDEO_PATH))
  remove = 0
  overall = 0
  duration = []
  roidb = []
  for vid in segment.keys():
    if vid[2:] in video_list:
      length = len(os.listdir(VIDEO_PATH + vid[2:]))
      #db = np.array(segment[vid])
      seg_tmp = segment[vid]
      db=[]
      db_lstm = []
      #db_lstm = np.ones((1,3,max_words*3))
      for s in seg_tmp:
        db.append([s[0],s[1]])
        db_lstm.append([s[2],s[3],s[4]])
      db = np.array(db)
      db_lstm = np.array(db_lstm)
      
      overall += len(db)
      if len(db) == 0:
        continue
      db = db * FPS
      debug = []

      for win in WINS:
        stride = win / LENGTH
        step = stride * STEP
        # Forward Direction
        for start in xrange(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length
          rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] < start))]
          rois_lstm = db_lstm[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] < start))]

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:,1] - rois[:,0] + 1
            rois = rois[duration >= min_length]
            rois_lstm = rois_lstm[duration >= min_length]
            

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end-1, rois[:,1]) - np.maximum(start, rois[:,0]) +1)*1.0
            overlap = time_in_wins / (rois[:,1] - rois[:,0] + 1)
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap >= overlap_thresh]
            rois_lstm = rois_lstm[overlap >= overlap_thresh]

          # Add data
          if len(rois) > 0:
            rois[:,0] = np.maximum(start, rois[:,0])
            rois[:,1] = np.minimum(end-1, rois[:,1])
            if rois.shape[0] > max_num_seg:
              max_num_seg = rois.shape[0] 
            tmp = generate_roi(rois, rois_lstm, vid[2:], start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)
            for d in rois:
              debug.append(d)
              
        # Backward Direction
        for end in xrange(length, win-1, - step):
          start = end - win
          assert start >= 0
          rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] < start))]
          rois_lstm = db_lstm[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] < start))]

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:,1] - rois[:,0] + 1
            rois = rois[duration >= min_length]
            rois_lstm = rois_lstm[duration >= min_length]

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end-1, rois[:,1]) - np.maximum(start, rois[:,0]) + 1)*1.0
            overlap = time_in_wins / (rois[:,1] - rois[:,0] + 1 )
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap >= overlap_thresh]
            rois_lstm = rois_lstm[overlap >= overlap_thresh]

          # Add data
          if len(rois) > 0:
            rois[:,0] = np.maximum(start, rois[:,0])
            rois[:,1] = np.minimum(end-1, rois[:,1])
            if rois.shape[0] > max_num_seg:
              max_num_seg = rois.shape[0] 
            tmp = generate_roi(rois, rois_lstm, vid[2:], start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)
            for d in rois:
              debug.append(d)

      debug_res=[list(x) for x in set(tuple(x) for x in debug)]
      if len(debug_res) < len(db):
        remove += len(db) - len(debug_res)

  print '\nthe maximum number of segments in each window is:', max_num_seg
  print remove, ' / ', overall
  return roidb




USE_FLIPPED = False
val_roidb_1 = generate_roidb('validation', val_segment_1)
print len(val_roidb_1)

# val_roidb_2 = generate_roidb('validation', val_segment_2)
# print len(val_roidb_2)

# val_roidb = generate_roidb('validation', val_segment)
# print len(val_roidb)

print "Save dictionary"



cPickle.dump(val_roidb_1, open('./val_data_modified_3fps_caption_768_1.pkl','w'), cPickle.HIGHEST_PROTOCOL)
#cPickle.dump(val_roidb_2, open('./val_data_modified_3fps_caption_768_2.pkl','w'), cPickle.HIGHEST_PROTOCOL)



