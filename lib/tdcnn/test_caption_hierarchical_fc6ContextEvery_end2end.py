# --------------------------------------------------------
# JEDDi-Net
# Copyright (c) 2018 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

"""Test a JEDDi-Net network."""

from tdcnn.config import cfg
from tdcnn.twin_transform import clip_wins, twin_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from tdcnn.nms_wrapper import nms
import cPickle
from utils.blob import video_list_to_blob, prep_im_for_blob
import os
import random


def softmax(softmax_inputs, temp):
    shifted_inputs = softmax_inputs - softmax_inputs.max()
    exp_outputs = np.exp(temp * shifted_inputs)
    exp_outputs_sum = exp_outputs.sum()
    if np.isnan(exp_outputs_sum):
        return exp_outputs * float('nan')
    assert exp_outputs_sum > 0
    if np.isinf(exp_outputs_sum):
        return np.zeros_like(exp_outputs)
    eps_sum = 1e-20
    return exp_outputs / max(exp_outputs_sum, eps_sum)


def random_choice_from_probs(softmax_inputs, temp=1):
    # temperature of infinity == take the max
    if temp == float('inf'):
        return np.argmax(softmax_inputs)
    probs = softmax(softmax_inputs, temp)
    r = random.random()
    cum_sum = 0.
    for i, p in enumerate(probs):
        cum_sum += p
        if cum_sum >= r: return i
    return 1  # return UNK?


def generate_sentence(net, fc6, temp=float('inf'), output='predict', max_words=50):
    cont_input = np.array([0])
    word_input = np.array([0])
    sentence = []
    while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
#        print cont_input.shape
#        print fc6.min()
        net.forward(cont_sentence=cont_input, input_sentence=word_input, caption_fc6=fc6.reshape(1,fc6.shape[0])) 
        output_preds = net.blobs[output].data[0, 0, :]
        sentence.append(random_choice_from_probs(output_preds, temp=temp))
        cont_input[0] = 1
        word_input[0] = sentence[-1]
    return sentence


def generate_sentence_t1(net, caption_fc6, temp=float('inf'), output='probs', max_words=50):
    cont_input = np.array([0])
    word_input = np.array([0])
    sentence = []
    while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
        net.forward(cont_sent_1=cont_input, input_sent_1=word_input, fc6_1=caption_fc6)
        output_preds = net.blobs[output].data[0, 0, :]
        sentence.append(random_choice_from_probs(output_preds, temp=temp))
        cont_input[0] = 1
        word_input[0] = sentence[-1]
    return sentence

def generate_sentence_t2(net, caption_fc6, topic, temp=float('inf'), output='probs', max_words=50):
    cont_input = np.array([0])
    word_input = np.array([0])
    sentence = []
    while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
        net.forward(cont_sentence=cont_input, input_sentence=word_input, fc6=caption_fc6, lstm_controller_gt=topic)
        output_preds = net.blobs[output].data[0, 0, :]
        sentence.append(random_choice_from_probs(output_preds, temp=temp))
        cont_input[0] = 1
        word_input[0] = sentence[-1]
    return sentence






def _get_video_blob(roidb,vocab):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """

    processed_videos = []

    item = roidb

    for key in item:
      print key, ": ", item[key]
      if key=='target_sentence':
          for dd in item[key]:
              input_vocab = [vocab[index] for index in dd if index!=-1]
              print key, ": ", input_vocab
    video_length = cfg.TEST.LENGTH[0]
    video = np.zeros((video_length, cfg.TEST.CROP_SIZE,
                      cfg.TEST.CROP_SIZE, 3))

    j = 0
    random_idx = [int(cfg.TEST.FRAME_SIZE[1]-cfg.TEST.CROP_SIZE) / 2,
                  int(cfg.TEST.FRAME_SIZE[0]-cfg.TEST.CROP_SIZE) / 2]

    if cfg.INPUT == 'video':
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        for idx in xrange(video_info[1], video_info[2], video_info[3]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))  
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                   cfg.TRAIN.CROP_SIZE, random_idx)   

          if item['flipped']:
              frame = frame[:, ::-1, :]  

          video[j] = frame
          j = j + 1

    else:
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        for idx in xrange(video_info[1], video_info[2]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))  # add one on frame id
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TEST.FRAME_SIZE[::-1]),
                                   cfg.TEST.CROP_SIZE, random_idx)

          if item['flipped']:
              frame = frame[:, ::-1, :]

          video[j] = frame
          j = j + 1

    while ( j < video_length):
      video[j] = frame
      j = j + 1
    processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob

def _get_blobs(video, rois = None):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'] = video
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs


def video_detect(net, video, wins=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        wins (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        wins (ndarray): R x (4*K) array of predicted bounding wins
    """
    blobs = _get_blobs(video)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:   #no use
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        wins = wins[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if not cfg.TEST.HAS_RPN:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if not cfg.TEST.HAS_RPN:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    pred_wins = blobs_out['rpn_rois_sorted']
    fc6 = blobs_out['fc6_sorted']
    pool5 = net.blobs['gt_fc6'].data

    return pred_wins, fc6, pool5

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        twin = dets[i, :2]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((twin[0], twin[1]),
                              twin[2] - twin[0],
                              twin[3] - twin[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_wins, thresh):
    """Apply non-maximum suppression to all predicted wins output by the
    test_net method.
    """
    num_classes = len(all_wins)
    num_images = len(all_wins[0])
    nms_wins = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_wins[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of wins
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_wins[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_wins

def test_net(net, roidb, LstmT2_net, LstmController_net, SentenceEmbed_net, vocab, max_per_image=100, thresh=0.05, vis=False):  
    """Test a Fast R-CNN network on an image database."""
    num_videos = len(roidb)
    # all detections are collected into:
    #    all_wins[cls][image] = N x 2 array of detections in
    #    (x1, x2, score)
    all_wins = [[[] for _ in xrange(num_videos)]
                 for _ in xrange(cfg.NUM_CLASSES)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

#    if not cfg.TEST.HAS_RPN:
#        roidb = imdb.roidb

    for i in xrange(num_videos):
        # filter out any ground truth wins
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['wins'][roidb[i]['gt_classes'] == 0]

        video = _get_video_blob(roidb[i], vocab)
        _t['im_detect'].tic()
        # scores, wins = video_detect(net, video, box_proposals) 
        wins, fc6, pool5 = video_detect(net, video, box_proposals)  ##### rewrite done!
        _t['im_detect'].toc()

        _t['misc'].tic()
        print "activity: ", 1
        # print cls_dets
        fc6_controller = pool5.reshape(1,4096)
        for d in xrange(wins.shape[0]):
            fc6_temp = fc6[d,:]
            #LstmT1_net, LstmT2_net, LstmController_net, SentenceEmbed_net            
            if d==0:
                # forward controller
                lstm_controller = LstmController_net.forward(cont_sent_controller=np.array([[0]]), embedded_input_sent_pool_controller=np.zeros((1,1,300)), fc6_controller=fc6_controller)['lstm_controller_reshape']
                sentence = generate_sentence_t2(LstmT2_net,fc6_temp.reshape(1,4096), lstm_controller)
                #print [vocabulary[index] for index in sentence_tmp]
                # sentence embed
                sentence_tmp = sentence + [-1] * (30 - len(sentence)) if len(sentence)<30 else sentence[:30]
                embedded_input_sent_pool_reshape_1 = SentenceEmbed_net.forward(input_sent_1 = np.array(sentence_tmp)[:,np.newaxis])['embedded_input_sent_pool_reshape_1']

            else:
                # forward controller
                lstm_controller = LstmController_net.forward(cont_sent_controller=np.array([[1]]), embedded_input_sent_pool_controller=embedded_input_sent_pool_reshape_1, fc6_controller=fc6_controller)['lstm_controller_reshape']
                sentence = generate_sentence_t2(LstmT2_net,fc6_temp.reshape(1,4096), lstm_controller)
                #print [vocabulary[index] for index in sentence_tmp]
                # sentence embed
                sentence_tmp = sentence + [-1] * (30 - len(sentence)) if len(sentence)<30 else sentence[:30]
                embedded_input_sent_pool_reshape_1 = SentenceEmbed_net.forward(input_sent_1 = np.array(sentence_tmp)[:,np.newaxis])['embedded_input_sent_pool_reshape_1']




            sentence_vocab = ""
            for index in sentence:
                if index != 0:
                    sentence_vocab+=vocab[index]
                    sentence_vocab+=' '
            # print sentence
            # print [vocabulary[index] for index in sentence]
            
            final_detection = []
            final_detection.append(wins[d,1])
            final_detection.append(wins[d,2])
            # print final_detection
            final_detection.append(sentence_vocab[:-1])
            print final_detection
        _t['misc'].toc()



        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_videos, _t['im_detect'].average_time,
                      _t['misc'].average_time)


