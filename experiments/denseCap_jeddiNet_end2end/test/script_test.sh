#!/bin/bash

# --------------------------------------------------------
# JEDDi-Net
# Copyright (c) 2018 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------


GPU_ID=1
EX_DIR=denseCap_jeddiNet_end2end

export PYTHONUNBUFFERED=true

for (( i=150; i<=150; i+=10 )); do

LOG="experiments/${EX_DIR}/test/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"


time python ./experiments/${EX_DIR}/test/test_net.py --gpu ${GPU_ID} \
  --def ./experiments/${EX_DIR}/test/test_rpn.prototxt \
  --defLSTM ./experiments/${EX_DIR}/test/test_lstm.prototxt \
  --defLstmController ./experiments/${EX_DIR}/test/lstm_controller.deploy.prototxt \
  --defSentenceEmbed ./experiments/${EX_DIR}/test/sentence_embedding.deploy.prototxt \
  --net ./experiments/${EX_DIR}/snapshot/activitynet_iter_${i}000.caffemodel \
  --netLSTM ./experiments/${EX_DIR}/snapshot/activitynet_iter_${i}000.caffemodel \
  --cfg ./experiments/${EX_DIR}/test/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG

done




