# --------------------------------------------------------
# JEDDi-Net
# Copyright (c) 2018 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------


#Export CUDA_HOME=/usr/local/cuda-7.5
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

#export CUDA_VISIBLE_DEVICES=2



export PYTHONUNBUFFERED=true

GPU_ID=0
EX_DIR=denseCap_jeddiNet_end2end

LOG="experiments/${EX_DIR}/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"


time python ./experiments/${EX_DIR}/train_net.py --gpu ${GPU_ID} \
  --solver ./experiments/${EX_DIR}/solver.prototxt \
  --weights ./pretrain/lstm_pretrain_iter_110000.caffemodel \
  --Detectionweights ./pretrain/proposal_pretrain_iter_10000.caffemodel \
  --cfg ./experiments/${EX_DIR}/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG



