#!/bin/bash
uname -a
#date
#env
date

DATASET=ade20k
DATA_PATH=/home/datasets/ADEChallengeData2016
TRAIN_LIST=./dataset/list/ade20k/training.odgt
VAL_LIST=./dataset/list/ade20k/validation.odgt
MODEL=caanet
BACKBONE=resnet101
RESTORE_PATH=/home/backbones/resnet101_v1c.pth
LR=1e-2
WD=1e-4
BS=16
BS_TEST=16
START=0
STEPS=150000
BASE_SIZE=2048,512
INPUT_SIZE=512,512
TEST_SIZE=512,512
OHEM=0
OS=8
SAVE_DIR=/home/outputs/caanet_ade_r101
PRED_MODE=whole

cd /home/PCAA
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST}  --num-workers 16\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--random-flip --random-scale --random-distort \
			--input-size ${INPUT_SIZE} --test-size ${TEST_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-iters ${START} --num-steps ${STEPS} \
			--ohem ${OHEM} --os ${OS} --snapshot-dir ${SAVE_DIR} \
			--predict-mode ${PRED_MODE} --multi-grid \
			--onehot --bin_h 4 --bin_w 4 --aux-loss 