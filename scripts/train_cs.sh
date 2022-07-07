#!/bin/bash
uname -a
#date
#env
date

DATASET=cityscapes
DATA_PATH=/home/datasets/cityscapes
TRAIN_LIST=./dataset/list/cityscapes/train.lst
VAL_LIST=./dataset/list/cityscapes/val.lst
MODEL=caanet
BACKBONE=resnet50
RESTORE_PATH=/home/backbones/resnet50_v1c.pth
LR=1e-2
WD=1e-4
BS=8
BS_TEST=8
START=0
STEPS=60000
BASE_SIZE=2048,1024
INPUT_SIZE=768,768
TEST_SIZE=1024,2048
OHEM=0
OS=8
SAVE_DIR=/home/outputs/caanet_cs_r50
PRED_MODE=sliding

cd /home/PCAA
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --num-workers 8\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--random-flip --random-scale --random-distort\
			--input-size ${INPUT_SIZE} --test-size ${TEST_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-iters ${START} --num-steps ${STEPS} \
			--ohem ${OHEM} --os ${OS} --snapshot-dir ${SAVE_DIR} \
			--predict-mode ${PRED_MODE} \
           	--onehot --bin_h 4 --bin_w 4 --aux-loss