#!/bin/bash
uname -a
#date
#env
date

DATASET=ade20k
DATA_PATH=/home/datasets/ADEChallengeData2016
LIST_PATH=./dataset/list/ade20k/validation.odgt
COLOR_PATH=./dataset/list/ade20k/ade20k_colors.txt
MODEL=caanet
BACKBONE=resnet101
RESTORE_PATH=/home/outputs/caanet_ade_r101/CS_scenes_best.pth
BS=1
INPUT_SIZE=512,512
BASE_SIZE=2048,512
SAVE=0
SAVE_DIR=/home/outputs/caanet_ade_r101/result_whole
PRED_MODE=whole

cd /home/PCAA
python evaluate.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --data-list ${LIST_PATH} --color-dir ${COLOR_PATH} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} --test-batch-size ${BS} \
                --save-path ${SAVE_DIR} --save ${SAVE} --predict-mode ${PRED_MODE} \
                --bin_h 4  --bin_w 4 --os 32 \
                --multi-grid