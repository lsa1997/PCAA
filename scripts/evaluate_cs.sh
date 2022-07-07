#!/bin/bash
uname -a
#date
#env
date

DATASET=cityscapes
DATA_PATH=/home/datasets/cityscapes
LIST_PATH=./dataset/list/cityscapes/val.lst
COLOR_PATH=./dataset/list/cityscapes/cs_colors.txt
MODEL=caanet
BACKBONE=resnet101
RESTORE_PATH=/home/outputs/caanet_cs_r50/CS_scenes_best.pth
BS=1
INPUT_SIZE=768,768
BASE_SIZE=2048,1024
SAVE=1
SAVE_DIR=/home/outputs/caanet_cs_r50/result_slide
PRED_MODE=sliding

cd /home/PCAA
python evaluate.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --data-list ${LIST_PATH} --color-dir ${COLOR_PATH} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} --test-batch-size ${BS} \
                --save-path ${SAVE_DIR} --save ${SAVE} --predict-mode ${PRED_MODE} \
                --bin_h 4 --bin_w 4  --os 32