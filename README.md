# PCAA for Semantic Segmentation
This repository is for the CVPR2022 paper [Partial Class Activation Attention for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Partial_Class_Activation_Attention_for_Semantic_Segmentation_CVPR_2022_paper.pdf).

## Introduction
For the first time, this paper explores modeling pixel relation via Class Activation Map (CAM). Beyond the previous CAM generated from imagelevel classification, we present Partial CAM, which subdivides the task into region-level prediction and achieves better localization performance. We further propose Partial Class Activation Attention (PCAA) that simultaneously utilizes local and global class-level representations for attention calculation. Notably, our method achieves state-of-the-art performance on several challenging benchmarks.

<div align="center">
  <img src="figures/network.png" width="600" />
</div>

## Usage
### Requirements
python>=3.6, torch>=1.3, Pillow, opencv-python

### Backbones
We use the pre-trained backbones provided by [open-mmlab](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json), including [resnet50_v1c](https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth) and [resnet101_v1c](https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth).

### Dataset
Prepare related datasets: [Cityscapes](https://www.cityscapes-dataset.com) and [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K). Data paths should be as follows:
```
.{YOUR_CS_PATH}
├── gtFine
│   ├── train
│   ├── val
├── leftImg8bit
│   ├── train
│   ├── val

.{YOUR_ADE_PATH}
├── annotations
│   ├── training
│   ├── validation
├── images
│   ├── training
│   ├── validation
```

### Train
Multi-GPU training is required. You should have at least 4 GPUs (>= 11G) to train a model on Cityscapes. You will need 8 GPUs to train a model based on ResNet-101 on ADE20K.

E.g. To train a model on Cityscapes, modify `DATA_PATH`, `BACKBONE`, `RESTORE_PATH`, and `SAVE_DIR` in `scripts/train_cs.sh` then run:
```
sh scripts/train_cs.sh
```

### Test
Single-GPU evaluation is supported.

E.g. To evaluate a model on Cityscapes, modify the settings in `scripts/evaluate_cs.sh` and run:
```
sh scripts/evaluate_cs.sh
```

## References
This repo is mainly built based on [pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox/tree/pytorch-1.1), [DNL](https://github.com/yinmh17/DNL-Semantic-Segmentation) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks for their great work!

## Citation
If you find our codes useful, please consider to cite with:
```
@inproceedings{liu2022partial,
  title={Partial Class Activation Attention for Semantic Segmentation},
  author={Liu, Sun-Ao and Xie, Hongtao and Xu, Hai and Zhang, Yongdong and Tian, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
