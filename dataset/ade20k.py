import os.path as osp
import numpy as np
import cv2
import json
from .base_dataset import BaseDataset

class Trainset(BaseDataset):
    num_classes = 150
    def __init__(self, root, list_path, mode='train', os=8, crop_size=(512, 512), 
             crop=False, flip=False, distort=False, scale=False,
             ignore_label=255, base_size=(2048,512), calc_onehot=False, resize=False):
        super(Trainset, self).__init__(mode, os, crop_size, ignore_label, base_size=base_size)
        self.root = root
        self.list_path = list_path

        self.is_resize = resize if mode == 'val' else True
        self.is_flip = flip
        self.is_scale = scale
        self.is_distort = distort
        self.is_crop = crop

        self.calc_onehot = calc_onehot
        self.list_sample = [json.loads(x.rstrip()) for x in open(list_path, 'r')]
        for item in self.list_sample:
            image_path, label_path = item["fpath_img"], item["fpath_segm"]
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.class_weights = None
        print('{} images are loaded!'.format(len(self.list_sample)))

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        name = datafiles["name"]
        label = np.array(label, dtype=np.int32)
        label = label - 1
        label[label < 0] = self.ignore_label

        if self.is_resize:
            image, label = self.resize(image, label, random_scale=self.is_scale)
        if self.is_crop:
            image, label = self.crop(image, label)
        if self.is_flip:
            image, label = self.random_flip(image, label)
        if self.is_distort:
            image = self.random_distortion(image)

        image = self.normalize(image)
        if self.is_crop:
            image, label = self.pad(self.crop_size, image, label)
        image = image.transpose((2, 0, 1)) # [H, W, C] -> [C, H, W]
            
        if self.calc_onehot:
            label_onehot = self.mask_to_onehot(label, self.num_classes)
            return image.copy(), label.copy(), label_onehot.copy(), name
        else:
            return image.copy(), label.copy(), name

