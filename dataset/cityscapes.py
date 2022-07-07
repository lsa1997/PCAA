import os.path as osp
import cv2
from .base_dataset import BaseDataset

class Trainset(BaseDataset):
    num_classes = 19
    def __init__(self, root, list_path, mode='train', os=8, crop_size=(768, 768), 
            crop=False, flip=False, distort=False, scale=False,
            ignore_label=255, base_size=(2048,1024), calc_onehot=False, resize=False):
        super(Trainset, self).__init__(mode, os, crop_size, ignore_label, base_size=base_size)
        self.root = root
        self.list_path = list_path

        self.is_resize = resize if mode == 'val' else True
        self.is_flip = flip
        self.is_scale = scale
        self.is_distort = distort
        self.is_crop = crop

        self.calc_onehot = calc_onehot
        self.os = os
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 
        #                                 1.0865, 1.1529, 1.0507])
        self.class_weights = None
        print('{} images are loaded!'.format(len(self.img_ids)))

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)
        name = datafiles["name"]
        
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
