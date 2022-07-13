import numpy as np
from numpy import random
import cv2
from torch.utils import data

class BaseDataset(data.Dataset):
    def __init__(self, mode='train', os=8, crop_size=(512, 512), 
            ignore_label=255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], base_size=(512, 512)):
        self.crop_size = crop_size

        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.os = os
        
        self.files = []
        self.mode = mode

        self.base_size = base_size
        self.ratio_range = (0.5, 2.0)
        self.cat_max_ratio = 0.75
        self.brightness_delta = 32
        self.contrast_lower = 0.5
        self.contrast_upper = 1.5
        self.saturation_lower = 0.5
        self.saturation_upper = 1.5
        self.hue_delta = 18
        
    def __len__(self):
        return len(self.files)

    def normalize(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def resize(self, image, label=None, random_scale=False):
        if random_scale:
            min_ratio, max_ratio = self.ratio_range
            f_scale = random.random_sample() * (max_ratio - min_ratio) + min_ratio
            output_size = int(self.base_size[0]*f_scale), int(self.base_size[1]*f_scale)
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
        else:
            output_size = self.base_size[0], self.base_size[1]
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
        image = cv2.resize(image, dsize, interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, dsize, interpolation = cv2.INTER_NEAREST)
            return image, label
        else:
            return image

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
    def brightness(self, img):
        if random.randint(2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img
    def contrast(self, img):
        if random.randint(2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img
    def bgr2hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def hsv2bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    def saturation(self, img):
        if random.randint(2):
            img = self.bgr2hsv(img)
            img[:, :, 1] = self.convert(img[:, :, 1], 
                    alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = self.hsv2bgr(img)
        return img
    def hue(self, img):
        if random.randint(2):
            img = self.bgr2hsv(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) +
                    random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = self.hsv2bgr(img)
        return img
    def random_distortion(self, img):
        img = self.brightness(img)
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        return img

    def pad(self, output_size, image, label=None):
        pad_h = max(output_size[0] - image.shape[0],0)
        pad_w = max(output_size[1] - image.shape[1],0)
        if pad_h > 0 or pad_w > 0:
            image_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            if label is not None:
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(self.ignore_label,))
        else:
            image_pad, label_pad = image, label
        if label is not None:
            return image_pad, label_pad
        else:
            return image_pad

    def random_flip(self, image, label):
        if random.randint(2):
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        return image, label

    def crop(self, image, label):
        img_h, img_w = label.shape
        crop_h, crop_w = self.crop_size
        margin_h = max(img_h - crop_h, 0)
        margin_w = max(img_w - crop_w, 0)
        if self.mode == 'train':
            h_off = np.random.randint(0, margin_h + 1)
            w_off = np.random.randint(0, margin_w + 1)
            if self.cat_max_ratio < 1.:
                for _ in range(10):
                    seg_temp = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_label]
                    if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                        break
                    h_off = np.random.randint(0, margin_h + 1)
                    w_off = np.random.randint(0, margin_w + 1)
        else:
            h_off = int(round(margin_h / 2.))
            w_off = int(round(margin_w / 2.))
        image = image[h_off : h_off+crop_h, w_off : w_off+crop_w]
        label = label[h_off : h_off+crop_h, w_off : w_off+crop_w]
        return image, label

    def mask_to_onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector

        """
        _mask = [(mask == i) for i in range(num_classes)]
        mask_onehot = np.array(_mask).astype(np.float32)
        return mask_onehot