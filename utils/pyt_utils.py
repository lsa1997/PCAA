# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from math import ceil
import cv2

def reduce_tensor(tensor, dst=0, norm=False, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if norm and dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor

def get_logger(prefix, output_dir, date_str):
    logger = logging.getLogger('PCAA')

    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    filename = os.path.join(output_dir, prefix + '_' + date_str + '.log')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    # only rank 0 will add a FileHandler
    if rank == 0:
        file_handler = logging.FileHandler(filename, 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    return logger

def prep_experiment(args):
    '''
    Make output directories, setup logging, snapshot code.
    '''
    log_path = args.snapshot_dir + '/log'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    args.log_path = log_path

    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    logger = get_logger('', log_path, args.date_str)

    open(os.path.join(args.snapshot_dir, args.date_str + '.txt'), 'w').write(
        str(args) + '\n\n')

    return logger

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    with torch.no_grad():
        tensor = tensor.detach()
        dist.all_reduce(tensor, op)
        if norm:
            tensor.div_(world_size)
    return tensor

def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        device = torch.device('cpu')
        checkpoint = torch.load(model_file, map_location=device)
        if 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        else: 
            state_dict = checkpoint
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logging.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logging.info(
        "Load model from {}, Time usage:\n\tIO: {}, initialize parameters: {}".format(model_file,
            t_ioend - t_start, t_end - t_ioend))

    return model

def parse_devices(input_devices):
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logging.info('using devices {}'.format(
        ', '.join([str(d) for d in devices])))

    return devices

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _dbg_interactive(var, value):
    from IPython import embed
    embed()

def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[2],0)
    cols_missing = max(target_size[1] - img.shape[3],0)
    if rows_missing > 0 or cols_missing > 0:
        padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
        return padded_img
    else:
        return img

def predict_whole(net, image, target_size, scale=1, stride=8, return_np=True, use_probs=False):
    N_, C_, H_, W_ = image.shape
    if scale != 1:
        assert N_ == 1 ,"Only batch size=1 is supported for scaling."
        image = image[0].transpose((1, 2, 0)).copy()
        dsize = int(image.shape[1] * scale), int(image.shape[0] * scale)
        align_h = int(np.ceil(dsize[1] / stride)) * stride
        align_w = int(np.ceil(dsize[0] / stride)) * stride
        scaled_img = cv2.resize(image, (align_w, align_h), interpolation = cv2.INTER_LINEAR)
        scaled_img = scaled_img.transpose((2, 0, 1))
        scaled_img = np.expand_dims(scaled_img, axis=0)
    else:
        align_h = int(np.ceil(H_ / stride)) * stride
        align_w = int(np.ceil(W_ / stride)) * stride
        if align_h != H_ or align_w != W_:
            image = image[0].transpose((1, 2, 0)).copy()
            scaled_img = cv2.resize(image, (align_w, align_h), interpolation = cv2.INTER_LINEAR)
            scaled_img = scaled_img.transpose((2, 0, 1))
            scaled_img = np.expand_dims(scaled_img, axis=0)
        else:
            scaled_img = image

    outputs = net(torch.from_numpy(scaled_img.copy()).cuda(non_blocking=True))
    if isinstance(outputs, list):
        prediction = outputs[0]
    prediction = F.interpolate(prediction, size=target_size, mode='bilinear', align_corners=False)
    if use_probs:
        prediction = F.softmax(prediction, dim=1)
    if return_np:
        return prediction.cpu().numpy().transpose(0,2,3,1)
    else:
        return prediction

def predict_sliding(net, image, tile_size, classes, target_size, scale=1, stride=8, use_probs=False):
    overlap = 1.0 / 3.0
    N, _, ori_h, ori_w = image.shape
    if scale != 1:
        assert N == 1 ,"Only batch size=1 is supported for scaling."
        scaled_img = cv2.resize(image[0].transpose((1, 2, 0)).copy(), None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)
        scaled_img = scaled_img.transpose((2, 0, 1))
        scaled_img = np.expand_dims(scaled_img, axis=0)
    else:
        scaled_img = image
    height, width = scaled_img.shape[2], scaled_img.shape[3]

    if height < tile_size[0] or width < tile_size[1]:
        prediction = predict_whole(net, image, target_size, scale, stride, True, use_probs)
        return prediction

    slide_stride_h = ceil(tile_size[0] * (1 - overlap))
    slide_stride_w = ceil(tile_size[1] * (1 - overlap))

    tile_rows = int(ceil((height - tile_size[0]) / slide_stride_h) + 1)  # strided convolution formula
    tile_cols = int(ceil((width - tile_size[1]) / slide_stride_w) + 1)

    full_probs = torch.zeros((N, classes, height, width)).cuda()
    count_predictions = torch.zeros((1, classes, height, width)).cuda()

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * slide_stride_w)
            y1 = int(row * slide_stride_h)
            x2 = min(x1 + tile_size[1], width)
            y2 = min(y1 + tile_size[0], height)
            x1 = max(int(x2 - tile_size[1]), 0)
            y1 = max(int(y2 - tile_size[0]), 0)
            
            img = scaled_img[:, :, y1:y2, x1:x2]
            prediction = predict_whole(net, img, img.shape[2:], stride=stride, return_np=False)
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    full_probs = F.interpolate(full_probs, size=target_size, mode='bilinear', align_corners=False)
    if use_probs:
        full_probs = F.softmax(full_probs, dim=1)
    full_probs = full_probs.cpu().numpy().transpose(0,2,3,1)
    return full_probs

def get_parameters(model, lr):
    wd_0 = []
    lr_1 = []
    lr_10 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'backbone' not in key:
            if value.__dict__.get('wd', -1) == 0:
                wd_0.append(value)
                print(key)
            elif 'gamma' in key or 'alpha' in key or 'beta' in key:
                wd_0.append(value)
                print(key)
            else:
                lr_10.append(value)
        else:
            lr_1.append(value)

    params = [{'params': lr_1, 'lr': lr},
                {'params': wd_0, 'lr': lr * 1.0, 'weight_decay': 0.0},
                {'params': lr_10, 'lr': lr * 1.0}]
    return params