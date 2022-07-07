import argparse
import numpy as np
import sys
import json
from tqdm import tqdm

import torch
import networks
import dataset
import os
from PIL import Image as PILImage
from utils.pyt_utils import load_model, predict_sliding, predict_whole, get_confusion_matrix

from engine import Engine

DATA_DIRECTORY = '/data'
VAL_LIST_PATH = './dataset/list/context/val.txt'
COLOR_PATH = './dataset/list/context/context_colors.txt'
BATCH_SIZE = 1
INPUT_SIZE = '512,512'
BASE_SIZE = '2048,512'
RESTORE_FROM = '/model/CS_scenes_40000.pth'
SAVE_PATH = '/output/predict'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--color-dir", type=str, default=COLOR_PATH,
                        help="Path to to the directory containing the palette.")
    parser.add_argument("--test-batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument('--backbone', type=str, default='resnet101',
                    help='backbone model, can be: resnet101 (default), resnet50')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--save", type=str2bool, default='False',
                        help="save predicted image.")
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--eval-mul", action="store_true", default=False,
                        help="Whether to use multiscale inference.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="path to save results")
    parser.add_argument("--predict-mode", type=str, default='sliding',
                        help="choose predict mode.")
    parser.add_argument("--multi-grid", action="store_true", default=False,
                        help="whether to use multi-grid")
    parser.add_argument('--bin_h', default=[2], nargs='+', type=int,
                        help='Bin sizes for height.')
    parser.add_argument('--bin_w', default=[2], nargs='+', type=int,
                        help='Bin sizes for width.')
    parser.add_argument("--use-probs", action="store_true", default=False,
                        help="Whether to use probs before resize.")
    parser.add_argument("--dropout", action="store_true", default=False,
                        help="whether to use dropout for seg.")
    parser.add_argument("--resize", action="store_true",
                        help="Whether to resize the inputs.")
    return parser

def predict_multiscale(net, image, target_size, scales, flip_evaluation, args):
    net.eval()
    N_ = image.shape[0]
    H_, W_ = target_size
    full_probs = np.zeros((N_, H_, W_, args.num_classes))  
    for scale in scales:
        scale = float(scale)
        if args.predict_mode == 'sliding':
            scaled_probs = predict_sliding(net, image, args.input_size, args.num_classes, target_size, scale=scale, stride=args.os, use_probs=args.use_probs)
        else:
            scaled_probs = predict_whole(net, image, target_size, scale=scale, stride=args.os, use_probs=args.use_probs)
        if flip_evaluation == True:
            if args.predict_mode == 'sliding':
                flip_scaled_probs = predict_sliding(net, image[:,:,:,::-1].copy(), args.input_size, args.num_classes, target_size, scale=scale, stride=args.os, use_probs=args.use_probs)
            else:
                flip_scaled_probs = predict_whole(net, image[:,:,:,::-1].copy(), target_size, scale=scale, stride=args.os, use_probs=args.use_probs)
            scaled_probs = 0.5 * (scaled_probs + flip_scaled_probs[:,:,::-1])
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs

def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        bin_size = []
        for bin_h, bin_w in zip(args.bin_h, args.bin_w):
            bin_size.append((bin_h, bin_w))
        args.bin_size = bin_size
        # cudnn.benchmark = False
        h, w = map(int, args.input_size.split(','))
        args.input_size = (h, w)
        h3, w3 = map(int, args.base_size.split(','))
        args.base_size = (h3, w3)
        
        testset = eval('dataset.' + args.dataset + '.Trainset')(
            args.data_dir, args.data_list, mode='val', os=args.os, 
            crop=False, base_size=args.base_size, resize=args.resize)

        test_loader, test_sampler = engine.get_test_loader(testset)
        args.ignore_label = testset.ignore_label
        args.num_classes = testset.num_classes
        if engine.distributed:
            test_sampler.set_epoch(0)
        seg_model = eval('networks.' + args.model + '.Seg_Model')(
            num_classes=args.num_classes,backbone=args.backbone, 
            multi_grid=args.multi_grid, bins=args.bin_size, dropout=args.dropout
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        load_model(model, args.restore_from)

        model.eval()
        confusion_matrix = np.zeros((args.num_classes,args.num_classes))

        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        palette = np.loadtxt(args.color_dir).astype('uint8')

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(test_loader)
        if args.eval_mul:
            scales = [0.5,0.75,1.0,1.25,1.5,1.75]
            flip = True
        else:
            scales= [1.0]
            flip = False
        for idx in pbar:
            image, label, name = dataloader.next()
            target_size = (label.shape[1],label.shape[2])
            with torch.no_grad():
                output = predict_multiscale(model, image.numpy(), target_size, scales, flip, args)

            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            seg_gt = np.asarray(label.numpy(), dtype=np.int)

            if args.save:
                for i in range(image.size(0)): 
                    output_im = PILImage.fromarray(seg_pred[i])
                    output_im.putpalette(palette)
                    output_im.save(os.path.join(save_path, name[i]+'.png'))
        
            ignore_index = seg_gt != args.ignore_label
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]
            confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

            print_str = ' Iter{}/{}'.format(idx + 1, len(test_loader))
            pbar.set_description(print_str, refresh=False)
        np.save(os.path.join(save_path, 'cmatrix.npy'), confusion_matrix)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        acc_cls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)   

        print({'meanIU':mean_IU, 'IU_array':IU_array,'allAcc':acc, 'mAcc':acc_cls})
        with open(save_path + '/result.txt', 'w') as f:
            f.write(json.dumps({'meanIU':mean_IU, 'IU_array':IU_array.tolist(),'allAcc':acc, 'mAcc':acc_cls}))

if __name__ == '__main__':
    main()
