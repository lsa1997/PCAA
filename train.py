import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import networks
import dataset

import random
from utils.pyt_utils import prep_experiment, load_model, predict_sliding, predict_whole, get_confusion_matrix, get_parameters
from loss import get_loss
from engine import Engine

BATCH_SIZE = 8
DATA_DIRECTORY = '/data'
DATA_LIST_PATH = './dataset/list/context/train.txt'
VAL_LIST_PATH = './dataset/list/context/val.txt'
INPUT_SIZE = '512,512'
TEST_SIZE = '512,512'
BASE_SIZE = '2048,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_STEPS = 60000
POWER = 0.9
RANDOM_SEED = 0
RESTORE_FROM = '/model/resnet_backbone/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = '/output'
WEIGHT_DECAY = 0.0001

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
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='Dataset for training')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--test-batch-size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the file listing the images in the val set.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--test-size", type=str, default=TEST_SIZE,
                        help="Comma-separated string with height and width of images for validation.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--print-frequency", type=int, default=100,
                        help="Number of training steps.") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-flip", action="store_true", default=False,
                        help="Whether to randomly flip the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--base-size", type=str, default=BASE_SIZE,
                        help="Base size of images for resize.")
    parser.add_argument("--random-distort", action="store_true", default=False,
                        help="Whether to randomly distort the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--predict-mode", type=str, default='sliding',
                        help="choose predict mode.")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="choose the number of workers.")
    parser.add_argument('--backbone', type=str, default='resnet101',
                    help='backbone model, can be: resnet101 (default), resnet50')
    parser.add_argument("--ohem", type=str2bool, default='False',
                        help="use hard negative mining")
    parser.add_argument("--ohem-thres", type=float, default=0.7,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--ohem-keep", type=int, default=100000,
                        help="choose the samples with correct probability underthe threshold.")
    parser.add_argument("--os", type=int, default=8, help="output stride")
    parser.add_argument("--multi-grid", action="store_true", default=False,
                        help="whether to use multi-grid")
    parser.add_argument('--bin_h', default=[2,4], nargs='+', type=int,
                        help='Bin sizes for height.')
    parser.add_argument('--bin_w', default=[2,4], nargs='+', type=int,
                        help='Bin sizes for width.')
    parser.add_argument("--onehot", action="store_true", default=False,
                        help="calculate edge from onehot")
    parser.add_argument("--use-apex", action="store_true", default=False,
                        help="whether to use apex for training.")
    parser.add_argument("--aux-loss", action="store_true", default=False,
                        help="whether to auxiliary loss for training.")
    parser.add_argument("--dropout", action="store_true", default=False,
                        help="whether to use dropout for seg.")
    return parser

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    for index in range(len(optimizer.param_groups)):
        optimizer.param_groups[index]['lr'] = lr
    return lr

def validate(model, dataloader, args):
    model.eval()
    confusion_matrix = np.zeros((args.num_classes,args.num_classes))
    for i, data in enumerate(dataloader):
        image, label, name = data
        with torch.no_grad():
            target_size = (label.shape[1],label.shape[2])
            if args.predict_mode == 'sliding':
                output = predict_sliding(model, image.numpy(), args.input_size, args.num_classes, target_size)
            else:
                output = predict_whole(model, image.numpy(), target_size)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_gt = np.asarray(label.numpy(), dtype=np.int)
    
        ignore_index = seg_gt != args.ignore_label
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)
    return confusion_matrix

def main():
    """Create the model and start the training."""
    parser = get_parser()
    torch_ver = torch.__version__[:3]

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        bin_size = []
        for bin_h, bin_w in zip(args.bin_h, args.bin_w):
            bin_size.append((bin_h, bin_w))
        args.bin_size = bin_size

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger = prep_experiment(args)

        cudnn.benchmark = True
        seed = args.random_seed
        if seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        # data loader
        h1, w1 = map(int, args.input_size.split(','))
        args.input_size = (h1, w1)
        h2, w2 = map(int, args.test_size.split(','))
        args.test_size = (h2, w2)
        h3, w3 = map(int, args.base_size.split(','))
        args.base_size = (h3, w3)

        trainset = eval('dataset.' + args.dataset + '.Trainset')(
            args.data_dir, args.train_list, os=args.os, 
            crop=True, crop_size=args.input_size, 
            scale=args.random_scale, base_size=args.base_size, 
            flip=args.random_flip, distort=args.random_distort, 
            calc_onehot=args.onehot)
        testset = eval('dataset.' + args.dataset + '.Trainset')(
            args.data_dir, args.val_list, mode='val', os=args.os, 
            crop=True, crop_size=args.test_size, base_size=args.base_size, resize=True)
     
        train_loader, train_sampler = engine.get_train_loader(trainset)
        test_loader, test_sampler = engine.get_test_loader(testset)
        args.ignore_label = trainset.ignore_label
        args.num_classes = trainset.num_classes
        args.class_weights = trainset.class_weights
        if engine.distributed:
            test_sampler.set_epoch(0)

        criterion = get_loss(args)
        if engine.distributed:
            if args.use_apex:
                import apex
                BatchNorm = apex.parallel.SyncBatchNorm
            else:
                BatchNorm = nn.SyncBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        if args.start_iters > 0:
            seg_model = eval('networks.' + args.model + '.Seg_Model')(
                num_classes=args.num_classes, criterion=criterion,
                backbone=args.backbone, norm_layer=BatchNorm,
                multi_grid=args.multi_grid, bins=args.bin_size, 
                aux_loss=args.aux_loss, dropout=args.dropout)
        else:
            seg_model = eval('networks.' + args.model + '.Seg_Model')(
                num_classes=args.num_classes, criterion=criterion,
                backbone=args.backbone, pretrained_model=args.restore_from, norm_layer=BatchNorm,
                multi_grid=args.multi_grid, bins=args.bin_size, 
                aux_loss=args.aux_loss, dropout=args.dropout)
            
        params = get_parameters(seg_model, args.learning_rate)
        optimizer = optim.SGD(params, lr=args.learning_rate, 
            momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            print(seg_model)

        if engine.distributed:
            if args.use_apex:
                model, optimizer = apex.amp.initialize(seg_model.cuda(), optimizer, opt_level='O0')
                model = apex.parallel.DistributedDataParallel(model)
            else:
                model = engine.data_parallel(seg_model.cuda())
        else:
            model = torch.nn.DataParallel(seg_model.cuda())

        if args.start_iters > 0:
            load_model(model, args.restore_from)

        if not os.path.exists(args.snapshot_dir) and engine.local_rank == 0:
            os.makedirs(args.snapshot_dir)
        best_miou = 0
        run = True
        global_iteration = args.start_iters
        loss_dict_memory = {}
        while run:
            epoch = global_iteration // len(train_loader)
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            model.train()
            for i, data in enumerate(train_loader):
                global_iteration += 1
                if args.onehot:
                    images, labels, labels_onehot, _ = data
                    labels_onehot = labels_onehot.float().cuda(non_blocking=True)
                else:
                    images, labels, _ = data
                
                images = images.cuda(non_blocking=True)
                labels = labels.long().cuda(non_blocking=True)
                
                lr = adjust_learning_rate(optimizer, args.learning_rate, global_iteration-1, args.num_steps, args.power)
                if args.onehot:
                    loss_dict = model(images, labels, labels_onehot)
                else:
                    loss_dict = model(images, labels)

                total_loss = loss_dict['total_loss']
                optimizer.zero_grad()
                if args.use_apex:
                    with apex.amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()
                optimizer.step()

                for loss_name, loss in loss_dict.items():
                    loss_dict_memory[loss_name] = engine.all_reduce_tensor(loss).item()

                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    if i % args.print_frequency == 0:
                        print_str = 'Epoch{}/Iters{}'.format(epoch, global_iteration) \
                            + ' Iter{}/{}:'.format(i + 1, len(train_loader)) \
                            + ' lr=%.2e' % lr 
                        for loss_name, loss_value in loss_dict_memory.items():
                            print_str += ' %s=%.4f' % (loss_name, loss_value)
                        logger.info(print_str)
                        
                    if global_iteration % args.save_pred_every == 0 or global_iteration >= args.num_steps:
                        print('taking snapshot ...')
                        if torch_ver < '1.6':
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(global_iteration)+'.pth'))
                        else:
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_'+str(global_iteration)+'.pth'), _use_new_zipfile_serialization=False)
                                
                if global_iteration >= args.num_steps:
                    run = False
                    break

            if global_iteration >= args.num_steps - 10000:
                confusion_matrix = validate(model, test_loader,args)
                confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
                confusion_matrix = engine.all_reduce_tensor(confusion_matrix, norm=False).cpu().numpy()
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)

                IU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IU = IU_array.mean()
                
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    logger.info({'meanIU':mean_IU, 'IU_array':IU_array})
                    if mean_IU >= best_miou:
                        print('taking snapshot ...')
                        if torch_ver < '1.6':
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_best.pth'))
                        else:
                            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'CS_scenes_best.pth'), _use_new_zipfile_serialization=False)
                        best_miou = mean_IU
        
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            logger.info({'best_IU':best_miou})


if __name__ == '__main__':
    main()