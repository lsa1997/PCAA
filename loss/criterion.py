import torch.nn as nn
import torch
from torch.nn import functional as F
from .OhemCrossEntropy import OhemCrossEntropy2d

class DSNLoss(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduction='mean', weight=None, ohem=False, thresh=0.7, min_kept=100000):
        super(DSNLoss, self).__init__()
        self.ignore_index = ignore_index
        if ohem:
            self.seg_criterion = OhemCrossEntropy2d(ignore_index, thresh, min_kept, weight=weight)
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=False)
            loss1 = self.seg_criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=False)
            loss2 = self.aux_criterion(scale_pred, target)

            total_loss = loss1 + loss2*0.4
            loss_dict = {'seg_loss':loss1, 'aux_loss':loss2, 'total_loss':total_loss}
            return loss_dict
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=False)
            loss = self.seg_criterion(scale_pred, target)
            loss_dict = {'total_loss':loss}
            return loss_dict

class JointClsLoss(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, reduction='mean', weight=None, bins=(1,2,4), ohem=False, thresh=0.7, min_kept=100000):
        super(JointClsLoss, self).__init__()
        self.ignore_index = ignore_index
        if ohem:
            self.seg_criterion = OhemCrossEntropy2d(ignore_index, thresh, min_kept, weight=weight)
        else:
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = FocalLoss(ignore_index=ignore_index)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label_onehot, bin_size, th=0.01):
        cls_percentage = F.adaptive_avg_pool2d(label_onehot, bin_size)
        cls_label = torch.where(cls_percentage>0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))
        cls_label[(cls_percentage<th)&(cls_percentage>0)] = self.ignore_index
        return cls_label

    def forward(self, preds, target, target_onehot):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=False)
        seg_loss = self.seg_criterion(scale_pred, target)
        if len(preds) == 3:
            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=False)
            dsn_loss = self.dsn_criterion(scale_pred, target)

            cls_list = preds[2]
            cls_loss = 0
            for cls_pred, bin_size in zip(cls_list, self.bins):
                cls_gt = self.get_bin_label(target_onehot, bin_size)
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_gt) / len(self.bins)

            total_loss = seg_loss + 0.4*dsn_loss + self.cls_weight*cls_loss
            loss_dict = {'seg_loss':seg_loss, 'aux_loss':dsn_loss, 'cls_loss':cls_loss, 'total_loss':total_loss}
            return loss_dict      
        else:
            cls_list = preds[1]
            cls_loss = 0
            for cls_pred, bin_size in zip(cls_list, self.bins):
                cls_gt = self.get_bin_label(target_onehot, bin_size)
                cls_loss = cls_loss + self.cls_criterion(cls_pred, cls_gt) / len(self.bins)

            total_loss = seg_loss + self.cls_weight*cls_loss
            loss_dict = {'seg_loss':seg_loss, 'cls_loss':cls_loss, 'total_loss':total_loss}
            return loss_dict

class FocalLoss(nn.Module):
    ''' focal loss '''
    def __init__(self, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.crit = nn.BCELoss(reduction='none')

    def binary_focal_loss(self, input, target, valid_mask):
        input = input[valid_mask]
        target = target[valid_mask]
        pt = torch.where(target == 1, input, 1 - input)
        ce_loss = self.crit(input, target)
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        loss = loss.mean()
        return loss
        
    def	forward(self, input, target):
        valid_mask = (target != self.ignore_index)
        K = target.shape[1]
        total_loss = 0
        for i in range(K):
            total_loss += self.binary_focal_loss(input[:,i], target[:,i], valid_mask[:,i])
        return total_loss / K