# Modified from https://github.com/speedinghzl/pytorch-segmentation-toolbox/blob/master/networks/deeplabv3.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbones import get_backbone

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36), norm_layer=nn.BatchNorm2d):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm_layer(inner_features),
                                   nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=False)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        out = self.bottleneck(out)
        return out

class Seg_Model(nn.Module):
    def __init__(self, num_classes, criterion=None, norm_layer=nn.BatchNorm2d, aux_loss=False, dropout=False, **kwargs):
        super(Seg_Model, self).__init__()
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.head = ASPPModule(2048, norm_layer=norm_layer)
        cls_convs = []
        if dropout:
            cls_convs.append(nn.Dropout2d(0.1))
        cls_convs.append(nn.Conv2d(512, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*cls_convs)
        self.use_dsn = aux_loss
        if self.use_dsn:
            dsn_convs = [
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True)
                ]
            if dropout:
                dsn_convs.append(nn.Dropout2d(0.1))
            dsn_convs.append(nn.Conv2d(256, num_classes, kernel_size=1))
            self.dsn = nn.Sequential(*dsn_convs)
        self.criterion = criterion

    def forward(self, x, labels=None):
        base_outs = self.backbone(x)
        if len(base_outs) == 1:
            x_dsn = base_outs[0]
            x = base_outs[0]
        else:
            x_dsn = base_outs[-2]
            x = base_outs[-1]
        x = self.head(x)
        x = self.classifier(x)

        if self.criterion is not None and labels is not None:
            if self.use_dsn:
                x_dsn = self.dsn(x_dsn)
                outs = [x, x_dsn]
            else:
                outs = [x]
            return self.criterion(outs, labels)
        else:
            outs = [x]
            return outs