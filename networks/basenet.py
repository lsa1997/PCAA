import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbones import get_backbone

class Seg_Model(nn.Module):
    def __init__(self, num_classes, criterion=None, norm_layer=nn.BatchNorm2d, aux_loss=True, dropout=False, **kwargs):
        super(Seg_Model, self).__init__()
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)

        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True)
            )
        cls_convs = []
        if dropout:
            cls_convs.append(nn.Dropout2d(0.1))
        cls_convs.append(nn.Conv2d(512, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*cls_convs)

        self.use_dsn = aux_loss
        if self.use_dsn:
            dsn_convs = [
                nn.Conv2d(self.backbone.dsn_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
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