# Modified from https://github.com/yinmh17/DNL-Semantic-Segmentation/blob/master/model/seg/nets/nonlocalbn.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from .backbones import get_backbone

class NonLocal2d(nn.Module):
    def __init__(self, inplanes, planes, use_out, norm_layer):
        super(NonLocal2d, self).__init__()
        if use_out:
            self.conv_value = nn.Conv2d(inplanes, planes, kernel_size=1)
            self.conv_out = nn.Sequential(
                nn.Conv2d(planes, inplanes, kernel_size=1, bias=False),
                norm_layer(inplanes),
                nn.ReLU(inplace=True)
                )
        else:
            self.conv_value = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None

        self.conv_query = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv_key = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.scale = math.sqrt(planes)

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x

        value = self.conv_value(x)
        value = value.view(value.size(0), value.size(1), -1)

        out_sim = None
        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(x)
        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
            
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        out = out_sim + residual

        return out

class NLModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, norm_layer=nn.BatchNorm2d, dropout=False):
        super(NLModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))

        self.ctb = NonLocal2d(inter_channels, inter_channels // 2,
                                   use_out=True,
                                   norm_layer=norm_layer)

        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(inplace=True))
        bottleneck_convs = [
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            bottleneck_convs.append(nn.Dropout2d(0.1))
        bottleneck_convs.append(nn.Conv2d(out_channels, num_classes, kernel_size=1))
        self.bottleneck = nn.Sequential(*bottleneck_convs)

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        if self.ctb is not None:
            for i in range(recurrence):
                output = self.ctb(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class Seg_Model(nn.Module):
    def __init__(self, num_classes, criterion=None, norm_layer=nn.BatchNorm2d, aux_loss=True, dropout=False, **kwargs):
        super(Seg_Model, self).__init__()
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.head = NLModule(2048, 512, num_classes, norm_layer=norm_layer, dropout=dropout)
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
