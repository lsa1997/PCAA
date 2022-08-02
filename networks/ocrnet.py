# Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/models/nets/ocrnet.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from .backbones import get_backbone

class Seg_Model(nn.Module):
    def __init__(self, num_classes, criterion=None, norm_layer=nn.BatchNorm2d, aux_loss=True, dropout=False, **kwargs):
        super(Seg_Model, self).__init__()
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True)
            )
        self.spatial_context_head = SpatialGather_Module()
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, key_channels=256, 
                                                  out_channels=512, norm_layer=norm_layer)
        cls_convs = []
        if dropout:
            cls_convs.append(nn.Dropout2d(0.1))
        cls_convs.append(nn.Conv2d(512, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*cls_convs)
        self.use_dsn = aux_loss
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

        x_dsn = self.dsn(x_dsn)
        x = self.conv_3x3(x)
        context = self.spatial_context_head(x, x_dsn)
        x = self.spatial_ocr_head(x, context)
        x = self.classifier(x)
        outs = [x, x_dsn]
        
        if self.criterion is not None and labels is not None:       
            return self.criterion(outs, labels)
        else:
            return outs

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self):
        super(SpatialGather_Module, self).__init__()

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c -> B x c x k x 1
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 norm_layer=nn.BatchNorm2d):
        super(ObjectAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            norm_layer(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            norm_layer(key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            norm_layer(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, bias=False),
            norm_layer(key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            norm_layer(key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # [B, N, C]

        key = self.f_object(proxy).view(batch_size, self.key_channels, -1) # [B, C, K]

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1) # [B, K, C]

        sim_map = torch.matmul(query, key) # [B, N, K]
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
 
        return context

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 norm_layer=nn.BatchNorm2d):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, 
                                                           key_channels,
                                                           norm_layer=norm_layer)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output