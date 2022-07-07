from .resnet import ResNetStem, Bottleneck
from utils.pyt_utils import load_model

def get_backbone(norm_layer, pretrained_model=None, backbone='resnet101', **kwargs):
    if backbone == 'resnet50':
        model = ResNetStem(Bottleneck,[3, 4, 6, 3], norm_layer=norm_layer, **kwargs)
        print('Backbone:resnet50stem')
    elif backbone == 'resnet101':
        model = ResNetStem(Bottleneck,[3, 4, 23, 3], norm_layer=norm_layer, **kwargs)
        print('Backbone:resnet101stem')
    else:
        raise RuntimeError('unknown backbone: {}'.format(backbone))
    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
