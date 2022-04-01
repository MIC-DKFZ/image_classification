import re
from dynamic_network_architectures.architectures.resnet import *
from base_model import ModelConstructor


def get_resnet(params):

    resnet_depth = re.findall(r'\d+', params['name'])[-1]

    if params['small_imgs']:
        if resnet_depth == '18':
            if params['bottleneck']:
                raise NotImplementedError
            else:
                resnet = ResNet18_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '34':
            if params['bottleneck']:
                raise NotImplementedError
            else:
                resnet = ResNet34_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '50':
            if params['bottleneck']:
                resnet = ResNet50bn_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
            else:
                resnet = ResNet50_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '152':
            if params['bottleneck']:
                resnet = ResNet152bn_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
            else:
                resnet = ResNet152_CIFAR(params['num_classes'], dropout=params['resnet_dropout'])
    else:
        if resnet_depth == '18':
            if params['bottleneck']:
                raise NotImplementedError
            else:
                resnet = ResNet18(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '34':
            if params['bottleneck']:
                raise NotImplementedError
            else:
                resnet = ResNet34(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '50':
            if params['bottleneck']:
                resnet = ResNet50bn(params['num_classes'], dropout=params['resnet_dropout'])
            else:
                resnet = ResNet50(params['num_classes'], dropout=params['resnet_dropout'])
        elif resnet_depth == '152':
            if params['bottleneck']:
                resnet = ResNet152bn(params['num_classes'], dropout=params['resnet_dropout'])
            else:
                resnet = ResNet152(params['num_classes'], dropout=params['resnet_dropout'])

    model = ModelConstructor(resnet, params)

    return model


