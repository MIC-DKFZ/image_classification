from efficientnet_pytorch import EfficientNet
from base_model import BaseModel


class BaseEfficientNet(BaseModel):

    def __init__(self, num_classes=10, hypparams={}, net_type='l2'):
        super(BaseEfficientNet, self).__init__(hypparams)

        # EfficientNet
        self.network = EfficientNet.from_name("efficientnet-"+net_type, image_size=32, num_classes=num_classes)

    def forward(self, x):
        out = self.network(x)
        return out


def EfficientNetL2(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'l2')

def EfficientNetB8(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b8')

def EfficientNetB7(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b7')

def EfficientNetB6(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b6')

def EfficientNetB5(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b5')

def EfficientNetB4(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b4')

def EfficientNetB3(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b3')

def EfficientNetB2(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b2')

def EfficientNetB1(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b1')

def EfficientNetB0(num_classes=10, hypparams={}):
    return BaseEfficientNet(num_classes, hypparams, 'b0')

