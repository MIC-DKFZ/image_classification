import re

from dynamic_network_architectures.architectures.vgg import VGG16 as dyn_VGG16
from dynamic_network_architectures.architectures.vgg import VGG19 as dyn_VGG19
from dynamic_network_architectures.architectures.vgg import (
    VGG16_cifar as dyn_VGG16_cifar,
)
from dynamic_network_architectures.architectures.vgg import (
    VGG19_cifar as dyn_VGG19_cifar,
)

from base_model import BaseModel


def get_vgg_args(params):
    vgg_args = {
        "n_classes": params["num_classes"],
        "n_input_channel": params["input_channels"],
        "input_dimension": params["input_dim"],
    }

    return vgg_args


class VGG16(BaseModel):
    def __init__(self, cifar_size, **hypparams):
        super(VGG16, self).__init__(**hypparams)
        vgg_args = get_vgg_args(hypparams)

        self.model = dyn_VGG16_cifar(**vgg_args) if cifar_size else dyn_VGG16(**vgg_args)

    def forward(self, x):
        out = self.model(x)
        return out


class VGG19(BaseModel):
    def __init__(self, cifar_size, **hypparams):
        super(VGG19, self).__init__(**hypparams)
        vgg_args = get_vgg_args(hypparams)

        self.model = dyn_VGG19_cifar(**vgg_args) if cifar_size else dyn_VGG19(**vgg_args)

    def forward(self, x):
        out = self.model(x)
        return out
