import re
from dynamic_network_architectures.architectures.vgg import *
from base_model import ModelConstructor


def get_vgg(params):

    vgg_depth = re.findall(r"\d+", params["name"])[-1]

    if params["cifar_size"]:
        if vgg_depth == "16":
            vgg = VGG16_cifar(
                params["num_classes"], n_input_channel=params["input_channels"], input_dimension=params["input_dim"]
            )
        elif vgg_depth == "19":
            vgg = VGG19_cifar(
                params["num_classes"], n_input_channel=params["input_channels"], input_dimension=params["input_dim"]
            )

    else:
        if vgg_depth == "16":
            vgg = VGG16(
                params["num_classes"], n_input_channel=params["input_channels"], input_dimension=params["input_dim"]
            )
        elif vgg_depth == "19":
            vgg = VGG19(
                params["num_classes"], n_input_channel=params["input_channels"], input_dimension=params["input_dim"]
            )

    model = ModelConstructor(vgg, params)

    return model

