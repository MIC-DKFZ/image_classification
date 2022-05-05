import re
from dynamic_network_architectures.architectures.resnet import *
from base_model import ModelConstructor


def get_resnet(params):

    resnet_depth = re.findall(r"\d+", params["name"])[-1]

    if params["cifar_size"]:
        if resnet_depth == "18":
            if params["bottleneck"]:
                raise NotImplementedError
            else:
                resnet = ResNet18_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "34":
            if params["bottleneck"]:
                raise NotImplementedError
            else:
                resnet = ResNet34_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "50":
            if params["bottleneck"]:
                resnet = ResNet50bn_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
            else:
                resnet = ResNet50_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "152":
            if params["bottleneck"]:
                resnet = ResNet152bn_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
            else:
                resnet = ResNet152_CIFAR(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
    else:
        if resnet_depth == "18":
            if params["bottleneck"]:
                raise NotImplementedError
            else:
                resnet = ResNet18(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "34":
            if params["bottleneck"]:
                raise NotImplementedError
            else:
                resnet = ResNet34(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "50":
            if params["bottleneck"]:
                resnet = ResNet50bn(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
            else:
                resnet = ResNet50(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
        elif resnet_depth == "152":
            if params["bottleneck"]:
                resnet = ResNet152bn(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )
            else:
                resnet = ResNet152(
                    params["num_classes"],
                    n_input_channels=params["input_channels"],
                    input_dimension=params["input_dim"],
                    final_layer_dropout=params["resnet_dropout"],
                    stochastic_depth_p=params["stochastic_depth"],
                    squeeze_excitation=params["squeeze_excitation"],
                    squeeze_excitation_rd_ratio=params["se_rd_ratio"],
                )

    model = ModelConstructor(resnet, params)

    return model
