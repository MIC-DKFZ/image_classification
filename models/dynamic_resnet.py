from dynamic_network_architectures.architectures.resnet import ResNet18 as dyn_ResNet18, ResNet18_CIFAR as dyn_ResNet18_CIFAR, ResNet34 as dyn_ResNet34, ResNet34_CIFAR as dyn_ResNet34_CIFAR, ResNet50 as dyn_ResNet50, ResNet50bn as dyn_ResNet50bn, ResNet50_CIFAR as dyn_ResNet50_CIFAR, ResNet50bn_CIFAR as dyn_ResNet50bn_CIFAR, ResNet152 as dyn_ResNet152, ResNet152bn as dyn_ResNet152bn, ResNet152_CIFAR as dyn_ResNet152_CIFAR, ResNet152bn_CIFAR as dyn_ResNet152bn_CIFAR
from base_model import BaseModel

def get_resnet_args(params, se_rd_ratio):
    resnet_args = {'n_classes': params["num_classes"],
                    'n_input_channels': params["input_channels"],
                    'input_dimension': params["input_dim"],
                    'final_layer_dropout': params["resnet_dropout"],
                    'stochastic_depth_p': params["stochastic_depth"],
                    'squeeze_excitation': params["squeeze_excitation"],
                    'squeeze_excitation_rd_ratio': se_rd_ratio
                    }
    
    return resnet_args


class ResNet18(BaseModel):
    def __init__(self, cifar_size, se_rd_ratio, bottleneck, **hypparams):
        super(ResNet18, self).__init__(**hypparams)
        resnet_args = get_resnet_args(hypparams, se_rd_ratio)
        
        if bottleneck:
            raise NotImplementedError
        else:
            self.model = dyn_ResNet18_CIFAR(**resnet_args) if cifar_size else dyn_ResNet18(**resnet_args)

    def forward(self, x):
        out = self.model(x)
        return out

class ResNet34(BaseModel):
    def __init__(self, cifar_size, se_rd_ratio, bottleneck, **hypparams):
        super(ResNet34, self).__init__(**hypparams)
        resnet_args = get_resnet_args(hypparams, se_rd_ratio)

        if bottleneck:
            raise NotImplementedError
        else:
            self.model = dyn_ResNet34_CIFAR(**resnet_args) if cifar_size else dyn_ResNet34(**resnet_args)

    def forward(self, x):
        out = self.model(x)
        return out

class ResNet50(BaseModel):
    def __init__(self, cifar_size, se_rd_ratio, bottleneck, **hypparams):
        super(ResNet50, self).__init__(**hypparams)
        resnet_args = get_resnet_args(hypparams, se_rd_ratio)

        if bottleneck:
            self.model = dyn_ResNet50bn_CIFAR(**resnet_args) if cifar_size else dyn_ResNet50bn(**resnet_args)
        else:
            self.model = dyn_ResNet50_CIFAR(**resnet_args) if cifar_size else dyn_ResNet50(**resnet_args)

    def forward(self, x):
        out = self.model(x)
        return out
    
class ResNet152(BaseModel):
    def __init__(self, cifar_size, se_rd_ratio, bottleneck, **hypparams):
        super(ResNet152, self).__init__(**hypparams)
        resnet_args = get_resnet_args(hypparams, se_rd_ratio)

        if bottleneck:
            self.model = dyn_ResNet152bn_CIFAR(**resnet_args) if cifar_size else dyn_ResNet152bn(**resnet_args)
        else:
            self.model = dyn_ResNet152_CIFAR(**resnet_args) if cifar_size else dyn_ResNet152(**resnet_args)

    def forward(self, x):
        out = self.model(x)
        return out
