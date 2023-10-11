from torchvision.models import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s

from base_model import BaseModel


class EfficientNetV2(BaseModel):
    def __init__(self, type, **kwargs):
        super(EfficientNetV2, self).__init__(**kwargs)

        if type == "S":
            self.model = efficientnet_v2_s(
                num_classes=kwargs["num_classes"],
                dropout=kwargs["resnet_dropout"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )
        elif type == "M":
            self.model = efficientnet_v2_m(
                num_classes=kwargs["num_classes"],
                dropout=kwargs["resnet_dropout"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )
        elif type == "L":
            self.model = efficientnet_v2_l(
                num_classes=kwargs["num_classes"],
                dropout=kwargs["resnet_dropout"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )

    def forward(self, x):
        return self.model(x)
