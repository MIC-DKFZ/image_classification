import timm
from base_model import BaseModel


class TimmModel(BaseModel):
    def __init__(self, type, **kwargs):
        super(TimmModel, self).__init__(**kwargs)

        self.model = timm.create_model(
            type,
            pretrained=kwargs["pretrained"],
            in_chans=kwargs["input_channels"],
            num_classes=kwargs["num_classes"],
        )

    def forward(self, x):
        return self.model(x)
