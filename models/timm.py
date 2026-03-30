import timm
from base_model import BaseModel
from models.classification_head import ClassificationHead


class TimmModel(BaseModel):
    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)

        self.model = timm.create_model(
            type,
            pretrained=kwargs["pretrained"],
            in_chans=kwargs["input_channels"],
            num_classes=0,  # strip timm's head; we use ClassificationHead
        )

        self.cls_head = ClassificationHead(
            self.model.num_features,
            kwargs["num_classes"],
            dropout=kwargs["classification_head_dropout"],
            patch_aggregation_method=kwargs["token_aggregation_method"],
        )

    @property
    def encoder_params(self):
        return self.model.parameters()

    @property
    def cls_head_params(self):
        return self.cls_head.parameters()

    def forward(self, x):
        x = self.model.forward_features(x)
        return self.cls_head(x)

    def extract_features(self, x):
        return self.model.forward_features(x)
