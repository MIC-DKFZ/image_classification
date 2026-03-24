import torch.nn as nn


class LNTuning:
    """
    LayerNorm Tuning: freeze all parameters except LayerNorm (γ, β) and the head.

    Adds zero new parameters — only the existing scale and shift of every
    nn.LayerNorm in the backbone are kept trainable. Typically ~0.15% of
    ViT-B parameters. Architecture-agnostic: works across all ViT variants
    without any module mapping.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.parameters():
            param.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True

        for name, param in self.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head"]):
                param.requires_grad = True
