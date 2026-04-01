class BitFit:
    """
    BitFit (ACL 2022): freeze all parameters except bias terms and the head.

    Trains only bias vectors across all transformer layers (attention, MLP,
    LayerNorm). Typically <0.1% of total parameters for ViT-B.

    Paper: https://arxiv.org/abs/2106.10199
    """

    def __init__(self, *args, **kwargs):
        for name, param in self.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head", "bias"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
