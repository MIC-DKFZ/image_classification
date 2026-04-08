MODEL_TO_ARCH_MAPPING = {
    "VisionTransformer": "VisionTransformer",
    "DinoVisionTransformer": "DinoVisionTransformer",
    "DINOv3ViTModel": "DINOv3ViTModel",
    "ViTModel": "ViTModel",
    "ViTMAEModel": "ViTModel",
    "Eva": "Eva",
}

BLOCK_MARKERS = {
    "VisionTransformer":     ["norm1", "attn", "norm2", "mlp"],
    "DinoVisionTransformer": ["norm1", "attn", "norm2", "mlp"],
    "DINOv3ViTModel":        ["norm1", "attention", "norm2", "mlp"],
    "ViTModel":              ["layernorm_before", "attention", "layernorm_after", "intermediate"],
    "Eva":                   ["norm1", "attn", "norm2", "mlp"],
}


class PartialFinetuning:
    """
    Partial Fine-Tuning: freeze the first `num_frozen_layers` transformer blocks,
    fine-tune all remaining blocks plus the head.

    Standard intermediate baseline between Linear Probing (all frozen) and
    Full Fine-Tuning (all trainable). Appears as a reference in the VPT,
    AdaptFormer, and SSF benchmark papers.
    """

    def __init__(self, num_frozen_layers: int, *args, **kwargs):
        target_arch = MODEL_TO_ARCH_MAPPING[self.model.__class__.__name__]
        markers = BLOCK_MARKERS[target_arch]

        blocks = [
            blk for blk in self.model.modules()
            if all(hasattr(blk, m) for m in markers)
        ]

        # Freeze everything, then selectively unfreeze
        for param in self.parameters():
            param.requires_grad = False

        for blk in blocks[num_frozen_layers:]:
            for param in blk.parameters():
                param.requires_grad = True

        for name, param in self.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head"]):
                param.requires_grad = True
