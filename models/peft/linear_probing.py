class LinearProbing:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, param in self.named_parameters():
            if any(sub in name for sub in ["head", "classifier", "cls_head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # TODO: Decide if this should be activated or not. Ask Jeremias and Dasha.
    # def on_save_checkpoint(self, checkpoint):
    #     if self.finetune_method == "linear_probing":
    #         # Modify checkpoint to only contain classifier weights
    #         head_state_dict = {
    #             k: v for k, v in checkpoint["state_dict"].items() if "cls_head" in k
    #         }
    #         checkpoint["state_dict"] = head_state_dict