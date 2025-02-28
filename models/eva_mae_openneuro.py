from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.eva import Eva
from nnssl.training.nnsslTrainer.evaMAE.dynamic_network.vit_embed_decode import (
    PatchEmbed,
)
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist
from einops import rearrange
from peft import get_peft_model, LoraConfig, TaskType
from base_model import BaseModel
from models.classification_head import ClassificationHead


class EvaEncoder(Module):
    def __init__(
        self,
        embed_dim,
        patch_embed_size,
        input_shape,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        patch_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        rope_kwargs,
        chpt_path,
        **hypparams,
    ):
        super(EvaEncoder, self).__init__()

        self.eva = Eva(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ref_feat_shape=tuple(
                [
                    i // ds for i, ds in zip(input_shape, patch_embed_size)
                ]  # input_shape=patch size, patch_embed_size=(8,8,8)
            ),
            num_reg_tokens=num_reg_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_kwargs=rope_kwargs,
            # class_token=True,
        )
        self.down_projection = PatchEmbed(
            patch_embed_size, hypparams["input_channels"], embed_dim
        )

    def forward(self, x):
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        x = rearrange(x, "b c w h d -> b (h w d) c")
        x, _ = self.eva(x)

        return x


class Eva_MAE(BaseModel):
    def __init__(
        self,
        embed_dim,
        patch_embed_size,
        input_shape,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        patch_drop_rate,
        proj_drop_rate,
        attn_drop_rate,
        rope_kwargs,
        chpt_path,
        **hypparams,
    ):
        super(Eva_MAE, self).__init__(**hypparams)

        self.eva_encoder = EvaEncoder(
            embed_dim,
            patch_embed_size,
            input_shape,
            depth,
            num_heads,
            num_reg_tokens,
            use_rot_pos_emb,
            use_abs_pos_emb,
            mlp_ratio,
            drop_path_rate,
            drop_path_scale,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            rope_kwargs,
            chpt_path,
            **hypparams,
        )

        if self.pretrained:
            self.eva_encoder = load_pretrained_weights(
                self.eva_encoder,
                chpt_path,
                handle_input_shape_mismatch=hypparams[
                    "pretraining_input_shape_mismatch"
                ],
                load_cls_token=hypparams["load_cls_token"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.eva_encoder.named_parameters():

                    if not hypparams["load_cls_token"] and "cls_token" in n:
                        param.requires_grad = True  # make cls_token trainable if it's not loaded from pretrained weights
                    else:
                        param.requires_grad = False

            elif hypparams["finetune_method"] == "lora":
                # Apply LoRA to attention layers

                lora_config = LoraConfig(
                    # task_type=TaskType.IMAGE_CLASSIFICATION,
                    r=8,  # LoRA rank
                    lora_alpha=32,  # Scaling factor
                    lora_dropout=0.1,
                    target_modules=["attn.qkv", "attn.proj"],
                )

                self.eva_encoder.eva = get_peft_model(self.eva_encoder.eva, lora_config)

                # Freeze all layers except LoRA-adapted ones
                for param in self.eva_encoder.parameters():
                    param.requires_grad = False

                for name, param in self.eva_encoder.eva.named_parameters():
                    if "lora" in name:
                        param.requires_grad = True

            else:
                raise NotImplementedError

        self.cls_head = ClassificationHead(
            embed_dim,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
            cls_token_available=hypparams["cls_token_available"],
        )

    def forward(self, x):
        x = self.eva_encoder(x)
        x = self.cls_head(x)

        return x


def load_pretrained_weights(
    eva_model,
    pretrained_weights_file,
    handle_input_shape_mismatch="interpolate",
    load_cls_token=True,
    verbose=True,
):

    # Load weights
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)

    pretrained_dict = saved_model["network_weights"]
    pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items()}

    if isinstance(eva_model, DDP):
        mod = eva_model.module
    else:
        mod = eva_model

    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    # Initialize model state dictionary
    model_dict = mod.state_dict()

    # adjust pos_embed if necessary
    if handle_input_shape_mismatch == "interpolate":

        pretrained_pos_embed = pretrained_dict["eva.pos_embed"]
        model_pos_embed_shape = model_dict["eva.pos_embed"].shape
        # Separate the class token and patch tokens
        cls_pos_embed = pretrained_pos_embed[:, :1, :]  # Shape: [1, 1, 864]
        patch_pos_embed = pretrained_pos_embed[:, 1:, :]  # Shape: [1, 13824, 864]

        # Interpolate the patch positional embeddings
        resized_patch_pos_embed = F.interpolate(
            patch_pos_embed.permute(0, 2, 1),  # [B, C, Tokens] for interpolation
            size=model_pos_embed_shape[1]
            - 1,  # Target number of patch tokens (subtract 1 for the class token)
            mode="linear",
            align_corners=False,
        ).permute(
            0, 2, 1
        )  # Back to [B, Tokens, C]

        # Concatenate the class token positional embedding with the resized patch embeddings
        resized_pos_embed = torch.cat([cls_pos_embed, resized_patch_pos_embed], dim=1)
        # Update the model dictionary
        pretrained_dict["eva.pos_embed"] = resized_pos_embed

    else:
        raise NotImplementedError

    # Filter out unnecessary keys based on match_encoder_only flag
    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.append(".decoder.")
    skip_strings_in_pretrained.append("up_projection")

    if not load_cls_token:
        skip_strings_in_pretrained.append("eva.cls_token")

    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                f"compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        pretrained_weights_file,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and nnUNet architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")

    mod.load_state_dict(model_dict)

    return mod
