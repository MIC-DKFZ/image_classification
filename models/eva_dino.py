from dynamic_network_architectures.architectures.dinov2_eva import Eva
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist
from peft import get_peft_model, LoraConfig
from base_model import BaseModel
from models.classification_head import ClassificationHead


class EvaEncoder(Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        proj_drop_rate,
        attn_drop_rate,
        global_crops_size,
        local_crops_size,
        patch_size,
        qkv_bias,
        qkv_fused,
        swiglu_mlp,
        scale_mlp,
        scale_attn_inner,
        dynamic_img_size,
        **hypparams,
    ):
        super(EvaEncoder, self).__init__()

        self.eva = Eva(
            input_channels=hypparams["input_channels"],
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            mlp_ratio=mlp_ratio,
            swiglu_mlp=swiglu_mlp,
            scale_mlp=scale_mlp,
            scale_attn_inner=scale_attn_inner,
            pos_drop_rate=0,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            drop_path_scale=drop_path_scale,
            num_reg_tokens=num_reg_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_emb,
            class_token=True,
            dynamic_img_size=dynamic_img_size,
        )

    def forward(self, x):

        x = self.eva(x)

        cls_token = x["x_norm_clstoken"].unsqueeze(1)
        patch_tokens = x["x_norm_patchtokens"]
        x = torch.concat([cls_token, patch_tokens], dim=1)

        return x


class Eva_Dino(BaseModel):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        num_reg_tokens,
        use_rot_pos_emb,
        use_abs_pos_emb,
        mlp_ratio,
        drop_path_rate,
        drop_path_scale,
        proj_drop_rate,
        attn_drop_rate,
        global_crops_size,
        local_crops_size,
        patch_size,
        qkv_bias,
        qkv_fused,
        swiglu_mlp,
        scale_mlp,
        scale_attn_inner,
        dynamic_img_size,
        chpt_path,
        **hypparams,
    ):
        super(Eva_Dino, self).__init__(**hypparams)

        self.eva_encoder = EvaEncoder(
            embed_dim,
            depth,
            num_heads,
            num_reg_tokens,
            use_rot_pos_emb,
            use_abs_pos_emb,
            mlp_ratio,
            drop_path_rate,
            drop_path_scale,
            proj_drop_rate,
            attn_drop_rate,
            global_crops_size,
            local_crops_size,
            patch_size,
            qkv_bias,
            qkv_fused,
            swiglu_mlp,
            scale_mlp,
            scale_attn_inner,
            dynamic_img_size,
            **hypparams,
        )

        if self.pretrained:
            self.eva_encoder = load_pretrained_weights(
                self.eva_encoder,
                chpt_path,
                # load_cls_token=hypparams["load_cls_token"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for param in self.eva_encoder.parameters():
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

    pretrained_dict = saved_model["teacher"]
    # Remove "backbone." prefix from keys
    fixed_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("backbone.", "eva.")  # Replace by eva prefix
        fixed_state_dict[new_key] = v
    fixed_state_dict = {
        k: v for k, v in fixed_state_dict.items() if not k.startswith("dino_head.")
    }

    if isinstance(eva_model, DDP):
        mod = eva_model.module
    else:
        mod = eva_model

    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    # Initialize model state dictionary
    model_dict = mod.state_dict()

    model_dict.update(fixed_state_dict)
    mod.load_state_dict(model_dict)

    return mod
