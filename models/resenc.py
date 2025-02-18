from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
import torch.distributed as dist

from base_model import BaseModel
from models.classification_head import ClassificationHead


class ResEncoder(Module):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder, self).__init__()

        self.res_unet = ResidualEncoderUNet(
            hypparams["input_channels"],
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=[
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            num_classes=hypparams["num_classes"],
        )
        self.res_unet.encoder.return_skips = False

        if hypparams["pretrained"]:
            self.res_unet = load_pretrained_weights(
                self.res_unet,
                hypparams["chpt_path"],
            )

            if hypparams["finetune_method"] == "full":
                pass

            elif hypparams["finetune_method"] == "linear_probing":
                # fully freeze encoder
                for n, param in self.res_unet.named_parameters():
                    param.requires_grad = False

    def forward(self, x):

        x = self.res_unet.encoder(x).mean(dim=[2, 3, 4])

        return x


class ResEncoder_Classifier(BaseModel):
    def __init__(
        self,
        **hypparams,
    ):
        super(ResEncoder_Classifier, self).__init__(**hypparams)

        self.encoder = ResEncoder(**hypparams)

        self.cls_head = ClassificationHead(
            320,
            hypparams["num_classes"],
            dropout=hypparams["classification_head_dropout"],
            patch_aggregation_method=hypparams["token_aggregation_method"],
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls_head(x)

        return x


def load_pretrained_weights(
    resenc_model,
    pretrained_weights_file,
):
    if dist.is_initialized():
        saved_model = torch.load(
            pretrained_weights_file,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(pretrained_weights_file, weights_only=False)
    pretrained_dict = saved_model["network_weights"]

    if isinstance(resenc_model, DDP):
        mod = resenc_model.module
    else:
        mod = resenc_model
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    in_conv_weights_model: torch.Tensor = model_dict[
        "encoder.stem.convs.0.all_modules.0.weight"
    ]
    in_conv_weights_pretrained: torch.Tensor = pretrained_dict[
        "encoder.stem.convs.0.all_modules.0.weight"
    ]

    in_channels_model = in_conv_weights_model.shape[1]
    in_channels_pretrained = in_conv_weights_pretrained.shape[1]

    if in_channels_model != in_channels_pretrained:
        assert in_channels_pretrained == 1, (
            f"The input channels do not match. Pretrained model: {in_channels_pretrained}; your network: "
            f"your network: {in_channels_model}"
        )

        repeated_weight_tensor = in_conv_weights_pretrained.repeat(
            1, in_channels_model, 1, 1, 1
        )
        target_data_ptr = in_conv_weights_pretrained.data_ptr()
        for key, weights in pretrained_dict.items():
            if weights.data_ptr() == target_data_ptr:
                # print(key)
                pretrained_dict[key] = repeated_weight_tensor

        # SPECIAL CASE HARDCODE INCOMING
        # Normally, these keys have the same data_ptr that points to the weights that are to be replicated:
        # - encoder.stem.convs.0.conv.weight
        # - encoder.stem.convs.0.all_modules.0.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.all_modules.0.weight
        # But this is not the case for 'VariableSparkMAETrainer_BS8', where we replace modules from the original
        # encoder architecture, so that the following two point to a different tensor:
        # - encoder.stem.convs.0.conv.weight
        # - decoder.encoder.stem.convs.0.conv.weight
        # resulting in a shape mismatch for the two missing keys in the check below.
        # It is important to note, that the weights being trained are located at 'all_modules.0.weight', so we
        # have to use those as the source of replication
        if "VariableSparkMAETrainer" in pretrained_weights_file:
            pretrained_dict["encoder.stem.convs.0.conv.weight"] = repeated_weight_tensor
            pretrained_dict["decoder.encoder.stem.convs.0.conv.weight"] = (
                repeated_weight_tensor
            )

        print(
            f"Your network has {in_channels_model} input channels. To accommodate for this, the single input "
            f"channel of the pretrained model is repeated {in_channels_model} times."
        )

    skip_strings_in_pretrained = [".seg_layers."]
    skip_strings_in_pretrained.extend(["decoder.stages", "decoder.transpconvs"])

    final_pretrained_dict = {}
    for key, v in pretrained_dict.items():
        if key in model_dict and all(
            [i not in key for i in skip_strings_in_pretrained]
        ):
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key].shape}. The pretrained model "
                f"does not seem to be compatible with your network."
            )
            final_pretrained_dict[key] = v

    model_dict.update(final_pretrained_dict)

    # print("################### Loading pretrained weights from file ", fname, '###################')
    # print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
    # for key, value in final_pretrained_dict.items():
    #     print(key, 'shape', value.shape)
    # print("################### Done ###################")
    # exit()
    mod.load_state_dict(model_dict)

    return mod
