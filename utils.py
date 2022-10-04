# Available Models
from base_model import ModelConstructor

# from models.vgg import VGG
from models.dynamic_vgg import get_vgg
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152

# from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet110, ResNet152
from models.dynamic_resnet import get_resnet
from models.wide_resnet import WRN2810
from models.efficientnet import *
from models.pyramidnet import PyramidNet110, PyramidNet272

registered_models = [
    "VGG16",
    "VGG19",
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet110",
    "ResNet152",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "EfficientNetB8",
    "EfficientNetL2",
    "WRN2810",
    "PyramidNet110",
    "PyramidNet272",
]

registered_datasets = ["CIFAR10", "CIFAR100", "Imagenet"]


def get_model(model_name, params):
    num_classes = params["num_classes"]
    # if model_name == "VGG16":
    #    model = VGG("VGG16", num_classes=num_classes, hypparams=params)

    if model_name.startswith("VGG"):
        model = get_vgg(params=params)

    elif model_name == "PreActResNet18":
        model = PreActResNet18(num_classes=num_classes, hypparams=params)
    elif model_name == "PreActResNet34":
        model = PreActResNet34(num_classes=num_classes, hypparams=params)
    elif model_name == "PreActResNet50":
        model = PreActResNet50(num_classes=num_classes, hypparams=params)
    elif model_name == "PreActResNet101":
        model = PreActResNet101(num_classes=num_classes, hypparams=params)
    elif model_name == "PreActResNet152":
        model = PreActResNet152(num_classes=num_classes, hypparams=params)

    elif model_name.startswith("ResNet"):

        model = get_resnet(params=params)

    # else:

    # if model_name == 'ResNet18':
    #    model = ResNet18(num_classes=num_classes, hypparams=params)
    # elif model_name == 'ResNet34':
    #    model = ResNet34(num_classes=num_classes, hypparams=params)
    # elif model_name == 'ResNet50':
    #    model = ResNet50(num_classes=num_classes, hypparams=params)
    # elif model_name == 'ResNet101':
    #    model = ResNet101(num_classes=num_classes, hypparams=params)
    # elif model_name == 'ResNet110':
    #    model = ResNet110(num_classes=num_classes, hypparams=params)
    # elif model_name == 'ResNet152':
    #    model = ResNet152(num_classes=num_classes, hypparams=params)

    elif model_name == "EfficientNetL2":
        model = EfficientNetL2(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB8":
        model = EfficientNetB8(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB7":
        model = EfficientNetB7(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB6":
        model = EfficientNetB6(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB5":
        model = EfficientNetB5(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB4":
        model = EfficientNetB4(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB3":
        model = EfficientNetB3(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB2":
        model = EfficientNetB2(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB1":
        model = EfficientNetB1(num_classes=num_classes, hypparams=params)
    elif model_name == "EfficientNetB0":
        model = EfficientNetB0(num_classes=num_classes, hypparams=params)

    elif model_name == "WRN2810":
        model = WRN2810(num_classes=num_classes, hypparams=params)

    elif model_name == "PyramidNet110":
        model = PyramidNet110(num_classes=num_classes, hypparams=params)
    elif model_name == "PyramidNet272":
        model = PyramidNet272(num_classes=num_classes, hypparams=params)

    return model


def detect_misconfigurations(model_name, args):

    # data
    assert (
        args.data in registered_datasets
    ), "Only {} datasets are supported. See the README for instructions how to add new datasets".format(
        registered_datasets
    )
    # Model
    assert model_name in registered_models, "Specified Model {} not available. Have you registered it?".format(
        model_name
    )
    # Training
    assert args.optimizer in ["SGD", "Madgrad", "Adam", "AdamW", "Rmsprop"], "Optimizer {} not recognized".format(
        args.optimizer
    )
    assert args.scheduler in ["", "MultiStep", "Step", "CosineAnneal"]
    assert not (
        args.ASAM and args.SAM
    ), "Either use adaptive (--ASAM) or non-adaptive (--SAM) Sharpness Aware Minimization"
    if args.nesterov:
        assert args.optimizer == "SGD", "Nesterov only available for SGD optimizer"
    if args.warmstart > 0:
        assert args.scheduler == "CosineAnneal", "Warmstart only available for Cosine Annealing Scheduler"
    if args.T_max:
        assert (
            args.scheduler == "CosineAnneal"
        ), "T_max is a parameter of the Cosine Annealing Scheduler, but you specified {}".format(args.scheduler)
    assert args.augmentation in [
        "baseline",
        "baseline_cutout",
        "autoaugment",
        "randaugment",
        "album",
    ], "not a valid augmentation"
    if args.mixup_alpha != 0.2:
        assert args.mixup, "Mixup has to be True for specifying the mixup alpha parameter"
    # Model specific settings
    if args.bottleneck:
        assert (
            "ResNet" in model_name or "Pyramid" in model_name and "18" not in model_name and "34" not in model_name
        ), "Bottleneck not available for {}".format(model_name)
    if args.zero_init_residual or args.se or args.shakedrop:
        assert (
            "ResNet" in model_name or "WRN" in model_name or "Pyramid" in model_name
        ), "specified regularization only for ResNet-like models"
    if args.stochastic_depth != 0 or args.final_layer_dropout != 0:
        assert (
            "ResNet" in model_name or "WRN" in model_name or "Pyramid" in model_name
        ), "specified regularization only for ResNet-like models"


def get_params(selected_data_dir, model_name, args, seed):

    args.SAM = True if args.ASAM else args.SAM

    params = {
        "data_dir": selected_data_dir,
        "name": model_name,
        "batch_size": args.batch_size,
        "random_batches": args.random_batches,
        "optimizer": args.optimizer,
        "sam": args.SAM,
        "adaptive_sam": args.ASAM,
        "lr": args.lr,
        "nesterov": args.nesterov,
        "scheduler": args.scheduler if len(args.scheduler) > 0 else None,
        "T_max": args.T_max if args.T_max else args.epochs,
        "warmstart": args.warmstart,
        "augmentation": args.augmentation,
        "mixup": args.mixup,
        "mixup_alpha": args.mixup_alpha if args.mixup else 0.0,
        "dataset": args.data,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "weight_decay": args.wd,
        "undecay_norm": args.undecay_norm,
        "label_smoothing": args.label_smoothing,
        "stochastic_depth": args.stochastic_depth,
        "resnet_dropout": args.final_layer_dropout,
        "squeeze_excitation": args.se,
        "se_rd_ratio": args.se_rd_ratio,
        "shakedrop": args.shakedrop,
        "zero_init_residual": args.zero_init_residual,
        "bottleneck": args.bottleneck,
        "seed": seed,
        "cifar_size": args.cifar_size,
        "input_dim": args.input_dim,
        "input_channels": args.input_channels,
        "regression": args.regression,
        "metrics": args.metrics,
        "num_gpus": args.gpu_count,
        "metric_computation_mode": args.metric_computation_mode,
    }

    return params


def get_params_to_log(params, model_name):
    # handle SAM and optimizer name
    if params["adaptive_sam"]:
        opt = params["optimizer"] + " - ASAM"
    elif params["sam"]:
        opt = params["optimizer"] + " - SAM"
    else:
        opt = params["optimizer"]

    params_to_log = {
        "batch_size": params["batch_size"],
        "scheduler": params["scheduler"],
        "undecay_norm": params["undecay_norm"],
        "T_max": params["T_max"] if params["scheduler"] == "CosineAnneal" else None,
        "warmstart": params["warmstart"],
        "optimizer_name": opt,
        "sam": params["sam"],
        "adaptive_sam": params["adaptive_sam"],
        "augmentation": params["augmentation"],
        "mixup": params["mixup"],
        "mixup_alpha": params["mixup_alpha"],
        "model": model_name,
        "label_smoothing": params["label_smoothing"],
        "stochastic_depth": params["stochastic_depth"],
        "final_layer_dropout": params["resnet_dropout"],
        "squeeze_excitation": params["squeeze_excitation"],
        "se_rd_ratio": params["se_rd_ratio"],
        "shakedrop": params["shakedrop"],
        "zero_init_residual": params["zero_init_residual"],
        "bottleneck": params["bottleneck"],
        "seed": params["seed"],
        "random_batches": params["random_batches"],
        "cifar_size": params["cifar_size"],
        "input_dim": params["input_dim"],
        "input_channels": params["input_channels"],
        "regression": params["regression"],
        "num_gpus": params["num_gpus"],
        "metric_computation_mode": params["metric_computation_mode"],
    }

    return params_to_log
