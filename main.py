import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
import mlflow.pytorch
from base_model import TimerCallback
from utils import detect_misconfigurations, get_model, get_params_to_log, get_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose your model configuration for training")

    ##### Model #####
    parser.add_argument("model", type=str, help="Name of the model")

    ##### Training Settings #####
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--random_batches",
        action="store_true",
        help="uses random batches with replacement instead of seeing each example once per epoch",
    )
    parser.add_argument("--optimizer", type=str, default="SGD", help="SGD / Madgrad / Adam / AdamW / Rmsprop")
    parser.add_argument("--SAM", action="store_true", help="Enables Sharpness Aware Minimization")
    parser.add_argument("--ASAM", action="store_true", help="Enables Adaptive Sharpness Aware Minimization")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--nesterov", action="store_true", help="Enables Nesterov acceleration for SGD")
    parser.add_argument("--wd", type=float, help="Weight Decay", default=5e-4)
    parser.add_argument(
        "--undecay_norm",
        action="store_true",
        help="If enabled weight decay is not applied to bias and batch norm parameters",
    )
    parser.add_argument(
        "--scheduler", type=str, default="", help="MultiStep / Step / CosineAnneal - By default no scheduler is used"
    )
    parser.add_argument(
        "--T_max",
        default=None,
        type=int,
        help=(
            "Defines the amount of epochs in which CosineAnneal scheduler decays LR to minimum LR, "
            "afterwards LR gets increased again to initial LR for T_max epochs before decaying again,"
            "if not specified, T_max will be set to the nb of epochs so that LR never gets increased"
        ),
    )
    parser.add_argument(
        "--warmstart",
        default=0,
        type=int,
        help=(
            "Specifies the nb of epochs for the CosineAnneal scheduler where "
            "the LR will be gradually increased as a warmstart"
        ),
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="baseline",
        help="baseline / baseline_cutout / autoaugment / randaugment / album",
    )
    parser.add_argument("--mixup", action="store_true", help="Enables mixing up data samples during training")
    parser.add_argument("--mixup_alpha", default=0.2, type=float)
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Label Smoothing parameter, range:0.0-1.0, the higher the more smoothing, default appliesno smoothing",
    )
    parser.add_argument(
        "--stochastic_depth",
        default=0.0,
        type=float,
        help="Dropout rate for stochastic depth, only for ResNet-like models, default applies nostochastic depth",
    )
    parser.add_argument(
        "--final_layer_dropout",
        default=0.0,
        type=float,
        help="Final layer dropout rate, only for Resnet-like models, default applies no dropout",
    )
    parser.add_argument("--se", action="store_true", help="Enables Squeeze and Excitation for ResNet-like models")
    parser.add_argument("--se_rd_ratio", default=0.25, type=float, help="Squeeze Excitation Reduction Ratio")
    parser.add_argument(
        "--shakedrop", action="store_true", help="Enables ShakeDrop Regularization for PyramidNet models"
    )
    parser.add_argument(
        "--zero_init_residual",
        action="store_true",
        help=(
            "Enables Zero-initialization of the last BN (or conv for PreAct models) in each "
            "residual branch, only for ResNet-like models"
        ),
    )
    parser.add_argument(
        "--bottleneck", action="store_true", help="Whether to use bottleneck building blocks for ResNet"
    )

    ##### Metrics #####
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["acc"],
        help=(
            "List of Metrics to be computed. acc=Accuracy, top5acc=Top-5 Accuracy, f1=F1 Score, pr=Precision and"
            " Recall, mse=Mean Squared Error, mae=Mean Absolute Errors"
        ),
    )

    ##### Seeding #####
    parser.add_argument("--seed", default=None, help="If a seed is specified training will be deterministic and slower")

    ##### Data #####
    parser.add_argument("--data", help="Name of the dataset", default="CIFAR10")
    parser.add_argument("--num_classes", help="Number of classes to classify", default=10)
    parser.add_argument(
        "--cifar_size",
        action="store_true",
        help="Whether input images to model are small. If yes, implementations for CIFAR will be used.",
    )
    parser.add_argument("--input_dim", type=int, help="Whether your input is 1, 2 or 3 dimensional", default=2)
    parser.add_argument("--input_channels", type=int, help="Number of channels of input data", default=3)
    parser.add_argument(
        "--regression",
        action="store_true",
        help=(
            "Flag whether target is continuous / regression task. Will set num_classes to 1 and use MSE Loss instead"
            " of CE"
        ),
    )

    ##### Directories #####
    parser.add_argument(
        "--data_dir",
        default=os.environ["DATASET_LOCATION"] if "DATASET_LOCATION" in os.environ.keys() else "./data",
        help="Location of the dataset",
    )
    parser.add_argument(
        "--exp_dir",
        default=os.environ["EXPERIMENT_LOCATION"] if "EXPERIMENT_LOCATION" in os.environ.keys() else "./experiments",
        help="Location where MLflow logs should be saved in local environment",
    )

    ##### Checkpoint saving #####
    parser.add_argument(
        "--save_model", action="store_true", help="Saves the model checkpoint after training in the exp_dir"
    )
    parser.add_argument(
        "--chpt_name",
        default="",
        help="Name of the checkpoint file, if not specified it will use the model name, epoch and test metrics",
    )

    ##### Environment #####
    parser.add_argument("--gpu_count", type=int, help="Nb of GPUs", default=1)
    parser.add_argument("--num_workers", help="Number of workers for loading the data", type=int, default=12)

    ##### Verbosity #####
    parser.add_argument(
        "--suppress_progress_bar", action="store_true", help="Will suppress the Lightning progress bar during training"
    )

    args = parser.parse_args()

    model_name = args.model
    data_dir = args.data_dir
    exp_dir = args.exp_dir
    seed = args.seed

    if seed:
        pl.seed_everything(seed)

    # select correct directories according to dataset
    selected_data_dir = os.path.join(data_dir, args.data if not args.data == "Imagenet" else "ILSVRC_2012")
    selected_exp_dir = os.path.join(exp_dir, args.data)

    # set MLflow and checkpoint directories
    chpt_dir = os.path.join(selected_exp_dir, "checkpoints")
    mlrun_dir = os.path.join(selected_exp_dir, "mlruns")

    # check for misconfigurations in the parameters
    detect_misconfigurations(model_name, args)

    # save specified parameters in dictionaries
    params = get_params(selected_data_dir, model_name, args, seed)
    params_to_log = get_params_to_log(params, model_name)

    # Choose correct model and num_classes
    if args.data.startswith("CIFAR"):
        num_classes = 10 if args.data == "CIFAR10" else 100
        params["cifar_size"] = True
    elif args.data == "Imagenet":
        num_classes = 1000
    else:
        num_classes = args.num_classes

    params["num_classes"] = num_classes if not params["regression"] else 1
    model = get_model(model_name, params)

    ## Pytorch Lightning Trainer
    # Checkpoint callback if model should be saved
    chpt_name = args.chpt_name if len(args.chpt_name) > 0 else model_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=chpt_dir, filename=chpt_name + "-{epoch}-{val_loss:.2f}-{val_acc:.3f}"
    )

    # Sharpness Aware Minimization fails with 16-bit precision because
    # GradScaler does not support closure functions at the moment
    # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
    precision_value = 32 if params["sam"] or args.gpu_count == 0 else 16

    # Make run deterministic if a seed is given
    benchmark = False if seed else True
    deterministic = True if seed else False

    # add checkpoint callback only if you want to save model weights
    all_lightning_callbacks = [TimerCallback(params["epochs"], args.gpu_count)]
    if args.save_model:
        all_lightning_callbacks.append(checkpoint_callback)

    # Configure Trainer
    trainer = pl.Trainer(
        logger=None,
        devices=args.gpu_count,
        accelerator="gpu" if args.gpu_count > 0 else "cpu",
        sync_batchnorm=True if args.gpu_count > 1 else False,
        callbacks=all_lightning_callbacks,
        enable_checkpointing=True if args.save_model else False,
        max_epochs=args.epochs,
        benchmark=benchmark,
        deterministic=deterministic,
        precision=precision_value,
        enable_progress_bar=not args.suppress_progress_bar,
        strategy="ddp" if args.gpu_count > 1 else None,
    )

    # Train the model
    if trainer.is_global_zero:  # in case of multi gpu training only log parameters in process 0

        # Log location
        mlflow.set_tracking_uri(mlrun_dir)
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(log_models=False)

        # set MLflow experiment name
        mlflow.set_experiment(args.data)  # creates exp if it does not exist yet
        run_name = f"{args.data}-{model_name}"

        with mlflow.start_run(run_name=run_name) as run:

            mlflow.log_params(params_to_log)
            trainer.fit(model)
    else:
        trainer.fit(model)
