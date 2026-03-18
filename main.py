from pathlib import Path
import os

import hydra
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
import torch
import importlib
from parsing_utils import make_omegaconf_resolvers


@hydra.main(version_base=None, config_path="./cli_configs", config_name="train")
def main(cfg):
    # Creating the logging directory. This has to happen before wandb.init.
    log_path = Path(cfg.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    wandb.init(**cfg.wandb)
    
    # seeding
    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = True

    # setup logger
    try:
        # gets automatically created, however logs are available in Weights and Biases
        # so we do not need to log twice
        Path("./main.log").unlink()
    except:
        pass

    # add sync_batchnorm if multiple GPUs are used
    if cfg.trainer.devices > 1 and cfg.trainer.accelerator == "gpu":
        cfg.trainer.sync_batchnorm = True

    # remove callbacks that are not enabled
    cfg.trainer.callbacks = [i for i in cfg.trainer.callbacks.values() if i]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            i
            for i in cfg.trainer.callbacks
            if i["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]

    print(OmegaConf.to_yaml(cfg))

    # in case of Cross Validation loop over the folds (default is 1 (no Cross Validation))
    for k in range(cfg.data.cv.k):
        if cfg.data.cv.k > 1:
            cfg.data.module.fold = k
        else:
            if cfg.data.module.fold is not None:
                pass
            else:
                cfg.data.module.fold = "0"

        if cfg.trainer["enable_checkpointing"]:
            for i in cfg.trainer.callbacks:
                if i["_target_"] == "lightning.pytorch.callbacks.ModelCheckpoint":
                    i["dirpath"] = os.path.join(cfg.log_dir, str(cfg.data.module.fold))

        # instantiate trainer, model and dataset
        trainer = instantiate(cfg.trainer)

        ModelClass = make_class(cfg.peft._target_, cfg.model._target_)
        model_args = dict(cfg.peft)
        model_args.update(dict(cfg.model))
        model_args.pop("_target_")
        model = ModelClass(**model_args)
        if cfg.model.compile:
            model = torch.compile(model, mode="default")
        dataset = instantiate(cfg.data).module

        # log hypperparams and drop stuff that shouldn't be logged
        ## Model
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["model"].pop("_target_")
        cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["model"])

        trainer.logger.log_hyperparams({"dataset": cfg_dict["dataset"]})

        cfg_dict["peft"].pop("_target_")
        trainer.logger.log_hyperparams(cfg_dict["peft"])

        ## Data
        cfg_dict["data"]["module"].pop("_target_")
        if cfg_dict["data"]["module"]["train_transforms"] is not None:
            cfg_dict["data"]["module"]["train_transforms"] = ".".join(
                cfg_dict["data"]["module"]["train_transforms"]["_target_"].split(".")[
                    -2:
                ]
            )
        if cfg_dict["data"]["module"]["test_transforms"] is not None:
            cfg_dict["data"]["module"]["test_transforms"] = ".".join(
                cfg_dict["data"]["module"]["test_transforms"]["_target_"].split(".")[
                    -2:
                ]
            )
        cfg_dict["data"]["module"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["data"]["module"])

        ## Trainer
        cfg_dict["trainer"].pop("_target_")
        cfg_dict["trainer"].pop("callbacks")
        cfg_dict["trainer"].pop("enable_checkpointing")
        cfg_dict["trainer"].pop("enable_progress_bar")
        cfg_dict["trainer"].pop("logger")
        cfg_dict["trainer"].pop("num_sanity_val_steps")
        trainer.logger.log_hyperparams(cfg_dict["trainer"])

        # start fitting
        #if cfg_dict["val_only"]:
        #    trainer.validate(model, dataset)
        #else:
        trainer.fit(model, dataset)

        wandb.finish()


def make_class(derived, base):
    derived = resolve_class(derived)
    base = resolve_class(base)
    return type(f"{derived.__name__}_{base.__name__}", (derived, base), {})


def resolve_class(path: str):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


if __name__ == "__main__":
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    make_omegaconf_resolvers()
    main()
