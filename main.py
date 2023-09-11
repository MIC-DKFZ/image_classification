from pathlib import Path
from uuid import uuid4

import hydra
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from parsing_utils import make_omegaconf_resolvers


@hydra.main(version_base=None, config_path="./cli_configs", config_name="train")
def main(cfg):
    make_omegaconf_resolvers()

    if cfg.seed:
        seed_everything(cfg.seed)
        cfg.trainer.benchmark = False
        cfg.trainer.deterministic = True

    # setup logger
    try:
        Path(
            "./main.log"
        ).unlink()  # gets automatically created, however logs are available in Weights and Biases so we do not need to log twice
    except:
        pass
    log_path = Path(cfg.trainer.logger.save_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    cfg.trainer.logger.group = str(uuid4())

    # remove callbacks that are not enabled
    cfg.trainer.callbacks = [i for i in cfg.trainer.callbacks.values() if i]
    if not cfg.trainer["enable_checkpointing"]:
        cfg.trainer.callbacks = [
            i for i in cfg.trainer.callbacks if i["_target_"] != "lightning.pytorch.callbacks.ModelCheckpoint"
        ]

    print(OmegaConf.to_yaml(cfg))

    # in case of Cross Validation loop over the folds (default is 1 (no Cross Validation))
    for k in range(cfg.data.cv.k):
        if cfg.data.cv.k > 1:
            cfg.data.module.fold = k

        # instantiate trainer, model and dataset
        trainer = instantiate(cfg.trainer)
        model = instantiate(cfg.model)
        dataset = instantiate(cfg.data).module

        # log hypperparams
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["model"].pop("_target_")
        cfg_dict["model"]["model"] = cfg_dict["model"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["model"])
        cfg_dict["data"]["module"].pop("_target_")
        cfg_dict["data"]["module"]["train_transforms"] = ".".join(
            cfg_dict["data"]["module"]["train_transforms"]["_target_"].split(".")[-2:]
        )
        cfg_dict["data"]["module"]["test_transforms"] = ".".join(
            cfg_dict["data"]["module"]["test_transforms"]["_target_"].split(".")[-2:]
        )
        cfg_dict["data"]["module"].pop("name")
        trainer.logger.log_hyperparams(cfg_dict["data"]["module"])

        # start fitting
        trainer.fit(model, dataset)
        wandb.finish()


if __name__ == "__main__":
    main()
