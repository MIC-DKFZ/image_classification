import os
from pathlib import Path
from uuid import uuid4

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from parsing_utils import make_omegaconf_resolvers


@hydra.main(version_base=None, config_path="./cli_configs", config_name="infer")
def inference(cfg):
    # delete automatically created hydra logger
    try:
        Path(
            "./main.log"
        ).unlink()
    except:
        pass

    # check if a fold was given, if yes scan only the fold dir for the checkpoint path
    if cfg.fold:
        ckp_paths = list(Path(Path(cfg.exp_dir) / str(cfg.fold)).glob("*.ckpt"))
    else:
        ckp_paths = list(Path(cfg.exp_dir).glob("*/*.ckpt"))
    
    logits = []
    for ckp_path in ckp_paths:

        # load the config that was used during training
        used_training_cfg = OmegaConf.load(os.path.join(cfg.exp_dir, "config.yaml"))
        used_training_cfg.trainer.pop("logger")
        used_training_cfg.trainer.pop("callbacks")
        used_training_cfg.model.metrics = cfg.metrics  # overwrite metrics

        # instantiate the model using this config
        model = instantiate(used_training_cfg.model)
        # load the weights
        model.load_state_dict(torch.load(ckp_path)["state_dict"])
        model.eval()

        # instantiate the dataset from the config if not some other dataset is specified in the infer.yaml
        dataset = instantiate(used_training_cfg.data).module
        # instantiate the trainer and also pass the new metrics
        trainer = instantiate(used_training_cfg.trainer)
       
        # run trainer.predict
        y, y_hat = zip(*trainer.predict(model, dataset))
        y = torch.cat(y)
        logits.append(torch.cat(y_hat))

    if len(logits)>1:
        # for an ensemble we sum the logits first
        final_pred = torch.argmax(torch.sum(torch.stack(logits), 0), 1)
    else:
        final_pred = torch.argmax(logits[0], 1)
    
    if cfg.pred_dir is not None:
        os.makedirs(cfg.pred_dir, exist_ok=True)
        #TODO

    # calculate metrics
    print(model.val_metrics(final_pred, y))

if __name__ == "__main__":
    make_omegaconf_resolvers()
    inference()
