import os
import wandb
import random
import torch
import torch.nn as nn
import lightning.pytorch as pl

# faster convergence - does not lower eer
from rff import GaussianFourierFeatureTransform
from key_dm import KeyDataModule
from key_module import KeyModule
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks import LearningRateMonitor
import hydra
from omegaconf import DictConfig, OmegaConf

from utils import *

def train(cfg: DictConfig, key_module):

  key_data_module = KeyDataModule(cfg)

  if cfg.wandb.use:
    wandb.login(key=cfg.wandb.key)
    logger=WandbLogger(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg),
            name=cfg.wandb.name,
            save_dir=os.getcwd(),
            offline=False,
        )
  else:
    logger=None

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
          monitor="val_eer",
          dirpath=os.getcwd() + cfg.checkpoint_dir,
          filename='{epoch}_{val_eer:.4f}',
          save_top_k=3,
          mode="min",
      )
  
  trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=cfg.train.max_epochs,
        callbacks=[LearningRateMonitor(logging_interval="step"),
                    checkpoint_callback],
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        strategy="ddp",
        )

  trainer.fit(key_module, key_data_module)
  

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

  torch.set_float32_matmul_precision("high")
  torch.cuda.seed_all()
  random.seed(42)

  if cfg.prep_data:
    cleanup_data(cfg.pretrain.data.path)
    cleanup_data(cfg.finetune.data.path)    
    prepare_data(cfg)
    exit()

  if cfg.new_finetune_data:
    # read the user_mapping
    curr_fine_mapping = pickle.load(open(cfg.finetune.data.path + "user_mapping.pkl", "rb"))
    cleanup_data(cfg.finetune.data.path, remove_mapping=False)
    split_data(curr_fine_mapping, cfg.finetune.data.train_size, cfg.finetune.data.test_size, cfg.finetune.data.path, cfg.raw_data_path)
    print("New finetune data created.")
    exit()


  if cfg.pretrain.run:
    pretrain_cfg = cfg.pretrain

    key_module = KeyModule(
      user_cnt=pretrain_cfg.data.user_cnt,
      feat_cnt=cfg.model_params.feat_cnt,
      key_cnt=cfg.model_params.key_cnt,
      key_emb_size=cfg.model_params.key_emb_size,
      dim_ff=cfg.model_params.dim_ff,
      num_heads=cfg.model_params.num_heads,
      num_layers=cfg.model_params.num_layers,
      dropout=cfg.model_params.trf_dropout,
      lr=pretrain_cfg.train.lr,
      causal_att=cfg.model_params.causal_att,
      use_user_emb=cfg.model_params.use_user_emb,
    )

    train(pretrain_cfg, key_module)

  if cfg.finetune.run:

    finetune_cfg = cfg.finetune

    key_module = KeyModule.load_from_checkpoint(
      finetune_cfg.checkpoint,
      user_cnt=cfg.pretrain.data.user_cnt,
      feat_cnt=cfg.model_params.feat_cnt,
      key_cnt=cfg.model_params.key_cnt,
      key_emb_size=cfg.model_params.key_emb_size,
      dim_ff=cfg.model_params.dim_ff,
      num_heads=cfg.model_params.num_heads,
      num_layers=cfg.model_params.num_layers,
      dropout=cfg.model_params.trf_dropout,
      lr=cfg.finetune.train.lr,
      causal_att=cfg.model_params.causal_att,
      use_user_emb=cfg.model_params.use_user_emb)

    # get user embeddings from model
    user_emb = key_module.stroke_net.user_embedding.weight.detach().cpu()

    # compute average user embedding
    avg_user_emb = torch.mean(user_emb, axis=0)

    # replace user embeddings table with average user embedding
    finetune_user_emb = torch.nn.Embedding(
      cfg.finetune.data.user_cnt, avg_user_emb.shape[0])
    
    # finetune_user_emb.weight = nn.Parameter(avg_user_emb.repeat(cfg.finetune.data.user_cnt, 1))

    # replace user embeddings table in model
    key_module.stroke_net.user_embedding = finetune_user_emb

    # # freeze key embeddings
    # key_module.stroke_net.keycode_embedding.weight.requires_grad = False

    train(finetune_cfg, key_module)
    



if __name__ == "__main__":
  main()