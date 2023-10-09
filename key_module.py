import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from model import StrokeNet, StrokeNetRot
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    ChainedScheduler,
    CosineAnnealingLR,
    ConstantLR,
    SequentialLR
)

from torchmetrics import Metric
from typing import Optional

class EER(Metric):

  full_state_update: bool = True
  higher_is_better: Optional[bool] = False

  def __init__(self):
    super().__init__()
    self.add_state("eers", default=[], dist_reduce_fx=None)

  def update(self, y_pred: torch.Tensor, y: torch.Tensor):
    # compute the current eer

    roc = torchmetrics.ROC(task="binary")
    fpr, tpr, thresholds = roc(y_pred, y)

    fnr = 1 - tpr
    diff = torch.abs(fpr - fnr)
    eer_idx = torch.argmin(diff)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    self.eers.append(eer)

  def compute(self):
    # mean eer
    return torch.mean(torch.tensor(self.eers))


class KeyModule(pl.LightningModule):

  def __init__(
      self,
      user_cnt: int,
      feat_cnt: int,
      key_cnt: int,
      key_emb_size: int,
      dim_ff: int,
      num_heads: int,
      num_layers: int,
      dropout: float,
      lr: float,
      causal_att: bool,
      use_user_emb: bool,
  ):
    super().__init__()

    self.lr = lr
    self.use_user_emb = use_user_emb

    self.stroke_net = StrokeNetRot(
        user_cnt=user_cnt,
        feat_cnt=feat_cnt,
        key_cnt=key_cnt,
        key_emb_size=key_emb_size,
        dim_ff=dim_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        causal_att=causal_att,
        use_user_emb=use_user_emb
    )
    self.cross_token = nn.BCEWithLogitsLoss()
    self.cross_user = nn.BCEWithLogitsLoss()

    self.val_eer = EER()
    self.val_eer_token = EER()


  def forward(self, batch):

    out_pr, out_user = self.stroke_net(*batch)

    return out_pr, out_user

  def training_step(self, batch, batch_idx):

    batch_size = batch[0].shape[0]

    # get the target
    y = batch[-1]
    user_target = batch[-2]

    y_pred, u_pred = self.forward(batch[:-2])

    # this aggregates all tokens into a large tensor
    y_pred = y_pred.view(-1, y_pred.shape[-1])
    y = y.view(-1)

    # get the indices where y is not -1
    idx = torch.where(y != -1)[0]
    y_pred = y_pred[idx]
    y = y[idx]

    y_oh = F.one_hot(y, num_classes=2).float()
    loss_token = self.cross_token(y_pred, y_oh)
    self.log("token_loss",
                loss_token,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                )  

    if self.use_user_emb:
      user_target_oh = F.one_hot(user_target, num_classes=2).float()
      loss_user = self.cross_user(u_pred, user_target_oh)
      self.log("user_loss",
                loss_user,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
                )

      loss = loss_token + loss_user
      self.log("train_loss_combined",
              loss,
              on_step=False,
              on_epoch=True,
              prog_bar=True,
              logger=True,
              batch_size=batch_size,
              )

      return loss

    return loss_token

  def validation_step(self, batch, batch_idx):

    batch_size = batch[0].shape[0]

    # get the target
    y = batch[-1]
    user_target = batch[-2]

    y_pred, u_pred = self.forward(batch[:-2])

    # this aggregates all tokens into a large tensor
    y_pred = y_pred.view(-1, y_pred.shape[-1])
    y = y.view(-1)

    # get the indices where y is not -1
    idx = torch.where(y != -1)[0]

    y_pred = y_pred[idx]
    y = y[idx]

    y_oh = F.one_hot(y, num_classes=2).float()
    loss_token = self.cross_token(y_pred, y_oh)
    self.log("val_token_loss",
             loss_token,
             on_step=False,
             on_epoch=True,
             prog_bar=True,
             logger=True,
             batch_size=batch_size,
             )
    
    if self.use_user_emb:

      user_target_oh = F.one_hot(user_target, num_classes=2).float()
      loss_user = self.cross_user(u_pred, user_target_oh)
      self.log("val_user_loss",
            loss_user,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
            )

      u_soft = torch.softmax(u_pred, dim=-1)[:, 1]
      self.val_eer(u_soft, user_target)
      self.log("val_eer",
              self.val_eer,
              on_step=False,
              on_epoch=True,
              prog_bar=True,
              logger=True,
              batch_size=batch_size,
              )

      loss = loss_token + loss_user
      self.log("val_loss_combined",
              loss,
              on_step=False,
              on_epoch=True,
              prog_bar=True,
              logger=True,
              batch_size=batch_size,
              sync_dist=True,
              )
      return loss
    
    return loss_token

  def test_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
    scheduler_ct = ConstantLR(optimizer, factor=1, total_iters=25)
    scheduler_ca = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-8)
    seq_scheduler = SequentialLR(optimizer, [scheduler_ct, scheduler_ca], milestones=[25])

    lr_scheduler = {
        'scheduler': seq_scheduler,
    }

    return {
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'monitor': 'train_loss_combined'
    }