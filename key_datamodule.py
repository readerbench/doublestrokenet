import os
import torch
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
from dataset import BigramDataset, BigramDatasetVal, BigramPlusDataset


class KeyDataModule(pl.LightningDataModule):
  """
  DataModule for the Key Prediction task.
  """

  def __init__(self,
               root_dir: str = "",
               user_cnt: int = 100,
               max_seq_len: int = 50,
               test_size: int = 5,
               replace_prob: float = 0.1,
               user_prob: float = 0.5,
               batch_size: int = 512,
               val_batch_size: int = 16,
               val_user_cnt: int = 1000,
               num_workers: int = 4,
               dataset_multiplier: int = 1):
    super().__init__()
    self.root_dir = root_dir
    self.user_cnt = user_cnt
    self.max_seq_len = max_seq_len
    self.test_size = test_size
    self.replace_prob = replace_prob
    self.user_prob = user_prob
    self.batch_size = batch_size
    self.val_batch_size = val_batch_size
    self.val_user_cnt = val_user_cnt
    self.num_workers = num_workers
    self.dataset_multiplier = dataset_multiplier
    self.raw_data_dir = os.path.join(self.root_dir, "clean")
    self.train_dir = os.path.join(self.root_dir, "train")
    self.test_dir = os.path.join(self.root_dir, "test")

  def split_data(self, user_mapping):
    """
    Split the data into train and test folders.
    """
    train_files = os.listdir(self.train_dir)
    test_files = os.listdir(self.test_dir)

    for file in train_files:
      os.remove(os.path.join(self.train_dir, file))

    for file in test_files:
      os.remove(os.path.join(self.test_dir, file))

    for user_file, user_id in user_mapping.items():
      user_df = pd.read_csv(os.path.join(self.raw_data_dir, user_file))
      # split data into train and test
      train_df, test_df = train_test_split(user_df,
                                           test_size=self.test_size,
                                           random_state=42)
      # save train and test data

      train_df.to_csv(os.path.join(self.train_dir,
                                   str(user_id) + ".csv"),
                      index=False)
      test_df.to_csv(os.path.join(self.test_dir,
                                  str(user_id) + ".csv"),
                     index=False)

  def collate_fn(self, batch):
    b0, b1, feat, user, target, user_target = zip(*batch)

    b0 = [torch.tensor(b) for b in b0]
    b1 = [torch.tensor(b) for b in b1]
    feat = [torch.tensor(f) for f in feat]

    b0_padded = torch.nn.utils.rnn.pad_sequence(b0,
                                                batch_first=True,
                                                padding_value=0)
    b1_padded = torch.nn.utils.rnn.pad_sequence(b1,
                                                batch_first=True,
                                                padding_value=0)
    feat_padded = torch.nn.utils.rnn.pad_sequence(feat,
                                                  batch_first=True,
                                                  padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target,
                                                    batch_first=True,
                                                    padding_value=-1)
    target_padded = target_padded.long()

    user = torch.tensor(user)
    user_target = torch.tensor(user_target)

    # attention mask
    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
      target_padded.size(1) + 1)

     # zero out the first row (= user)
    attn_mask[0, :] = 0

    mask = target_padded == -1

    # add user mask to the batch
    mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim=1)

    return b0_padded, b1_padded, feat_padded, mask, user, attn_mask, user_target, target_padded

  def collate_val_fn(self, batch):
    b0s, b1s, feats, users, targets, user_target = zip(*batch)

    b0 = [torch.tensor(b) for b0 in b0s for b in b0]
    b1 = [torch.tensor(b) for b1 in b1s for b in b1]
    feat = [torch.tensor(f) for fs in feats for f in fs]
    target = [t for ts in targets for t in ts]
    user = [u for us in users for u in us]
    user_target = [ut for uts in user_target for ut in uts]

    b0_padded = torch.nn.utils.rnn.pad_sequence(b0,
                                                batch_first=True,
                                                padding_value=0)
    b1_padded = torch.nn.utils.rnn.pad_sequence(b1,
                                                batch_first=True,
                                                padding_value=0)
    feat_padded = torch.nn.utils.rnn.pad_sequence(feat,
                                                  batch_first=True,
                                                  padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target,
                                                    batch_first=True,
                                                    padding_value=-1)
    target_padded = target_padded.long()

    user = torch.tensor(user)
    user_target = torch.tensor(user_target)

    # attention mask
    attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
      target_padded.size(1) + 1)

    # zero out the first row (= user)
    attn_mask[0, :] = 0

    # padding mask
    mask = target_padded == -1

    # add user mask to the batch
    mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim=1)
    
    return b0_padded, b1_padded, feat_padded, mask, user, attn_mask, user_target, target_padded

  def prepare_data(self):
    """
    Sample user_cnt users from the raw data.
    Split the data from each user into train and test.
    """
    if len(os.listdir(self.train_dir)) > 0:
      return
    else:
      valid_user_files = os.listdir(self.raw_data_dir)
      chosen_users = random.sample(valid_user_files, self.user_cnt)
      user_mapping = {user: i for i, user in enumerate(chosen_users)}
      self.split_data(user_mapping)
      print("Data prepared.")
  
  def setup(self, stage=None):
    """
    Load the data from the train and test folders.
    """
    if stage == "fit" or stage is None:      
      self.train_dataset = BigramPlusDataset(self.train_dir,
                                         self.max_seq_len,
                                         self.replace_prob,
                                         self.dataset_multiplier)
      
    self.val_dataset = BigramDatasetVal(self.test_dir,
                                        self.max_seq_len,
                                        self.val_user_cnt)

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=self.collate_fn,
                                       num_workers=self.num_workers,
                                       persistent_workers=True,
                                       pin_memory=True)

  def val_dataloader(self):
    assert self.val_batch_size == 1, "val_batch_size must be 1"
    return torch.utils.data.DataLoader(self.val_dataset,
                                       batch_size=self.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.collate_val_fn,
                                       num_workers=self.num_workers,
                                       pin_memory=True)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset,
                                       batch_size=self.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.collate_val_fn,
                                       num_workers=self.num_workers)