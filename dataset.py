import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from collections import defaultdict
from time import time


def bigram_feat(session, max_len):
  """
  Bigram features for a single session.
  (keycode_1, keycode_2) -> [hl, il, pl, rl]
  """

  typing_features = []
  for idx, (tstamp, event, key) in enumerate(session):

    if event == 0:
      continue

    # get the release event
    for idx_rel, (tstamp_rel, event_rel,
                  key_rel) in enumerate(session[idx + 1:]):
      if event_rel == 0 and key_rel == key:

        hl = tstamp_rel - tstamp

        # next pressed key
        for idx_next, (tstamp_next, event_next,
                       key_next) in enumerate(session[idx + 1:]):
          if event_next == 1:

            il = tstamp_next - tstamp_rel
            pl = tstamp_next - tstamp

            # next release key
            for idx_next_rel, (tstamp_next_rel, event_next_rel,
                               key_next_rel) in enumerate(session[idx_next +
                                                                  1:]):
              if event_next_rel == 0 and key_next_rel == key_next:
                rl = tstamp_next_rel - tstamp_rel
                typing_features.append(
                    ((key, key_next),
                     [hl / 1000, il / 1000, pl / 1000, rl / 1000]))
                break
            break
        break

  # truncate to max_len
  if len(typing_features) > max_len:
    typing_features = typing_features[:max_len]

def new_bigram_feat(session, max_len):

  typing_features = []
  for idx, (press_ts, rel_ts, keycode) in enumerate(session):
      
      if idx == len(session) - 1:
        break
  
      press_ts_next, rel_ts_next, keycode_next = session[idx + 1]
  
      hl = rel_ts - press_ts
      il = press_ts_next - rel_ts
      pl = press_ts_next - press_ts
      rl = rel_ts_next - rel_ts
  
      typing_features.append(
          ((keycode, keycode_next), [hl / 1000, il / 1000, pl / 1000, rl / 1000]))

  # truncate to max_len
  if len(typing_features) > max_len:
    typing_features = typing_features[:max_len]

  return typing_features


def to_event_seq(session):
  # transform (timestamp_press, timestamp_release, keycode) to (timestamp, event, keycode)
  seq = []
  for idx, (timestamp_press, timestamp_release, keycode) in enumerate(session):
    seq.append((timestamp_press, 1, keycode))
    seq.append((timestamp_release, 0, keycode))

  # sort by timestamp
  seq = sorted(seq, key=lambda x: x[0])

  return seq



class BigramDataset(Dataset):

  def __init__(self,
               data_path,
               max_len=50,
               replace_prob=0.1,
               user_prob=0.5,
               dataset_multiplier=1):

    user_files = os.listdir(data_path)
    self.users_bigr = defaultdict(lambda: defaultdict(list))
    self.bigr_users = defaultdict(list)
    self.max_len = max_len
    self.sessions_feats = []
    self.sessions_bigrams_0 = []
    self.sessions_bigrams_1 = []
    self.users = []
    self.token_replace_prob = replace_prob
    self.user_replace_prob = user_prob

    for user_file in user_files:

      user_id = int(user_file.split(".")[0])
      user_df = pd.read_csv(os.path.join(data_path, user_file))

      for i, row in user_df.iterrows():

        raw_sesh = eval(row['SEQUENCE'])
        bi_feat_sesh = new_bigram_feat(raw_sesh, max_len=self.max_len)

        if bi_feat_sesh == []:
          continue

        bigrams = [bigram for bigram, feat in bi_feat_sesh]
        feats = [feat for bigram, feat in bi_feat_sesh]

        # access dict[user][bigram] -> list of features for that bigram
        for bigram, feat in bi_feat_sesh:
          self.users_bigr[user_id][bigram].append(feat)
          self.bigr_users[bigram].append(user_id)

        # dataset samples
        self.users.append(user_id)
        self.sessions_feats.append(feats)
        bigrams_0 = [bigram[0] for bigram in bigrams]
        bigrams_1 = [bigram[1] for bigram in bigrams]
        self.sessions_bigrams_0.append(bigrams_0)
        self.sessions_bigrams_1.append(bigrams_1)

    # set(list) on self.big_users
    self.bigr_users = {k: set(v) for k, v in self.bigr_users.items()}

    self.users = self.users * dataset_multiplier
    self.sessions_feats = self.sessions_feats * dataset_multiplier
    self.sessions_bigrams_0 = self.sessions_bigrams_0 * dataset_multiplier
    self.sessions_bigrams_1 = self.sessions_bigrams_1 * dataset_multiplier

  def __getitem__(self, index):

    impostor_user_prob = random.choices([0, 1],
                                       weights=(1 - self.user_replace_prob, self.user_replace_prob),
                                       k=1)[0]
    # init target
    target = torch.zeros(len(self.sessions_bigrams_0[index]))
    
    if impostor_user_prob == 1:
      # randomly sample a probability between 0 and self.replace_prob
      curr_replace_prob = random.uniform(0, self.token_replace_prob)

      # replace at least one token
      replace_cnt = max(int(curr_replace_prob * len(self.sessions_bigrams_0[index])), 1)

      # get indices of bigrams to replace
      to_replace_indices = random.sample(
          range(len(self.sessions_bigrams_0[index])), replace_cnt)

      replaced_indices = []

      for repl_idx in to_replace_indices:
        current_bigram = (self.sessions_bigrams_0[index][repl_idx],
                          self.sessions_bigrams_1[index][repl_idx])

        users_with_bigram = list(self.bigr_users[current_bigram] - set([self.users[index]]))

        if users_with_bigram == []:
          continue

        # get random user
        random_user = random.choice(users_with_bigram)
        # get same bigram from random user
        curr_bigram_rnd_feats = self.users_bigr[random_user][current_bigram]

        # get random feature
        random_feat = random.choice(curr_bigram_rnd_feats)

        # replace features from current bigram
        self.sessions_feats[index][repl_idx] = random_feat

        replaced_indices.append(repl_idx)

      if replaced_indices == []:
        impostor_user_prob = 0
      else:
        target[replaced_indices] = 1
  
    return self.sessions_bigrams_0[index], self.sessions_bigrams_1[
        index], self.sessions_feats[index], self.users[index], target, impostor_user_prob

  def __len__(self):
    return len(self.sessions_bigrams_0)


class BigramPlusDataset(Dataset):

  def __init__(self,
               data_path,
               max_len=50,
               replace_prob=[.25, .5, .25],
               dataset_multiplier=1):
    
    user_files = os.listdir(data_path)
    self.users_bigr = defaultdict(lambda: defaultdict(list))
    self.bigr_users = defaultdict(list)
    self.max_len = max_len
    self.sessions_feats = []
    self.sessions_bigrams_0 = []
    self.sessions_bigrams_1 = []
    self.users = []
    self.token_replace_prob = replace_prob
    self.users_dict = defaultdict(list)
    self.user_bigram_count = defaultdict(lambda: defaultdict(int))

    for user_file in user_files:

      user_id = int(user_file.split(".")[0])
      user_df = pd.read_csv(os.path.join(data_path, user_file))

      for i, row in user_df.iterrows():

        raw_sesh = eval(row['SEQUENCE'])

        # raw_sesh = to_event_seq(raw_sesh)

        bi_feat_sesh = new_bigram_feat(raw_sesh, max_len=self.max_len)

        if bi_feat_sesh == []:
          continue

        bigrams = [bigram for bigram, feat in bi_feat_sesh]
        feats = [feat for bigram, feat in bi_feat_sesh]

        # access dict[user][bigram] -> list of features for that bigram
        for bigram, feat in bi_feat_sesh:
          self.users_bigr[user_id][bigram].append(feat)
          self.bigr_users[bigram].append(user_id)

        # dataset samples
        self.users.append(user_id)
        self.sessions_feats.append(feats)
        bigrams_0 = [bigram[0] for bigram in bigrams]
        bigrams_1 = [bigram[1] for bigram in bigrams]
        self.sessions_bigrams_0.append(bigrams_0)
        self.sessions_bigrams_1.append(bigrams_1)
        self.users_dict[user_id].append((feats, bigrams_0, bigrams_1))

    # set(list) on self.big_users
    self.bigr_users = {k: set(v) for k, v in self.bigr_users.items()}

    self.user_set = set(self.users)
    self.users = self.users * dataset_multiplier
    self.sessions_feats = self.sessions_feats * dataset_multiplier
    self.sessions_bigrams_0 = self.sessions_bigrams_0 * dataset_multiplier
    self.sessions_bigrams_1 = self.sessions_bigrams_1 * dataset_multiplier


    self.user_bigram_count = {k: {k1: len(v1) for k1, v1 in v.items()} for k, v in self.users_bigr.items()}


  def __getitem__(self, index):

    impostor_user_prob = random.choices([0, 1, 2],
                                       weights=self.token_replace_prob,
                                       k=1)[0]
    # init target
    target = torch.zeros(len(self.sessions_bigrams_0[index]))

    if impostor_user_prob == 0:
      # positive sample -> the sequence remains the same
      return self.sessions_bigrams_0[index], self.sessions_bigrams_1[index],\
              self.sessions_feats[index], self.users[index], target, impostor_user_prob
    if impostor_user_prob == 1:
      # negative sample -> the sequence is replaced with a random one from other user
      # get random user
      # set - current user
      random_user = random.choice(list(self.user_set - set([self.users[index]])))

      # get random session from random user
      random_session = random.choice(self.users_dict[random_user])

      target = torch.ones(len(random_session[0]))

      return random_session[1], random_session[2], random_session[0],\
              self.users[index], target, impostor_user_prob
    if impostor_user_prob == 2:
      # get a random index to replace
      random_idx = random.randint(0, len(self.sessions_bigrams_0[index]) - 1)
      current_bigram = (self.sessions_bigrams_0[index][random_idx],
                        self.sessions_bigrams_1[index][random_idx])
      
      # get the same bigram from other user
      users_with_bigram = list(self.bigr_users[current_bigram] - set([self.users[index]]))

      if users_with_bigram == []:
        return self.sessions_bigrams_0[index], self.sessions_bigrams_1[index],\
                self.sessions_feats[index], self.users[index], target, 0
      
      random_user = random.choice(users_with_bigram)

      # get same bigram from random user
      curr_bigram_rnd_feats = self.users_bigr[random_user][current_bigram]

      # replace features from current bigram
      self.sessions_feats[index][random_idx] = random.choice(curr_bigram_rnd_feats)

      # replace with ones from random index to the end
      target[random_idx:] = 1

      return self.sessions_bigrams_0[index], self.sessions_bigrams_1[index],\
              self.sessions_feats[index], self.users[index], target, 1

  def __len__(self):
    return len(self.sessions_bigrams_0)


class BigramDatasetVal(Dataset):

  def __init__(self, data_path, max_len, val_user_cnt) -> None:
    user_files = os.listdir(data_path)

    # randomly selsct val_user_cnt users§
    user_files = random.sample(user_files, val_user_cnt)

    self.max_len = max_len

    self.sessions_feats = []
    self.sessions_bigrams_0 = []
    self.sessions_bigrams_1 = []

    self.users = []
    self.users_dict = defaultdict(list)
    

    user_set = set()

    for user_file in user_files:

      if not user_file.endswith(".csv"):
        continue

      user_id = int(user_file.split(".")[0])
      user_df = pd.read_csv(os.path.join(data_path, user_file))

      self.users.append(user_id)
      user_set.add(user_id)

      for i, row in user_df.iterrows():

        raw_sesh = eval(row['SEQUENCE'])

        # raw_sesh = to_event_seq(raw_sesh)

        bi_feat_sesh = new_bigram_feat(raw_sesh, max_len=self.max_len)

        if bi_feat_sesh == []:
          continue

        bigrams = [bigram for bigram, feat in bi_feat_sesh]
        feats = [feat for bigram, feat in bi_feat_sesh]

        bigrams_0 = [bigram[0] for bigram in bigrams]
        bigrams_1 = [bigram[1] for bigram in bigrams]

        # feats, bigrams_0, bigrams_1 for user_id
        self.users_dict[user_id].append((feats, bigrams_0, bigrams_1))

    self.unique_users = list(set(user_set))

  def __getitem__(self, index):

    user_id = self.users[index]

    pos_feats = []
    pos_bigrams_0 = []
    pos_bigrams_1 = []
    pos_targets = []

    user_id_genuine_samples = self.users_dict[user_id]
    for (feats, bigrams_0, bigrams_1) in user_id_genuine_samples:
      pos_feats.append(feats)
      pos_bigrams_0.append(bigrams_0)
      pos_bigrams_1.append(bigrams_1)

      target = torch.zeros(len(feats))
      pos_targets.append(target)

    neg_feats = []
    neg_bigrams_0 = []
    neg_bigrams_1 = []
    neg_targets = []

    # get random user
    for impostor in self.unique_users:
      if impostor != user_id:
        impostor_samples = self.users_dict[impostor]
        if impostor_samples == []:
          continue
        random_sample = random.choice(impostor_samples)
        neg_feats.append(random_sample[0])
        neg_bigrams_0.append(random_sample[1])
        neg_bigrams_1.append(random_sample[2])

        target = torch.ones(len(random_sample[0]))
        neg_targets.append(target)

    feats = pos_feats + neg_feats
    bigrams_0 = pos_bigrams_0 + neg_bigrams_0
    bigrams_1 = pos_bigrams_1 + neg_bigrams_1
    targets = pos_targets + neg_targets
    users = [user_id] * len(feats)
    users_targets = [0] * len(pos_feats) + [1] * len(neg_feats)

    return bigrams_0, bigrams_1, feats, users, targets, users_targets

  def __len__(self):
    return len(self.users)
