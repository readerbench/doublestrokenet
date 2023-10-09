import os
import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def split_data(user_mapping, train_size, test_size, train_type_dir, raw_data_dir) -> None:
  """
  Split the data into train and test folders.
  """
  train_dir = os.path.join(train_type_dir, "train")
  test_dir = os.path.join(train_type_dir, "test")

  for user_file, user_id in user_mapping.items():
    user_df = pd.read_csv(os.path.join(raw_data_dir, user_file))
    # split data into train and test
    train_df, test_df = train_test_split(user_df,
                                          test_size=test_size,
                                          train_size=train_size,
                                          random_state=42)
    # save train and test data
    train_df.to_csv(os.path.join(train_dir,
                                  str(user_id) + ".csv"),
                    index=False)
    test_df.to_csv(os.path.join(test_dir,
                                str(user_id) + ".csv"),
                    index=False)
    
def split_data_benchmark(user_mapping, train_size, test_size, train_type_dir, raw_data_dir) -> None:
  """
  Split the data into train and test folders.
  """
  train_dir = os.path.join(train_type_dir, "train")
  test_dir = os.path.join(train_type_dir, "test")

  for user_file, user_id in user_mapping.items():
    user_df = pd.read_csv(os.path.join(raw_data_dir, user_file))
    # split data into train and test
    train_df, test_df = train_test_split(user_df,
                                          test_size=test_size,
                                          train_size=train_size,
                                          random_state=42)
    # save train and test data
    train_df.to_csv(os.path.join(train_dir,
                                  str(user_id) + ".csv"),
                    index=False)
    test_df.to_csv(os.path.join(test_dir,
                                str(user_id) + ".csv"),
                    index=False)

def prepare_data(cfg) -> None:
  valid_user_files = os.listdir(cfg.raw_data_path)
  chosen_users = random.sample(valid_user_files, cfg.pretrain.data.user_cnt + cfg.finetune.data.user_cnt)
  # split into pretrain and finetune
  pretrain_user_files = chosen_users[:cfg.pretrain.data.user_cnt]
  finetune_user_files = chosen_users[cfg.pretrain.data.user_cnt:]
  # create mappings
  pretrain_user_mapping = {user: i for i, user in enumerate(pretrain_user_files)}
  finetune_user_mapping = {user: i for i, user in enumerate(finetune_user_files)}
  # split data
  split_data(pretrain_user_mapping, cfg.pretrain.data.train_size, cfg.pretrain.data.test_size, cfg.pretrain.data.path, cfg.raw_data_path)
  split_data(finetune_user_mapping, cfg.finetune.data.train_size, cfg.finetune.data.test_size, cfg.finetune.data.path, cfg.raw_data_path)
  # save the pretrain and finetune user mappings
  with open(os.path.join(cfg.pretrain.data.path, "user_mapping.pkl"), "wb") as f:
    pickle.dump(pretrain_user_mapping, f)
  with open(os.path.join(cfg.finetune.data.path, "user_mapping.pkl"), "wb") as f:
    pickle.dump(finetune_user_mapping, f)
  
  print("Preatrain and finetune data prepared.")

def prepare_data_benchmark(cfg) -> None:
  """
  Used for testing on the TypeFormer benchmark data
  Not used anymore. 
  """
  
  test_users = os.listdir(cfg.finetune.data.path)

  train_users = os.listdir(cfg.pretrain.data.path)
  train_users = random.sample(train_users, cfg.pretrain.data.user_cnt)

  # create mappings
  train_user_mapping = {user: i for i, user in enumerate(train_users)}
  test_user_mapping = {user: i for i, user in enumerate(test_users)}

  # split data
  split_data_benchmark(train_user_mapping, cfg.pretrain.data.train_size, cfg.pretrain.data.test_size, cfg.pretrain.data.path_prc, cfg.pretrain.data.path)
  split_data_benchmark(test_user_mapping, cfg.finetune.data.train_size, cfg.finetune.data.test_size, cfg.finetune.data.path_prc, cfg.finetune.data.path)

  # save the pretrain and finetune user mappings
  with open(os.path.join(cfg.pretrain.data.path_prc, "user_mapping.pkl"), "wb") as f:
    pickle.dump(train_user_mapping, f)
  with open(os.path.join(cfg.finetune.data.path_prc, "user_mapping.pkl"), "wb") as f:
    pickle.dump(test_user_mapping, f)


def cleanup_data(train_type_dir, remove_mapping=True) -> None:
  """
  Delete the raw data folder.
  """
  # remove the mapping
  if remove_mapping:
    try:
      os.remove(os.path.join(train_type_dir, "user_mapping.pkl"))
    except:
      print("No user mapping found to be removed.")
  # remove the user files
  train_dir = os.path.join(train_type_dir, "train")
  test_dir = os.path.join(train_type_dir, "test")
  for file in os.listdir(train_dir):
    os.remove(os.path.join(train_dir, file))
    os.remove(os.path.join(test_dir, file))

