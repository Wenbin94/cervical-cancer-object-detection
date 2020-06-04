"""
This file generates annotations for pos (classification)
"""
import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import json

combined = pd.read_csv("./analyses/tile_coordinates.csv")

tile_id = {}
i = 1
for fp in glob.glob("./data/roi_tiles/*.jpg"):
    tile_id[fp.replace("./data/roi_tiles/", "")] = i
    i += 1

train_pos, val_pos = train_test_split(combined['tile_file'].unique(), test_size=0.1)

all_neg = np.asarray(list(set(tile_id.keys()) - set(combined['tile_file'].unique())))
train_neg = np.random.choice(all_neg, train_pos.shape[0], replace=False)
val_neg = np.random.choice(np.setdiff1d(all_neg, train_neg), val_pos.shape[0], replace=False)

train_df = pd.concat(
    [pd.DataFrame({'tile_name': train_pos, 'class': 1}), pd.DataFrame({'tile_name': train_neg, 'class': 0})])
val_df = pd.concat(
    [pd.DataFrame({'tile_name': val_pos, 'class': 1}), pd.DataFrame({'tile_name': val_neg, 'class': 0})])

train_df.to_csv("./data/tile_classify_train.csv", index=False)
val_df.to_csv("./data/tile_classify_val.csv", index=False)

