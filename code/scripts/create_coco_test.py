"""
This file generates annotations (only images) for pos
"""
import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import json
from tqdm import tqdm
import re

test_pos = pd.read_csv("./data/posneg_predict.csv")
test_pos_file = test_pos[test_pos['class'] == 1]['file'].values
test_tiles = []

# for img in tqdm(test_pos_file):
#     for fp in glob.glob(f"./data/test_tiles/{img.replace('.jpg', '')}_*.jpg"):
#         test_tiles.append(fp.replace("./data/test_tiles/", ""))
# pickle.dump(test_tiles, open("./data/test_tiles.pkl", "wb"))


# test_tiles = pickle.load(open("./data/test_tiles.pkl", "rb")) # this is for faster rcnn r50 -> predicted all tiles

# this is for one model selecting positive tiles only
# tile_classify_test = pd.read_csv("./data/tile_posneg_predict.csv")
# test_tiles = tile_classify_test[tile_classify_test['class'] == 1]['tile_name'].values

# this is for splitting the model into two small vs large
tile_classify_test = pd.read_csv("./data/tile_posneg_predict.csv")
test_tiles = tile_classify_test[tile_classify_test['class'] == 1]['tile_name'].values
mode = 'large'
if mode == 'small':
    mode_images = pd.read_csv('./analyses/all_test_sizes_small.csv')['filename'].str.replace('.kfb', '').values
    test_tiles = [tile for tile in test_tiles if re.search("(.*_.*)_.*", tile).group(1) in mode_images]
elif mode == 'large':
    mode_images = pd.read_csv('./analyses/all_test_sizes_large.csv')['filename'].str.replace('.kfb', '').values
    test_tiles = [tile for tile in test_tiles if re.search("(.*_.*)_.*", tile).group(1) in mode_images]

dataset = {'info': {'description': 'cervical cytology'}, 'images': [], 'annotations': [],
           'categories': [{"id": 1, "name": "pos"}]}

img_id = 1
for img in test_tiles:
    tmp_img = {'file_name': img, "id": img_id, 'width': 1000, 'height': 1000}
    dataset['images'].append(tmp_img)
    img_id += 1

json.dump(dataset, open(f"./data/annotations/tile_test_{mode}_after_classify.json", "w"))
