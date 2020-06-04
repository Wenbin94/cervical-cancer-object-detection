"""
This file generates annotations for pos
"""
import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import json

seed = 42

combined = pd.read_csv("./analyses/tile_coordinates.csv")
mode = 'large'

if mode == 'small':
    df = pd.read_csv('./analyses/all_pos_sizes_small.csv')
    combined = combined[combined['file'].isin(df['filename'].str.replace('kfb', 'json').values)]
elif mode == 'large':
    df = pd.read_csv('./analyses/all_pos_sizes_large.csv')
    combined = combined[combined['file'].isin(df['filename'].str.replace('kfb', 'json').values)]


tile_id = {}
i = 1
for fp in glob.glob("./data/roi_tiles/*.jpg"):
    tile_id[fp.replace("./data/roi_tiles/", "")] = i
    i += 1

combined_train, combined_val = train_test_split(combined, test_size=0.1, shuffle=True, random_state=seed)


def construct_tile_ann_coco(combined_pos):
    dataset = {'info': {'description': 'cervical cytology'}, 'images': [], 'annotations': [],
               'categories': [{"id": 1, "name": "ROI"}]}
    ann_id = 1
    for file in combined_pos['tile_file'].unique():
        tmp_img = {'file_name': file, "id": tile_id[file], 'width': 1000, 'height': 1000}

        dataset['images'].append(tmp_img)

        pos_tmp_df = combined_pos[combined_pos['tile_file'] == file].reset_index()
        for i in range(pos_tmp_df.shape[0]):
            tmp_ann = {"image_id": tile_id[file], "id": ann_id}
            ann_id += 1
            tmp_ann['bbox'] = [float(pos_tmp_df.loc[i, 'x_tile']),
                               float(pos_tmp_df.loc[i, 'y_tile']),
                               float(pos_tmp_df.loc[i, 'w']),
                               float(pos_tmp_df.loc[i, 'h'])]
            tmp_ann['segmentation'] = [
                list(np.asarray(
                    [(x, y) for x in [tmp_ann['bbox'][0], tmp_ann['bbox'][0] + tmp_ann['bbox'][2]] for y in
                     [tmp_ann['bbox'][1], tmp_ann['bbox'][1] + tmp_ann['bbox'][3]]]).flatten())]

            tmp_ann['category_id'] = 1
            tmp_ann['iscrowd'] = 0
            tmp_ann['area'] = float(pos_tmp_df.loc[i, 'w']) * float(pos_tmp_df.loc[i, 'h'])
            dataset['annotations'].append(tmp_ann)
    return dataset


# a = construct_tile_ann_coco(combined_train)
json.dump(construct_tile_ann_coco(combined_train), open(f"./data/annotations/tile_train_{mode}.json", "w"))
json.dump(construct_tile_ann_coco(combined_val), open(f"./data/annotations/tile_val_{mode}.json", "w"))
