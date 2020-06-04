import pandas as pd
import os
import kfbReader
from PIL import Image
import glob
import cv2 as cv
from tqdm import trange
import json
import numpy as np

scale = 20
size = 1000
overlap = 500
res = {}
keys = ['file', 'x', 'y', 'w', 'h', 'class']
for k in keys:
    res[k] = []
for fp in glob.glob("./data/labels/*.json"):
    tmp = json.load(open(fp, "r"))
    fp_name = fp.replace("./data/labels/", "")
    for item in tmp:
        res['file'].append(fp_name)
        for k in item.keys():
            res[k].append(item[k])
res_df = pd.DataFrame(res)

res_df_roi = res_df[res_df['class'] == 'roi'].reset_index(drop=True)
res_df_pos = res_df[res_df['class'] != 'roi'].reset_index(drop=True)


def generate_tile_img(res_df_roi):
    res_df_roi['file'] = res_df_roi['file'].str.replace(".json", ".kfb")
    for i in trange(res_df_roi.shape[0]):
        file = res_df_roi.loc[i, 'file']
        read = kfbReader.reader()
        read.ReadInfo(f"./data/train_pos/{file}", scale, False)
        roi = read.ReadRoi(int(res_df_roi.loc[i, 'x']), int(res_df_roi.loc[i, 'y']), int(res_df_roi.loc[i, 'w']),
                           int(res_df_roi.loc[i, 'h']),
                           scale)
        roi = np.asarray(roi)
        # roi = H x W x C
        x, y = 0, 0
        counter = 0
        while y + size < roi.shape[0]:
            x = 0
            while x + size < roi.shape[1]:
                tile = roi[y:y + size, x:x + size, :]
                cv.imwrite(f"./data/roi_tiles/{file.replace('.kfb', '')}_{counter}.jpg", tile)
                counter += 1
                x += min((size - overlap), roi.shape[1] - (x + size))
            y += min((size - overlap), roi.shape[0] - (y + size))


def generate_tile_ann(res_df_roi, res_df_pos):
    tile_ann_df = []
    for i in trange(res_df_roi.shape[0]):
        file = res_df_roi.loc[i, 'file']
        # roi = H x W x C
        x, y = 0, 0
        counter = 0

        while y + size < int(res_df_roi.loc[i, 'h']):
            x = 0
            while x + size < int(res_df_roi.loc[i, 'w']):
                x_ann_tile_start = x + int(res_df_roi.loc[i, 'x'])
                x_ann_tile_end = x + int(res_df_roi.loc[i, 'x']) + size
                y_ann_tile_start = y + int(res_df_roi.loc[i, 'y'])
                y_ann_tile_end = y + int(res_df_roi.loc[i, 'y']) + size
                tile_ann = res_df_pos[(res_df_pos['file'] == file) & (res_df_pos['x'] >= x_ann_tile_start) & (
                        res_df_pos['x'] + res_df_pos['w'] <= x_ann_tile_end) & (res_df_pos['y'] >= y_ann_tile_start) & (
                                              res_df_pos['y'] + res_df_pos['h'] <= y_ann_tile_end)].reset_index(
                    drop=True)
                tile_ann['x_tile'] = tile_ann['x'].map(lambda x_ann: x_ann - x_ann_tile_start)
                tile_ann['y_tile'] = tile_ann['y'].map(lambda y_ann: y_ann - y_ann_tile_start)
                tile_ann['tile_file'] = f'{file.replace(".json", "")}_{counter}.jpg'
                tile_ann_df.append(tile_ann)
                counter += 1
                x += min((size - overlap), int(res_df_roi.loc[i, 'w']) - (x + size))
            y += min((size - overlap), int(res_df_roi.loc[i, 'h']) - (y + size))
    return tile_ann_df


ann_df = pd.concat(generate_tile_ann(res_df_roi, res_df_pos))
ann_df.to_csv("./analyses/tile_coordinates.csv", index=False)
