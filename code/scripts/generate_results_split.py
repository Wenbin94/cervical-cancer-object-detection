"""
0. run mmdet dist_test.py
1. generate_results.py
2. nms.py
"""
import json
import pandas as pd
import pickle
import re
from tqdm import tqdm
import numpy as np
import os


def nms(list1, thresh):
    """Pure Python NMS baseline."""
    n = len(list1)
    dets = np.zeros((n, 5))
    for i in range(n):
        dets[i][0] = int(list1[i]['x'])
        dets[i][1] = int(list1[i]['y'])
        dets[i][2] = int(list1[i]['w'])
        dets[i][3] = int(list1[i]['h'])
        dets[i][4] = float(list1[i]['p'])
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


small_THRESHOLD = 0.0
large_THRESHOLD = 0.0
size = 1000
overlap = 200
# path = '20191111_032245_epoch_8_only_pos'
# path = '20191113_134607_libra_x101_epoch_20_only_pos'
# path = '20191113_134607_libra_x101_new_epoch_15_only_pos'
path = '20191115_065112_small_e16_20191115_084745_large_e19_libra_x101_only_pos'

only_pos_tile = True

if not os.path.isdir(f"submissions/{path}/T{small_THRESHOLD}_{large_THRESHOLD}"):
    os.system(f"mkdir -p submissions/{path}/T{small_THRESHOLD}_{large_THRESHOLD}")

tile_classify_test = pd.read_csv("./data/tile_posneg_predict.csv")
test_tiles = tile_classify_test[tile_classify_test['class'] == 1]['tile_name'].values

small_res_file = './mmdetection/work_dirs/tile_small_libra_faster_rcnn_x101_64x4d_fpn_1x/20191115_065112_small_libra_x101_epoch_16_onlypos.bbox.json'
small_pred = json.load(open(small_res_file, "r"))

large_res_file = './mmdetection/work_dirs/tile_large_libra_faster_rcnn_x101_64x4d_fpn_1x/20191115_084745_large_libra_x101_epoch_19.bbox.json'
large_pred = json.load(open(large_res_file, "r"))

small_images = pd.read_csv('./analyses/all_test_sizes_small.csv')['filename'].str.replace('.kfb', '').values
small_test_tiles = [tile for tile in test_tiles if re.search("(.*_.*)_.*", tile).group(1) in small_images]
large_images = pd.read_csv('./analyses/all_test_sizes_large.csv')['filename'].str.replace('.kfb', '').values
large_test_tiles = [tile for tile in test_tiles if re.search("(.*_.*)_.*", tile).group(1) in large_images]

all_test_sizes = pd.read_csv("./analyses/all_test_sizes.csv")


def gen_results(test_tiles, pred, path, threshold):
    img_id = 1
    img_id_name = {}
    for img in test_tiles:
        tmp_img = {'file_name': img, "id": img_id, 'width': 1000, 'height': 1000}
        img_id_name[img_id] = img
        img_id += 1

    res = {'tile': [], 'img': [], 'tile_x': [], 'tile_y': [], 'x': [], 'y': [], 'w': [], 'h': [], 'p': []}
    for pos in pred:
        res['tile'].append(img_id_name[pos['image_id']])
        res['img'].append(re.search('(.*_.*)_.*', img_id_name[pos['image_id']]).group(1) + ".kfb")
        res['tile_x'].append(pos['bbox'][0])
        res['tile_y'].append(pos['bbox'][1])
        res['w'].append(pos['bbox'][2])
        res['h'].append(pos['bbox'][3])
        res['p'].append(pos['score'])
        res['x'].append(0)
        res['y'].append(0)
    res_df = pd.DataFrame(res)
    res_df_filtered = res_df[res_df['p'] > threshold]

    all_test_sizes = pd.read_csv("./analyses/all_test_sizes.csv")
    tile_loc_mapping = {}
    for img in tqdm(all_test_sizes['filename'].unique()):
        res_df_tmp = res_df_filtered[res_df_filtered['img'] == img].reset_index(drop=True)
        all_sizes_tmp = all_test_sizes[all_test_sizes['filename'] == img].reset_index(drop=True)
        width = all_sizes_tmp['width'].values[0]
        height = all_sizes_tmp['height'].values[0]
        # roi = H x W x C
        x, y = 0, 0
        counter = 0
        while y + size < height:
            x = 0
            img_prefix = img.replace(".kfb", "")
            while x + size < width:
                tile_name = f'{img_prefix}_{counter}.jpg'
                tile_loc_mapping[tile_name] = {'x': x, 'y': y}
                counter += 1
                x += min((size - overlap), width - (x + size))
            y += min((size - overlap), height - (y + size))

    res_df_filtered['x'] = res_df_filtered.apply(lambda row: row['tile_x'] + tile_loc_mapping[row['tile']]['x'], axis=1)
    res_df_filtered['y'] = res_df_filtered.apply(lambda row: row['tile_y'] + tile_loc_mapping[row['tile']]['y'], axis=1)

    res_df_filtered = res_df_filtered.sort_values(by=['img', 'tile'])

    pos_tiles = pd.read_csv("./data/tile_posneg_predict.csv")
    pos_tiles = pos_tiles[pos_tiles['class'] == 1]['tile_name'].values

    if only_pos_tile:
        res_df_filtered = res_df_filtered[res_df_filtered['tile'].isin(pos_tiles)]

    res_df_filtered['x'] = res_df_filtered['x'].astype(int)
    res_df_filtered['y'] = res_df_filtered['y'].astype(int)
    res_df_filtered['w'] = res_df_filtered['w'].astype(int)
    res_df_filtered['h'] = res_df_filtered['h'].astype(int)

    for img in tqdm(res_df_filtered['img'].unique()):
        preds = res_df_filtered[res_df_filtered['img'] == img].values
        preds_final = []
        for row in range(preds.shape[0]):
            preds_final.append(
                {'x': preds[row, 4], 'y': preds[row, 5], 'w': preds[row, 6], 'h': preds[row, 7], 'p': preds[row, 8]})
        preds_final_filtered = [preds_final[i] for i in nms(preds_final, 0.7)]
        json.dump(preds_final_filtered,
                  open(f"submissions/{path}/T{small_THRESHOLD}_{large_THRESHOLD}/{img.replace('kfb', 'json')}", "w"))
    return res_df_filtered


small_res_df = gen_results(small_test_tiles, small_pred, path, small_THRESHOLD)
large_res_df = gen_results(large_test_tiles, large_pred, path, large_THRESHOLD)

no_pos = set(all_test_sizes['filename'].unique()) - set(small_res_df['img'].unique()) - set(
    large_res_df['img'].unique())

for image in no_pos:
    json.dump([], open(f'./submissions/{path}/T{small_THRESHOLD}_{large_THRESHOLD}/{image.replace(".kfb", "")}.json', 'w'))
