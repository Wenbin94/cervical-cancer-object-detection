"""
This file generates annotations for ROI
"""
import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import json

RESIZE = 1024

all_sizes = pd.read_csv("./analyses/all_pos_sizes.csv")
all_sizes = all_sizes.rename(columns={"filename": "file"})
all_sizes['file'] = all_sizes['file'].str.replace(".kfb", "")

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
res_df['file'] = res_df['file'].str.replace(".json", "")

combined = pd.merge(res_df, all_sizes, on=['file'], how='left')

combined['x_scaled'] = combined.apply(
    lambda row: round(
        row['x'] * RESIZE / row['width'] if row['height'] > row['width'] else row['x'] * RESIZE / row['height']),
    axis=1)
combined['w_scaled'] = combined.apply(
    lambda row: round(
        row['w'] * RESIZE / row['width'] if row['height'] > row['width'] else row['w'] * RESIZE / row['height']),
    axis=1)
combined['y_scaled'] = combined.apply(
    lambda row: round(
        row['y'] * RESIZE / row['height'] if row['width'] > row['height'] else row['y'] * RESIZE / row['width']),
    axis=1)
combined['h_scaled'] = combined.apply(
    lambda row: round(
        row['h'] * RESIZE / row['height'] if row['width'] > row['height'] else row['h'] * RESIZE / row['width']),
    axis=1)

combined_roi = combined[combined['class'] == 'roi']
image_id = {}
i = 1
for file in combined_roi['file'].unique():
    image_id[file] = i
    i += 1

combined_roi_train, combined_roi_val = train_test_split(combined_roi, test_size=0.1)


def construct_ann_coco(combined_roi):
    dataset = {'info': {'description': 'cervical cytology'}, 'images': [], 'annotations': [],
               'categories': [{"id": 1, "name": "ROI"}]}
    ann_id = 1
    for file in combined_roi['file'].unique():
        tmp_img = {'file_name': file + ".jpg", "id": image_id[file]}
        ori_width = all_sizes[all_sizes['file'] == file]['width'].values[0]
        ori_height = all_sizes[all_sizes['file'] == file]['height'].values[0]
        if ori_width < ori_height:
            tmp_img['width'] = RESIZE
            tmp_img['height'] = int(RESIZE * ori_height / ori_width)
        else:
            tmp_img['height'] = RESIZE
            tmp_img['width'] = int(RESIZE * ori_width / ori_height)
        dataset['images'].append(tmp_img)

        roi_tmp_df = combined_roi[combined_roi['file'] == file].reset_index()
        for i in range(roi_tmp_df.shape[0]):
            tmp_ann = {"image_id": image_id[file], "id": ann_id}
            ann_id += 1
            tmp_ann['bbox'] = [float(roi_tmp_df.loc[i, 'x_scaled']),
                               float(roi_tmp_df.loc[i, 'y_scaled']),
                               float(roi_tmp_df.loc[i, 'w_scaled']),
                               float(roi_tmp_df.loc[i, 'h_scaled'])]
            tmp_ann['segmentation'] = [
                list(np.asarray(
                    [(x, y) for x in [tmp_ann['bbox'][0], tmp_ann['bbox'][0] + tmp_ann['bbox'][2]] for y in
                     [tmp_ann['bbox'][1], tmp_ann['bbox'][1] + tmp_ann['bbox'][3]]]).flatten())]

            tmp_ann['category_id'] = 1
            tmp_ann['iscrowd'] = 0
            tmp_ann['area'] = float(roi_tmp_df.loc[i, 'w_scaled']) * float(roi_tmp_df.loc[i, 'h_scaled'])
            dataset['annotations'].append(tmp_ann)
    return dataset


def construct_ann(combined_roi):
    dataset = []
    for file in combined_roi['file'].unique():
        tmp_dict = {'filename': file + ".jpg"}
        ori_width = all_sizes[all_sizes['file'] == file]['width'].values[0]
        ori_height = all_sizes[all_sizes['file'] == file]['height'].values[0]
        if ori_width < ori_height:
            tmp_dict['width'] = RESIZE
            tmp_dict['height'] = int(RESIZE * ori_height / ori_width)
        else:
            tmp_dict['height'] = RESIZE
            tmp_dict['width'] = int(RESIZE * ori_width / ori_height)
        tmp_dict['ann'] = {'bboxes': [], 'labels': []}
        roi_tmp_df = combined_roi[combined_roi['file'] == file].reset_index()

        for i in range(roi_tmp_df.shape[0]):
            bboxes = np.asarray([float(roi_tmp_df.loc[i, 'x_scaled'] / tmp_dict['width']),
                                 float(roi_tmp_df.loc[i, 'y_scaled'] / tmp_dict['height']),
                                 float(roi_tmp_df.loc[i, 'w_scaled'] / tmp_dict['width']),
                                 float(roi_tmp_df.loc[i, 'h_scaled'] / tmp_dict['height'])])
            tmp_dict['ann']['bboxes'].append(bboxes)
            tmp_dict['ann']['labels'].append(1)
        tmp_dict['ann']['bboxes'] = np.asarray(tmp_dict['ann']['bboxes'], dtype='float32')
        tmp_dict['ann']['labels'] = np.asarray(tmp_dict['ann']['labels'], dtype='int64')
        dataset.append(tmp_dict)
    return dataset


# pickle.dump(construct_ann(combined_roi_train), open("./data/annotations/roi_train.pkl", "wb"))
# pickle.dump(construct_ann(combined_roi_val), open("./data/annotations/roi_val.pkl", "wb"))
a = construct_ann_coco(combined_roi_train)
json.dump(construct_ann_coco(combined_roi_train), open("./data/annotations/roi_train.json", "w"))
json.dump(construct_ann_coco(combined_roi_val), open("./data/annotations/roi_val.json", "w"))
