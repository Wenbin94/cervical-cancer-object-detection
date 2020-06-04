"""
0. run mmdet dist_test.py
1. generate_results.py
2. nms.py
"""
import numpy as np
import json
import glob
import os
from tqdm import tqdm, trange
import pandas as pd
from joblib import Parallel, delayed

def nms(list1, thresh, reverse=False):
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
    if not reverse:
        order = scores.argsort()[::-1]
    else:
        order = scores.argsort()

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


def nms_avg(list1, thresh):
    n = len(list1)
    dets = np.zeros((n, 6))
    for i in range(n):
        dets[i][0] = int(list1[i]['x'])
        dets[i][1] = int(list1[i]['y'])
        dets[i][2] = int(list1[i]['w'])
        dets[i][3] = int(list1[i]['h'])
        dets[i][4] = float(list1[i]['p'])
        dets[i][5] = float(list1[i]['weight'])

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    res = []
    while order.size > 0:
        i = order[0]
        # find other regions meeting IOU threshold
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds_tmp = np.where(ovr > thresh)[0]
        inds = np.where(ovr <= thresh)[0]

        inds_tmp = inds_tmp + 1
        inds_tmp = np.append(inds_tmp, 0)

        tmp_order = order[inds_tmp]
        tmp = dets[tmp_order, :]
        res.append({'x': np.round(np.average(tmp[:, 0], weights=tmp[:, 5])),
                    'y': np.round(np.average(tmp[:, 1], weights=tmp[:, 5])),
                    'w': np.round(np.average(tmp[:, 2], weights=tmp[:, 5])),
                    'h': np.round(np.average(tmp[:, 3], weights=tmp[:, 5])),
                    'p': np.max(tmp[:, 4])})

        order = order[inds + 1]

    return res


res_list = ['./submissions/20191113_134607_libra_x101_new_epoch_15_only_pos/T0.0/',
            './submissions/20191111_032245_epoch_8_only_pos/T0.0/',
            './submissions/20191115_065112_small_e16_20191115_084745_large_e19_libra_x101_only_pos/T0.0_0.0/',
            './submissions/result_cascade_teol512/']
weight_list = [1, 0.9, 0.9, 0.9]
file_list = os.listdir(res_list[0])
THRESHOLD = 0.3
path = 'libra_x101_new_e15_r50_e8_split_e16_e19_teol512'
if not os.path.isdir(f"submissions/{path}/T{THRESHOLD}"):
    os.system(f"mkdir -p submissions/{path}/T{THRESHOLD}")
if not os.path.isdir(f"submissions/{path}_avg/T{THRESHOLD}"):
    os.system(f"mkdir -p submissions/{path}_avg/T{THRESHOLD}")

json_list_all = {}


def process_nms(fp, path, THRESHOLD):
    json_list = [json.load(open(f"{res}/{fp}", "r")) for res in res_list]
    # scale scores
    for j in range(len(json_list)):
        for item in json_list[j]:
            item['p'] *= weight_list[j]
            item['weight'] = weight_list[j]

    # combine list
    combined = []
    for item in json_list:
        combined.extend(item)
    keep = nms(combined, THRESHOLD)
    combined_nms = [combined[i] for i in keep]
    combined_avg = nms_avg(combined, THRESHOLD)
    json.dump(combined_avg, open(f"./submissions/{path}_avg/T{THRESHOLD}/{fp}", "w"))
    json.dump(combined_nms, open(f"./submissions/{path}/T{THRESHOLD}/{fp}", "w"))


Parallel(n_jobs=-1)(delayed(process_nms)(file_list[i], path, THRESHOLD) for i in trange(len(file_list)))
#
# for fp in tqdm(file_list):
#     json_list = [json.load(open(f"{res}/{fp}", "r")) for res in res_list]
#     # scale scores
#     for j in range(len(json_list)):
#         for item in json_list[j]:
#             item['p'] *= weight_list[j]
#             item['weight'] = weight_list[j]
#
#     # combine list
#     combined = []
#     for item in json_list:
#         combined.extend(item)
#     combined_df = pd.DataFrame(combined)
#     keep = nms(combined, THRESHOLD)
#     combined_nms = [combined[i] for i in keep]
#     combined_nms_df = pd.DataFrame(combined_nms)
#     combined_avg = nms_avg(combined, THRESHOLD)
#     combined_avg_df = pd.DataFrame(combined_avg)
#     if len(keep) != combined_avg_df.shape[0]:
#         print(fp, len(keep), combined_avg_df.shape[0])
#
#     json.dump(combined_avg, open(f"./submissions/{path}_avg/T{THRESHOLD}/{fp}", "w"))
#     json.dump(combined_nms, open(f"./submissions/{path}/T{THRESHOLD}/{fp}", "w"))
#     # json_list_all[fp] = combined
