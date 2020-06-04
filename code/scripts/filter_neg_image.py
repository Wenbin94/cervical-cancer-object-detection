"""
This script filters an existing prediction folder with more negative WSI.
"""
import pandas as pd
import glob
import json
import os

image_pred = pd.read_csv("./data/posneg_predict_0.6.csv")
image_pred['class'] = image_pred['confs'] > 0.5
image_pred['class'].sum()

image_neg = image_pred[image_pred['class'] == 0]['file'].str.replace('jpg', 'json').values

path = 'prediction_result/20191113_134607_libra_x101_new_epoch_15_only_pos/T0.0/'
new_path = 'prediction_result/result/'
if not os.path.isdir(new_path):
    os.system(f'mkdir -p {new_path}')

for fp in glob.glob(f"{path}/*.json"):
    if fp.replace(f'{path}/', '') in image_neg:
        json.dump([], open(fp.replace(path, new_path), "w"))
    else:
        os.system(f'cp {fp} {new_path}')
