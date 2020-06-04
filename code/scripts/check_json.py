import pandas as pd
import glob
import json

res1 = './submissions/libra_x101_new_epoch_15_only_pos_T0.0_epoch_8_only_pos_T0.0/T0.3'
res2 = './submissions/libra_x101_new_epoch_15_only_pos_T0.0_epoch_8_only_pos_T0.0_avg/T0.3'

res1_list = []
res2_list = []
for fp in glob.glob(f"{res1}/*.json"):
    res1_list.extend(json.load(open(fp, "r")))
    res2_list.extend(json.load(open(fp.replace(res1, res2), "r")))

res1_df = pd.DataFrame(res1_list)
res2_df = pd.DataFrame(res2_list)
