import kfbReader
import glob
from tqdm import tqdm
import pandas as pd

scale = 20

res = {'filename': [], 'height': [], 'width': []}

for fp in tqdm(glob.glob("./data/test/*")):
    f_name = fp.replace('./data/test/', '')
    read = kfbReader.reader()
    read.ReadInfo(fp, scale, False)
    w = read.getWidth()
    h = read.getHeight()
    res['filename'].append(f_name)
    res['height'].append(h)
    res['width'].append(w)

res_df = pd.DataFrame(res)
res_df.to_csv("./analyses/all_test_sizes.csv", index=False)
print("finished.")
