import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

all_R, all_G, all_B = [], [], []
for fp in tqdm(glob.glob("./data/train_pos_1024/*")):
    img = np.array(Image.open(fp).convert('RGB'))
    all_R.append(img[:, :, 0].flatten())
    all_G.append(img[:, :, 1].flatten())
    all_B.append(img[:, :, 2].flatten())

all_R = np.concatenate(all_R)
all_G = np.concatenate(all_G)
all_B = np.concatenate(all_B)
print(np.mean(all_R))
print(np.mean(all_G))
print(np.mean(all_B))

print(np.std(all_R))
print(np.std(all_G))
print(np.std(all_B))