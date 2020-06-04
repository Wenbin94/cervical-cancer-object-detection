import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

seed = 42
BATCH_SIZE = 2**3
NUM_WORKERS = 6
LEARNING_RATE = 5e-5
LR_STEP = 5
LR_FACTOR = 0.1
NUM_EPOCHS = 15
LOG_FREQ = 20
RESIZE = 512
GB_STD = 12
WD = 0
THRESHOLD = 0.5
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True


class ImageDataset(Dataset):
    def __init__(self, dataframe, mode, hflip = False, vflip = False):
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode
        self.hflip = hflip
        self.vflip = vflip

        print(f"mode: {mode}, shape: {self.df.shape}")

        transforms_list = [
            transforms.Resize(RESIZE),
            transforms.CenterCrop(RESIZE)
        ]

        transforms_list.extend([
            transforms.ToTensor()
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        directory = './data/test_1024/'
        filename = self.df['file'].values[index]

        sample = Image.open(f'./{directory}/{filename}')
        assert sample.mode == 'RGB'

        if self.hflip:
            sample = F.hflip(sample)
        if self.vflip:
            sample = F.vflip(sample)
        image = self.transforms(sample)

        return image

    def __len__(self):
        return self.df.shape[0]


def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_confs = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            input_ = data

            output = torch.nn.functional.softmax(model(input_.to(device)))
            all_confs.append(output)
    confs = torch.cat(all_confs)

    return confs.cpu().numpy()


labels = pd.read_csv("./data/posneg_test.csv") # change to new file

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.fc = nn.Linear(model.fc.in_features, 2)
model = nn.DataParallel(model)
checkpoint = torch.load("./models/posneg/resnext_7.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

test_dataset = ImageDataset(labels, mode='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

confs = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', hflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

confs_hf = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', vflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

confs_vf = inference(test_loader, model)

test_dataset = ImageDataset(labels, mode='test', hflip=True, vflip=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

confs_hf_vf = inference(test_loader, model)
confs_all = np.asarray(
    [confs,
     confs_hf,
     confs_vf,
    confs_hf_vf])
confs_all = np.mean(confs_all, axis=0)
labels['class'] = (confs_all[:, 1] > 0.5).astype(int)
labels['confs'] = confs_all[:, 1]
labels.to_csv("./data/posneg_predict_0.6.csv", index=False)
