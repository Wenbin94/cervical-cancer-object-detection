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
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

seed = 42
BATCH_SIZE = 2**6
NUM_WORKERS = 6
LEARNING_RATE = 5e-5
LR_STEP = 5
LR_FACTOR = 0.1
NUM_EPOCHS = 15
LOG_FREQ = 20
RESIZE = 512
GB_STD = 12
WD = 0
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
        directory = './data/test_tiles/'
        filename = self.df['tile_name'].values[index]

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


labels = pd.read_csv("./data/tile_classify_test.csv") # change to new file
test_tiles = pickle.load(open("./data/test_tiles.pkl", "rb"))
labels = labels[labels['tile_name'].isin(test_tiles)]


model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.fc = nn.Linear(model.fc.in_features, 2)
model = nn.DataParallel(model)
checkpoint = torch.load("./models/tile_classify/tile_classify_11.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

test_dataset = ImageDataset(labels, mode='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

confs = inference(test_loader, model)

THRESHOLD = 0.99
labels['class'] = (confs[:, 0] - confs[:, 1] < THRESHOLD).astype(int)
labels['conf'] = confs[:, 1]
labels.to_csv("./data/tile_posneg_predict.csv", index=False)