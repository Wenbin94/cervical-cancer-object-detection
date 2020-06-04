#%%

import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import numpy as np  # linear algebra
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm, trange, tnrange
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

#%%

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

#%%

class ImageDataset(Dataset):
    def __init__(self, dataframe, mode):
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        print(f"mode: {mode}, shape: {self.df.shape}")

        transforms_list = [
            transforms.Resize(RESIZE),
            transforms.CenterCrop(RESIZE)
        ]

        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=(0, 360), translate=(0.1, 0.1),
                                            scale=(0.8, 1.2),
                                            resample=Image.BILINEAR)
                ])
            ])

        transforms_list.extend([
            transforms.ToTensor()
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        directory = './data/train_all_1024/'
        filename = self.df['file'].values[index]

        sample = Image.open(f'./{directory}/{filename}')
        assert sample.mode == 'RGB'
        image = self.transforms(sample)
        target = self.df['class'].values[index]
        
        return image, target

    def __len__(self):
        return self.df.shape[0]

#%%

class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n = 1) :
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#%%

def train(train_loader, model, criterion, optimizer, epoch, logging=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_f1 = AverageMeter()
    avg_auc = AverageMeter()
    avg_recall = AverageMeter()

    model.train()
    num_steps = len(train_loader)

    end = time.time()
    lr_str = ''

    for i, (input_, targets) in enumerate(train_loader):
        if i >= num_steps:
            break

        output = model(input_.to(device))
        loss = criterion(output, targets.to(device))

        predicts = (output.data[:,0] - output.data[:,1] < THRESHOLD).int()
        predicts = predicts.cpu().numpy()
        targets = targets.cpu().numpy()
        avg_f1.update(f1_score(targets, predicts))
        avg_recall.update(recall_score(targets, predicts))
        if not ((targets == 1).all() or (predicts == 1).all()):
            avg_auc.update(roc_auc_score(targets, output.detach().cpu().numpy()[:,1]))

        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if logging and i % LOG_FREQ == 0:
            print(f'{epoch} [{i}/{num_steps}]\t'
                  f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'F1 {avg_f1.val:.4f} ({avg_f1.avg:.4f})\t'
                  f'recall {avg_recall.val:.4f} ({avg_recall.avg:.4f})\t'
                  f'auc {avg_auc.val:.4f} ({avg_auc.avg:.4f})\t' + lr_str)
            sys.stdout.flush()

    print(f' * average F1 on train {avg_f1.avg:.4f}')
    print(f' * average Recall on train {avg_recall.avg:.4f}')
    print(f' * average AUC on train {avg_auc.avg:.4f}')
    if epoch > 5:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, f"./models/posneg/resnext_{epoch}.pth")
    return avg_recall.avg

#%%

def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_predicts, all_confs, all_targets = [], [], []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_, target = data

            output = model(input_.to(device))
            all_confs.append(output)
            predicts = (output.data[:,0] - output.data[:,1] < THRESHOLD).int()
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, confs, targets

#%%

def validate(val_loader, model):
    predicts, confs, targets = inference(val_loader, model)
    predicts = predicts.cpu().numpy()
    confs = confs.cpu().numpy()
    targets = targets.cpu().numpy()
    
    f1 = f1_score(targets, predicts)
    recall = recall_score(targets, predicts)
    if not ((targets == 1).all() or (predicts == 1).all()):
        auc = roc_auc_score(targets, confs[:,1])

    print(f"val real f1: {f1:.4f}")
    print(f"val real recall: {recall:.4f}")
    print(f"val real auc {auc:.4f}")

    return recall

#%%

def train_loop(epochs, train_loader, val_loader, model, criterion, optimizer):
    train_res = []
    val_res = []
    for epoch in tnrange(1, epochs + 1):
        print(f"learning rate: {lr_scheduler.get_lr()}")
        start_time = time.time()
        train_score = train(train_loader, model, criterion,
                            optimizer, epoch, logging=True)

        train_res.append(train_score)
        lr_scheduler.step()

        val_recall = validate(val_loader, model)
        val_res.append(val_recall)
        print(f"epoch {epoch} validation recall: {val_recall:.4f}\n")

    return np.asarray(train_score), np.asarray(val_recall)

#%%

labels = pd.read_csv("./data/posneg.csv") # change to new file
# labels = labels.sample(n=200)

#%%

train_df, val_df = train_test_split(labels,
                                    stratify=labels['class'],
                                    test_size=0.1,
                                    random_state=seed, shuffle = True)

#%%

train_dataset = ImageDataset(train_df, mode='train')
val_dataset = ImageDataset(val_df, mode='val')

#%%

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True,
                          num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

#%%

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.fc = nn.Linear(model.fc.in_features, 2)

#%%

model = nn.DataParallel(model)

#%%

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=LR_STEP,
                                               gamma=LR_FACTOR)

#%%

train_res, val_res = train_loop(NUM_EPOCHS, train_loader, val_loader, model,
                                criterion, optimizer)

#%%

train_res, val_res = train_loop(NUM_EPOCHS, train_loader, val_loader, model,
                                criterion, optimizer)

#%%


