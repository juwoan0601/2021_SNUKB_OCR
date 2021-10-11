import torch
from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt

import os
import time
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import easydict
import datetime

opt = easydict.EasyDict({
    "batchSize" : 1,
    "nEpochs" : 200,
    "snapshots" : 20,                                                                           # 모델 저장 주기
    "lambda_pixel" : 100,                                                                       # loss_func_pix 가중치
    "data_dir" : "C:\\Users\\Hye-Young\\Desktop\\kb",
    "save_dir" : "C:\\Users\\Hye-Young\\Desktop\\kb\\Pix2PixTrans_model_ep200_batch128_data30060_initHe_defaultFont\\",  # 모델 저장 dir
    "log_dir" : "C:\\Users\\Hye-Young\\Desktop\\kb\\logs\\log_ep200_batch128_data30060_initHe_defaultFont_",      # logs 저장 dir
    "weight_init" : "He",                                                                   # Option : "Normal", "He"
    "train_A" : "images_preprocessed",                                                                       # 원본 이미지 폴더 이름
    "train_B" : "images_trans(default font)",                                                                 # 원본 이미지에 mapping 되는 이미지 폴더 이름
    "Img_num" : 1000                                                                             # 훈련에 사용할 이미지 수 (총 이미지 수 = 30060)
})

# Costum dataset 생성
class snukb_dataset(Dataset):
    def __init__(self, path2img, num, direction='origin2trans', transform=False):
        super().__init__()
        self.direction = direction
        self.num = num
        self.path2a = join(path2img, opt.train_A)
        self.path2b = join(path2img, opt.train_B)
        self.img_filenames = [x for x in listdir(self.path2a)[:self.num]]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        if self.direction == 'origin2trans':
            return a, b
        else:
            return b, a

    def __len__(self):
        return self.num