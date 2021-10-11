#from google.colab import drive
#drive.mount('/content/gdrive')

from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt

import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import easydict
import datetime
from pytz import timezone
import torchvision.transforms.functional as TF
from torchvision.transforms import Pad
from tqdm import tqdm

from GeneratorUNet import GeneratorUNet

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="./images")
parser.add_argument("--save_dir", type=str, default="./saved_model")
parser.add_argument("--model_name", type=str, default="gen_ep190_batch128_data30060_initHe.pt")
parser.add_argument("--Test", type=str, default="")
parser.add_argument("--Img_num", type=int, default=6)
parser.add_argument("--StartImg", type=int, default=0)
parser.add_argument("--ImgSave_dir", type=str, default="./result")
parser.add_argument("--save_orig_img", action="store_true")
opt = parser.parse_args()

#device = 'cpu'
device=torch.device("cuda")

# 모델 로드
model_gen = GeneratorUNet().to(device)

# 가중치 불러오기
weights = torch.load(os.path.join(opt.save_dir, opt.model_name), map_location="cuda:0")
model_gen.load_state_dict(weights)

# evaluation mode
model_gen.eval()

# Costum dataset 생성
class snukb_dataset(Dataset):
    def __init__(self, pathTestimg, num, transform=False):
        super().__init__()
        self.num = num
        self.path = join(pathTestimg, opt.Test)
        filenames = [x for x in listdir(self.path)]
        self.img_filenames = filenames[opt.StartImg:self.num+opt.StartImg]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path, self.img_filenames[index])).convert('RGB')
        size = a.size  # 이미지의 원본 사이즈를 return

        if self.transform:
            a = self.transform(a)

        return a, size, self.img_filenames[index]

    def __len__(self):
        return self.num

# transforms 정의
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 데이터셋 불러오기
print("[GAN] load image data...")
pathTestimg = opt.data_dir
train_ds = snukb_dataset(pathTestimg, opt.Img_num, transform=transform)
print("[GAN] load image data finish!")

# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=opt.batchSize, shuffle=True)

# 가짜 이미지 생성
print("[GAN] Generate fake image...")
from image_separator import separateImage, mergeImage
startTime = time.time()
numImage = 0
for a, size, filename in tqdm(train_dl, desc='GAN Processing Image'):
    with torch.no_grad():
        sep_fake_img = []
        image_size = (size[0].item(), size[1].item())
        separatedImages = separateImage(a.numpy(), image_size)

    for img in separatedImages:
        img = torch.from_numpy(img)

        img = TF.resize(img, (256,256))
        img = img.view([1, 3, 256, 256])
        part_fake_img = model_gen(img.to(device)).cpu()
        sep_fake_img.append(part_fake_img)

    fake_img = torch.from_numpy(mergeImage(sep_fake_img)).to(device)
    real_img = a

    fake_image = to_pil_image(0.5 * fake_img[0] + 0.5).resize(image_size)
    filename = filename[0] if len(filename) >= 1 else filename
    fakeImg_dir = os.path.join(opt.ImgSave_dir, filename)
    fake_image.save(fakeImg_dir)
    numImage += 1
print("[GAN] Generate {0} fake image finish! {1} sec/image".format(numImage, int((time.time()-startTime)/(numImage))))
