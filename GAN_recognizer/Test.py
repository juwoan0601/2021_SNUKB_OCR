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
import copy
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
#weights = torch.load(os.path.join(opt.save_dir, opt.model_name), map_location=torch.device('cpu'))
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
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((256,256))
])

# 데이터셋 불러오기
print("[GAN] load image data...")
pathTestimg = opt.data_dir
train_ds = snukb_dataset(pathTestimg, opt.Img_num, transform=transform)
print("[GAN] load image data finish!")

# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=opt.batchSize, shuffle=True)

# 가짜 이미지 생성
# with torch.no_grad():
#     for a in train_dl:
#         fake_imgs = model_gen(a.to(device))
#         real_imgs = a
#         break

#fake_imgs = []
#real_imgs = []
#image_size = []
#filenames = []

print("[GAN] Generate fake image...")
startTime = time.time()
for a, size, filename in tqdm(train_dl, desc='GAN Processing Image'):
    with torch.no_grad():
        fake_imgs = model_gen(a.to(device))
        #real_imgs.append(a)
        #image_size.append((size[0].item(), size[1].item()))  # image 의 원본 size 를 저장해 둔 list 생성
        image_size = (size[0].item(), size[1].item())
        #filenames.append(filename[0])
        filenames = filename[0]
    torch.cuda.empty_cache()

    # 가짜 이미지 시각화
    #fake_image = to_pil_image(0.5*fake_imgs[i][0]+0.5).resize(image_size[i])  # Resize image (fit original image)
    fake_image = to_pil_image(0.5*fake_imgs[0]+0.5).resize(image_size)
    plt.imshow(fake_image)
    fakeImg_dir = os.path.join(opt.ImgSave_dir, filenames)
    fake_image.save(fakeImg_dir)
    # print(fake_imgs[i].shape)
    plt.axis('off')
print("[GAN] Generate {0} fake image finish! {1} sec/image".format(len(fake_imgs), int((time.time()-startTime)/(len(fake_imgs)))))
print("[GAN] Image size: ", image_size[0])

'''
# 가짜 이미지 시각화
plt.figure(figsize=(15, 80))

for i in range(0,opt.Img_num):
    plt.subplot(opt.Img_num,2,2*i+1)
    fake_image = to_pil_image(0.5*fake_imgs[i][0]+0.5).resize(image_size[i])  # Resize image (fit original image)
    plt.imshow(fake_image)
    fakeImg_dir = os.path.join(opt.ImgSave_dir, filenames[i])
    fake_image.save(fakeImg_dir)
    # print(fake_imgs[i].shape)
    plt.axis('off')
    plt.subplot(opt.Img_num,2,2*i+2)
    real_image = to_pil_image(0.5*real_imgs[i][0]+0.5).resize(image_size[i])  # Resize image (fit original image)
    plt.imshow(real_image)                           # transform 에서 Normalize 해 주었으므로 다시 원래대로 돌려서 출력해야 함
    # realImg_dir = os.path.join(opt.ImgSave_dir, filenames[i])
    # real_image.save(realImg_dir)
    plt.axis('off')
'''