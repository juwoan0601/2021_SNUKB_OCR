# -*- coding: utf-8 -*-
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
# from torch.utils.tensorboard import SummaryWriter
import easydict
import datetime
from GeneratorUNet import GeneratorUNet
from DiscriminatorPatchGan import Discriminator
from Custom_dataset import snukb_dataset

'''
----- Tensorboard 를 위한 log_dir 의 저장 시간 ------
-> 시간 정보를 활용한 폴더 생성 (import datetime 추가)
'''
from pytz import timezone

UTC = datetime.datetime.now(timezone('UTC'))
date_time = UTC.astimezone(timezone('Asia/Seoul')).strftime("%Y_%m_%d_%H:%M:%S")

# plt 깔기
#

opt = easydict.EasyDict({
    "batchSize" : 24,
    "nEpochs" : 200,
    "snapshots" : 20,                                                                           # 모델 저장 주기
    "lambda_pixel" : 100,                                                                       # loss_func_pix 가중치
    "data_dir" : "./",
    "save_dir" : "./saved_model",  # 모델 저장 dir
    "log_dir" : "./logs",      # logs 저장 dir
    "weight_init" : "He",                                                                   # Option : "Normal", "He"
    "train_A" : "images_preprocessed",                                                                       # 원본 이미지 폴더 이름
    "train_B" : "images_trans(default font)",                                                                 # 원본 이미지에 mapping 되는 이미지 폴더 이름
    "Img_num" : 30060                                                                             # 훈련에 사용할 이미지 수 (총 이미지 수 = 30060)
})


# CUDA 확인
CUDA = torch.cuda.is_available()
print(CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device : {}".format(device))

# transforms 정의
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((256,256))
])

# 데이터셋 불러오기
path2img = opt.data_dir
train_ds = snukb_dataset(path2img, opt.Img_num, transform=transform)

# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=opt.batchSize, shuffle=True)
print("Batch Count : {}".format(len(train_dl)))

# 모델 로드
model_gen = GeneratorUNet().to(device)
model_dis = Discriminator().to(device)

# 가중치 초기화
def Normal_initialize(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

def He_initialize(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.kaiming_normal_(model.weight.data)

# 가중치 초기화 적용
if opt.weight_init == "Normal":
  model_gen.apply(Normal_initialize)
  model_dis.apply(Normal_initialize)
if opt.weight_init == "He":
  model_gen.apply(He_initialize)
  model_dis.apply(He_initialize)

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# patch 수
patch = (1,256//2**4,256//2**4)

# 최적화 파라미터
from torch import optim
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

# Optimizer
opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))


# 학습
model_gen.train()  # model 을 train mode 로 변경
model_dis.train()

batch_count = 0
start_time = time.time()

loss_hist = {'gen': [],
             'dis': []}

'''
----- Tensorboard 를 위한 writer 정의 ----------------
-> 시간 정보를 활용한 폴더 생성 (import datetime 추가)
'''
#writer = SummaryWriter(opt.log_dir)

for epoch in range(opt.nEpochs):
    for batch_idx, batch_data in enumerate(train_dl, start=1):
        a = batch_data[0]
        b = batch_data[1]
        ba_si = a.size(0)

        # real image
        real_a = a.to(device)  # 입력 이미지
        real_b = b.to(device)  # 글자체 변환 이미지

        # patch label
        real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
        fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

        # generator
        model_gen.zero_grad()

        fake_b = model_gen(real_a)  # 가짜 이미지 생성
        out_dis = model_dis(fake_b, real_b)  # 가짜 이미지와 글자체 변환 이미지 식별

        gen_loss = loss_func_gan(out_dis, real_label)  # (글자체 변환 이미지, real_label=1)
        pixel_loss = loss_func_pix(fake_b, real_b)  # (가짜 이미지, 글자체 변환 이미지)

        g_loss = gen_loss + opt.lambda_pixel * pixel_loss
        g_loss.backward()
        opt_gen.step()

        # discriminator
        model_dis.zero_grad()

        out_dis = model_dis(real_b, real_a)  # 진짜 이미지 식별
        real_loss = loss_func_gan(out_dis, real_label)

        out_dis = model_dis(fake_b.detach(), real_a)  # 가짜 이미지 식별
        fake_loss = loss_func_gan(out_dis, fake_label)

        d_loss = (real_loss + fake_loss) / 2.
        d_loss.backward()
        opt_dis.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
        '''
        ----- Tensorboard 에 매 Batch Count 마다 loss 저장하기 -----------------------------
        '''
        '''
        writer.add_scalars('Batch Count Loss_ep{}_batch{}_data{}_init{}'.format(opt.nEpochs, opt.batchSize, opt.Img_num,
                                                                                opt.weight_init),
                           {'G_Loss': g_loss.item(),
                            'D_Loss': d_loss.item()},
                           batch_count)
        # 그래프의 가로축 = batch_count, 세로 축 = g_loss.item(), d_loss.item()
        '''

        print('Epoch: %.0f, Batch_Count : %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
            epoch + 1, batch_count, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60))

    # 가중치 저장
    if (epoch + 1) % opt.snapshots == 0 or epoch == 0:
        path2models = opt.save_dir
        os.makedirs(path2models, exist_ok=True)
        path2weights_gen = os.path.join(path2models,
                                        'gen_ep{}_batch{}_data{}_init{}.pt'.format(epoch + 1, opt.batchSize,
                                                                                   opt.Img_num, opt.weight_init))
        path2weights_dis = os.path.join(path2models,
                                        'dis_ep{}_batch{}_data{}_init{}.pt'.format(epoch + 1, opt.batchSize,
                                                                                   opt.Img_num, opt.weight_init))

        torch.save(model_gen.state_dict(), path2weights_gen)
        torch.save(model_dis.state_dict(), path2weights_dis)

    '''
    ----- Tensorboard 에 매 epoch 마다 loss 저장하기 -----------------------------

    writer.add_scalars(
        'Epoch Loss_ep{}_batch{}_data{}_init{}'.format(opt.nEpochs, opt.batchSize, opt.Img_num, opt.weight_init),
        {'G_Loss': g_loss.item(),
         'D_Loss': d_loss.item()},
        epoch)
    # 그래프의 가로축 = epoch, 세로 축 = g_loss.item(), d_loss.item()
    '''

#writer.close()
# writer 가 더이상 필요하지 않으므로 닫아준다

# loss history
plt.figure(figsize=(10, 5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.show()