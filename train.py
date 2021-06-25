# -*- coding:utf-8 -*-
import torch
from byol.byol_pytorch import BYOL
from funcs.MyDataset import MyDataset
from torch.utils.data import DataLoader
from model.model_3s import SDM
# from torchvision import models


EPOCH_NUM=500
BATCH_NUM=20

img_size = 64
feature_size = 571*4*4
projection_size = 571*4*4
hidden_size=1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=SDM().to(device)

learner = BYOL(
    net,
    image_size = img_size,
    hidden_layer = 'merge.maxpool',
    use_momentum = False,       # turn off momentum in the target encoder
    projection_size = feature_size,
    projection_hidden_size = hidden_size,

)


datasets=MyDataset(true_paths=["zongjiao/","dubo/"],false_paths=["ceshi/","un_sstv/"],random=False,img_mode="origin",imgs_run_part="total")
data_loader = DataLoader(dataset=datasets, batch_size=BATCH_NUM,num_workers=8,shuffle=True)


opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, img_size, img_size)


for epoch in range(EPOCH_NUM):
    correct=0
    epoch_total=0
    for i,data in enumerate(data_loader):
        loss = learner(data)#imgs
        print("Epoch:{}     batch:{}   loss:{}".format(epoch,i,loss.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item()<1:
            torch.save(net.state_dict(), './self-weights/weight-epoch{}-batch{}.pkl'.format(epoch,i))