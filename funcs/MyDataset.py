# -*- coding:utf-8 -*-
import os, torch, math
from torch.utils.data import Dataset
from torchvision import transforms
from funcs.img_read import read_img, filter_cannot_read
from funcs.img_transform import *
from funcs.dataset_helper import get_spilt_list

random_dict = {
    "RT": (1, random_rotate_func),
    "MR": (1, random_mirror),
    "RS": (5, random_resize),
    "CP": (5, random_crop),
    "OG": (15, get_original)  # 方法
}

class MyDataset(Dataset):
    """
    作用：将图像封装为可batch的对象
    参数：
        true_paths:正样本图像路径列表
        false_paths:负样本图像路径列表
        color:图片是否读取为彩色
        img_mode:图片采用那种处理策略
        filter:是否过滤无法读取的图片
        random:是否采用随机变换策略
        spilt:是否将读取的图像集分割为【train】【eval】=train/eval/else
    """

    def __init__(self, true_paths=[], false_paths=[], color=True, img_mode="3s_64", filter=False, random=False,
                 imgs_run_part="total"):

        self.color = color
        self.img_mode = img_mode
        self.true_paths = true_paths
        self.false_paths = false_paths
        self.random = random
        self.imgs = []
        self.imgs_run_part = imgs_run_part
        self.init_imgs_list()
        if filter:
            self.imgs = filter_cannot_read(self.imgs)

        if random:
            self.random_transformer = RandomFunc(random_dict)

    def random_transform(self, img):
        random_trans_func = self.random_transformer.get_random_func()
        img = random_trans_func(img)
        return img

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        img = read_img(imgpath, color=self.color)

        if self.random:
            img = self.random_transform(img)
        if self.img_mode == "3s_64":
            img = self.read_3s_img(img, len=64)
        if self.img_mode == "origin":
            pass#不做任何处理
        else:
            img = None
            print("sorry!author not do perfect")
            exit()
        img_splts = imgpath.split("/")
        filepath = "/".join(img_splts[:-1]) + "/"
        if filepath in self.true_paths:
            label = 1
        else:
            label = 0
        label = torch.tensor([label], dtype=torch.float)
        return img, label  # ,imgpath

    def __len__(self):
        return len(self.imgs)

    def init_imgs_list(self):
        paths = self.true_paths + self.false_paths
        path_num = len(paths)
        for i in range(path_num):
            path = paths[i]
            print("[{}/{}]initing imgs path ...     [path:{}]".format(i + 1, path_num, path))
            imgs = os.listdir(path)
            prcs_imgs = [path + img for img in imgs]
            # if self.spilt == "train":
            if self.imgs_run_part == "train":
                sets = get_spilt_list(prcs_imgs,train_proportion=0.7)
                prcs_imgs = sets[0]
            if self.imgs_run_part == "eval":
                sets = get_spilt_list(prcs_imgs,train_proportion=0.7)
                prcs_imgs = sets[1]
            self.imgs += prcs_imgs
        print("* init completed !     mode : {}".format(self.imgs_run_part))
        print("* imgs total num : {}".format(len(self.imgs)))

    def read_resize_img(self,img,len=64):
        img_processing = transforms.Compose([
            transforms.Resize((len, len)),
            transforms.ToTensor(),
        ])
        img=img_processing(img)
        return img

    def read_3s_img(self, img, len=64):

        (w, h) = img.size
        (min_len, min_name) = (h, "h") if h < w else (w, "w")
        if min_name == "h":
            sclr = len / h
            rs_h = len
            rs_w = math.ceil(w * sclr)
        else:
            sclr = len / w
            rs_w = len
            rs_h = math.ceil(h * sclr)
        tran_5 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((rs_h, rs_w)),
            transforms.ToTensor(),
            transforms.FiveCrop((len, len)),
        ])
        tranimgs = tran_5(img)
        if min_name == "h":
            img_mix = torch.torch.cat([tranimgs[0], tranimgs[1], tranimgs[4]], dim=0)
        else:
            img_mix = torch.torch.cat([tranimgs[0], tranimgs[2], tranimgs[4]], dim=0)

        return img_mix
