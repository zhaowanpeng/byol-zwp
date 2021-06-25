# -*- coding:utf-8 -*-
import math,random
from torchvision import transforms
from funcs.random_func import RandomFunc,random_func


ratate_free = transforms.RandomRotation((-30, 30), expand=True)
ratate_90 = transforms.RandomRotation((90,90), expand=True)
ratate_180 = transforms.RandomRotation((180,180), expand=True)
ratate_270 = transforms.RandomRotation((270,270), expand=True)

ratate_random_dict={
    "RF":(1,ratate_free),
    "R90":(1,ratate_90),
    "R180":(1,ratate_180),
    "R270":(1,ratate_270)
}
#旋转
random_rotater = RandomFunc(ratate_random_dict)
def random_rotate_func(img):
    func=random_rotater.get_random_func()
    img=func(img)
    return img
#random_rotate_func=random_rotater.get_random_func()


# 镜像
random_mirror = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

#裁剪
def random_crop(img):
    (w, h) = img.size
    sclr = random.uniform(0.7, 0.9)
    # sclr_h = random.uniform(0.7, 0.9)
    resize = transforms.RandomResizedCrop((round(h * sclr), round(w * sclr)),scale=(sclr,sclr))
    img = resize(img)
    return img

#伸缩
def random_resize(img):
    (w, h) = img.size
    sclr_w=random.uniform(0.7,0.9)
    sclr_h=random.uniform(0.7,0.9)
    resize = transforms.Resize((round(h*sclr_h), round(w*sclr_w)))
    img = resize(img)
    return img

def get_original(img):
    return img



