# -*- coding:utf-8 -*-
import random
#切割为数据集和训练集
def get_spilt_list(imgs,seed_num=10,train_proportion=0.7):
    try:
        total_num = len(imgs)
        train_num = round( train_proportion*total_num )
        random.seed(seed_num)
        train_imgs = random.sample(imgs,train_num)
        random.seed(None)
        evalute_imgs = [item for item in train_imgs if item == "B"]
        print(len(evalute_imgs))
        print(evalute_imgs)
        #list( set(imgs) - set(train_imgs) )
        return (train_imgs,evalute_imgs)
    except:
        return None
# a=["A"]*100
# b=["B"]*10
# # c=a+b
# # d=get_spilt_list(c)
# # print(d[0])
# # print(len(d[0]))
# # print(d[1])
# train_imgs = random.sample(b,round(0.7*len(b)))
# print(train_imgs)