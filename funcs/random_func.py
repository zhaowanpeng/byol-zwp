# -*- coding:utf-8 -*-
import random

def random_func(random_dict):
    assemble=[]
    for key in random_dict:
        assemble+=[key for i in range(random_dict[key][0])]
    focus=random.choice(assemble)
    return random_dict[focus][1]

#
class RandomFunc():
    def __init__(self,random_dict):
        self.assemble = []
        self.random_dict=random_dict
        self.focus=None
        for key in random_dict:
            self.assemble += [key for i in range(random_dict[key][0])]

    def get_random_func(self):
        self.focus = random.choice(self.assemble)
        return self.random_dict[self.focus][1]

    def get_focus_func(self,focus):
        return self.random_dict[focus][1]