# -*- coding:utf-8 -*-
import torch
from byol.byol_pytorch import BYOL
from torchvision import models

resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    use_momentum = False       # turn off momentum in the target encoder
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)  # loss is calculate by BYOL module
    opt.zero_grad()
    loss.backward()
    opt.step()

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')