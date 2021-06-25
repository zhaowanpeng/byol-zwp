# -*- coding:utf-8 -*-
import torch
from byol.byol_pytorch import BYOL
# from torchvision import models

# a=torch.randn(20, 3)

# resnet = models.resnet50(pretrained=True)
img_size = 64
feature_size = 571*4*4
projection_size = 571*4*4
hidden_size=1024


from model.model_3s import SDM
model = SDM()


learner = BYOL(
    model,
    image_size = img_size,
    hidden_layer = 'merge.maxpool',
    use_momentum = False,       # turn off momentum in the target encoder
    projection_size = feature_size,
    projection_hidden_size = hidden_size,

)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, img_size, img_size)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)  # loss is calculate by BYOL module
    opt.zero_grad()
    loss.backward()
    opt.step()

# save your improved network
torch.save(model.state_dict(), './improved-net.pt')