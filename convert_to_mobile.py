import torch

import models

model = models.__dict__["resnet50q"]([1,2,4,8,32], 6).cuda()
checkpoint = torch.load("results/activityRecognition50/ckpt/model_latest.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
torch.jit.save(model, "model.pt")