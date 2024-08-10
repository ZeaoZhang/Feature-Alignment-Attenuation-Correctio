import torch
from networks import unet3D

from utils import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

model = unet3D.unet3D(in_channels=1, out_channels=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch > 10 else 1.0)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)

for epoch in range(20):
    scheduler.step(epoch)
    print(epoch, optimizer.state_dict()['param_groups'][0]['lr'])


