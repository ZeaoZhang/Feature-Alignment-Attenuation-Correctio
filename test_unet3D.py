import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

from torch.utils import data
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from dataset import get_dataLoader
from networks import unet3D
from torch.cuda.amp import autocast as autocast, GradScaler
from metric import runningScore, averageMeter
from utils import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR


def train(args):
    # define dataLoader
    # args.batch_size = 1
    base_path = './vis_result/unet+nc'
    if not os.path.exists(base_path):
       os.makedirs(base_path)
    dataLoader = get_dataLoader(args)
    test_iter = dataLoader["test"]

    model = unet3D.unet3D(in_channels=args.in_channels, out_channels=args.num_classes).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["model_state"])
    # 验证
    model.eval()
    with torch.no_grad():
        running_metrics = runningScore()
        for _ in range(len(test_iter)):
            ncs, acs, ctas, studyUIDs = next(test_iter)
            # print(studyUID)
            ncs, acs, ctas = ncs.cuda(), acs.cuda(), ctas.cuda()
            ncs = ncs.type(torch.cuda.FloatTensor)
            acs = acs.type(torch.cuda.FloatTensor)
            ctas = ctas.type(torch.cuda.FloatTensor)
            inputs = ncs

            u_map = model(inputs)
            acs_p = ncs + u_map
            running_metrics.update(acs_p.squeeze().cpu().numpy(), acs.squeeze().cpu().numpy())
            for nc, ac, cta, umap, ac_p, studyUID in  zip(ncs, acs, ctas, u_map, acs_p, studyUIDs):
                save_path = os.path.join(base_path, studyUID)
                if not os.path.exists(save_path):
                  os.makedirs(save_path)
                np.save(os.path.join(save_path, 'nc.npy'), nc.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'ac.npy'), ac.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'cta.npy'), cta.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'umap.npy'), umap.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'gen_ac.npy'), ac_p.squeeze().cpu().numpy())
    # run.log({"mse": scores["mse"], "pnsr": scores["pnsr"], "ssim": scores["ssim"]}, step=iter_num)
    scores = running_metrics.get_scores()
    print(scores)

# 知道为什么训都收敛了，因为每次都加载了模型，所以该学习率没用，MD
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AC')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--root', type=str, nargs='?', default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/model/splits', help="dataset csv root")

    parser.add_argument('--in_channels', type=int, default=1, help="in_channels")
    parser.add_argument('--num_classes', type=int, default=1, help="class number")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num works")

    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight_decay")
    parser.add_argument('--num_epoch', type=int, default=50000, help="epoch number")
    
    parser.add_argument('--resume', type=bool, nargs='?', default=False, help="resume")
    parser.add_argument('--uid', type=str, nargs='?', default=None, help="uid")
    parser.add_argument('--job_type', type=str, nargs='?', default=None, help="job_type")
    parser.add_argument('--use_cta', action='store_true', default=False)
    # parser.add_argument('--checkpoints', type=str, default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/unet3D')

    args = parser.parse_args()
    # if args.uid is None:
    #   args.uid = wandb.util.generate_id()
    # if args.resume:
    #   args.resume = os.path.join(args.checkpoints, "last_checkpoint_model.pth")
    # else:
    args.resume = '/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/unet3D/st/best/best_checkpoint_model.pth'

    # run = wandb.init(config=args, project="AC_final_new", name=args.uid, group="unet3D", job_type=args.job_type, reinit=True, id=args.uid)
    # run = None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # if not os.path.exists(args.checkpoints):
    #   os.makedirs(args.checkpoints)

    train(args)



    