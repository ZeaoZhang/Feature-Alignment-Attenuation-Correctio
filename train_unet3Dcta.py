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

def test(args, test_iter, model):
    running_metrics = runningScore()
    with torch.no_grad():
        for _ in range(len(test_iter)):
            ncs, acs, ctas = next(test_iter)
            ncs, acs, ctas = ncs.cuda(), acs.cuda(), ctas.cuda()
            ncs = ncs.type(torch.cuda.FloatTensor)
            acs = acs.type(torch.cuda.FloatTensor)
            ctas = ctas.type(torch.cuda.FloatTensor)

            if args.use_cta:
                inputs = (ncs, ctas)
            else:
                inputs = ncs

            u_map = model(inputs)
            acs_p = ncs + u_map

            running_metrics.update(acs_p.squeeze().cpu().numpy(), acs.squeeze().cpu().numpy())

    return running_metrics.get_scores()


def train(args, run):
    # define dataLoader
    dataLoader = get_dataLoader(args)
    train_iter = dataLoader["train"]
    test_iter = dataLoader["test"]
    epochs_without_improvement = 0

    # define network
    # model = unet3D.unet3D(in_channels=args.in_channels, out_channels=args.num_classes).cuda()
    # if args.use_cta:
    model = unet3D.unet3Dcta().cuda()
    # else:
    # model = unet3D.unet3D(in_channels=args.in_channels, out_channels=args.num_classes).cuda()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    # init_parameter
    train_loss_meter = averageMeter()
    best_evaluation = {"epoch": 0, "mse": 100, "pnsr": 0, "ssim": 0}

    scaler = GradScaler()
    model.train()
    pbar = tqdm(range(args.num_epoch), leave=False)

    run.define_metric('mse', summary='min')
    # wandb.define_metric()
    for iter_num in pbar:
        for _ in range(len(train_iter)):
            ncs, acs, ctas = next(train_iter)
            ncs, acs, ctas = ncs.cuda(), acs.cuda(), ctas.cuda()

            ncs = ncs.type(torch.cuda.FloatTensor)
            acs = acs.type(torch.cuda.FloatTensor)
            ctas = ctas.type(torch.cuda.FloatTensor)
            if args.use_cta:
                # inputs = torch.cat([ncs, ctas], 1)
                inputs = (ncs, ctas)
            else:
                inputs = ncs

            with autocast():
                if args.use_umap:
                    u_map = model(inputs)
                    task_loss = criterion(ncs+u_map, acs)
                else:
                    gen_acs = model(inputs)
                    task_loss = criterion(gen_acs, acs)

            optimizer.zero_grad()
            # task_loss.backward()
            # optimizer.step()
            scaler.scale(task_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_meter.update(task_loss.item())

        run.log({"train_loss": train_loss_meter.avg}, step=iter_num)
        pbar.set_description("train_loss %.4f; Best epoch %d, mse %.4f, pnsr %.4f, ssim %.4f" % (train_loss_meter.avg, best_evaluation["epoch"], 
                                    best_evaluation["mse"], best_evaluation["pnsr"], best_evaluation["ssim"]))
        train_loss_meter.reset()

        # 验证
        model.eval()
        scores = test(args, test_iter, model)
        run.log({"mse_stct": scores["mse"], "pnsr_stct": scores["pnsr"], "ssim_stct": scores["ssim"]}, step=iter_num)

        curr_state = {
            "epoch": iter_num,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "evaluation": {
                "mse": scores["mse"],
                "pnsr": scores["pnsr"],
                "ssim": scores["ssim"],
            },
        }

        if scores["mse"] < best_evaluation["mse"]:
            best_evaluation["epoch"] = iter_num
            best_evaluation["mse"] = scores["mse"]
            best_evaluation["pnsr"] = scores["pnsr"]
            best_evaluation["ssim"] = scores["ssim"]
            torch.save(curr_state, os.path.join(args.checkpoints, "best_checkpoint_model.pth")) 
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(curr_state, os.path.join(args.checkpoints, "last_checkpoint_model.pth"))
        if epochs_without_improvement == 200:
            print('Early stopping at epoch {}...'.format(iter_num+1))
            break

        model.train()

# 知道为什么训都收敛了，因为每次都加载了模型，所以该学习率没用，MD
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AC')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--root', type=str, nargs='?', default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/split/five_fold', help="dataset csv root")

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
    parser.add_argument('--group_name', type=str, nargs='?', default='backbone', help="group name")
    parser.add_argument('--use_cta', action='store_true', default=False)
    parser.add_argument('--use_umap', action='store_true', default=False)
    parser.add_argument('--fold', type=int, default=0, help="dataste fold")   
    parser.add_argument('--save_dir', type=str, default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/Unet3D_cta')

    
    args = parser.parse_args()

    args.checkpoints = os.path.join(args.save_dir, args.group_name, args.job_type)
    if not os.path.exists(args.checkpoints):
      os.makedirs(args.checkpoints)

    if args.uid is None:
      args.uid = wandb.util.generate_id()
    if args.resume:
      args.resume = os.path.join(args.checkpoints, "last_checkpoint_model.pth")
    else:
      args.resume = None

    run = wandb.init(config=args, project="AC_final", name=args.uid, group=args.group_name, job_type=args.job_type, reinit=True, id=args.uid)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if not os.path.exists(args.checkpoints):
      os.makedirs(args.checkpoints)

    train(args, run)



    