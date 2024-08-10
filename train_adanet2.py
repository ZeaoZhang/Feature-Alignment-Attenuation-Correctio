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
from torch.autograd import Variable

import wandb
from dataset import get_dataLoader
from networks import adanet3D
from torch.cuda.amp import autocast as autocast, GradScaler
from metric import runningScore, averageMeter
from utils import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from loss_funcs import TransferLoss

def test(args, test_iter, model):
    running_metrics_st = runningScore()
    running_metrics_stct = runningScore()
    with torch.no_grad():
        for _ in range(len(test_iter)):
            ncs, acs, ctas = next(test_iter)
            ncs, acs, ctas = ncs.cuda(), acs.cuda(), ctas.cuda()
            ncs = ncs.type(torch.cuda.FloatTensor)
            acs = acs.type(torch.cuda.FloatTensor)
            ctas = ctas.type(torch.cuda.FloatTensor)

            outputs = model((ncs, ctas))
            if args.use_umap:
                acs_st = outputs["final_st"] + ncs
                acs_stct = outputs["final_ct"] + ncs
            else:
                acs_st = outputs["final_st"]
                acs_stct = outputs["final_ct"]

            running_metrics_st.update(acs_st.squeeze().cpu().numpy(), acs.squeeze().cpu().numpy())
            running_metrics_stct.update(acs_stct.squeeze().cpu().numpy(), acs.squeeze().cpu().numpy())

    return running_metrics_st.get_scores(), running_metrics_stct.get_scores()

def train(args, run):
    # define dataLoader
    dataLoader = get_dataLoader(args)
    train_iter = dataLoader["train"]
    test_iter = dataLoader["test"]
    epochs_without_improvement = 0
    # define network
    model = adanet3D.Adanet(in_channels=args.in_channels, out_channels=args.num_classes).cuda()
    disc = adanet3D.Discriminator(nums=args.d_nums).cuda()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.SGD(disc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    criterion_d = torch.nn.BCEWithLogitsLoss()
    criterion_tar = TransferLoss(args.target_loss)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.1, patience=5, verbose=True)

    # init_parameter
    train_loss_meter = averageMeter()
    train_loss_d_meter = averageMeter()
    train_loss_m_meter = averageMeter()
    best_evaluation = {"epoch": 0, "mse": 100, "pnsr": 0, "ssim": 0}
    clip_value = 0.01

    scaler = GradScaler()
    model.train()
    pbar = tqdm(range(args.num_epoch), leave=False)

    run.define_metric('mse_st', summary='min')
    for iter_num in pbar:
        for _ in range(len(train_iter)):
            ncs, acs, ctas = next(train_iter)

            ncs, acs, ctas = ncs.cuda(), acs.cuda(), ctas.cuda()
            ncs = ncs.type(torch.cuda.FloatTensor)
            acs = acs.type(torch.cuda.FloatTensor)
            ctas = ctas.type(torch.cuda.FloatTensor)

            one_label = Variable(torch.cuda.FloatTensor(ncs.size(0), 1).fill_(1.0), requires_grad=False)
            zero_label = Variable(torch.cuda.FloatTensor(ncs.size(0), 1).fill_(0.0), requires_grad=False)

            # generator
            with autocast():
                outputs = model((ncs, ctas))
                if args.use_umap:
                    loss_t = (criterion(outputs["final_st"]+ncs, acs) + criterion(outputs["final_ct"]+ncs, acs)) / 2
                else:
                    loss_t = (criterion(outputs["final_st"], acs) + criterion(outputs["final_ct"], acs)) / 2

                if args.use_disc:
                    if args.use_wgan:
                        loss_d = -torch.mean(disc(outputs["st"][-args.d_nums:]))
                    else:
                        # loss_d = criterion_d(disc(outputs["st"][-args.d_nums:]), one_label)
                        if args.up:
                            loss_d = criterion_d(disc(outputs["up_st"][-args.d_nums:]), one_label)
                        elif args.bi:
                            loss_d = (criterion_d(disc(outputs["up_st"][-args.d_nums:]), one_label) + criterion_d(disc(outputs["st"][-args.d_nums:]), one_label)) / 2
                        else:
                            loss_d = criterion_d(disc(outputs["st"][-args.d_nums:]), one_label)
                    loss = loss_t + loss_d * args.disc_lambda
                else:
                    loss = loss_t
                
                if args.use_metric:
                    loss_m = 0
                    for i in range(args.t_nums):
                        feat_st = outputs["st"][-(i+1)]
                        feat_ct = outputs["ct"][-(i+1)]
                        feat_st = F.avg_pool3d(feat_st, kernel_size=(feat_st.size(2), feat_st.size(3), feat_st.size(4))).squeeze()
                        feat_ct = F.avg_pool3d(feat_ct, kernel_size=(feat_ct.size(2), feat_ct.size(3), feat_ct.size(4))).squeeze()
                        loss_m += criterion_tar(feat_st, feat_ct)
                    loss_m /= args.t_nums

                    loss += loss_m * args.metric_lambda
                    train_loss_m_meter.update(loss_m.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            train_loss_meter.update(loss_t.item())

            # discriminator
            if args.use_disc:
                with autocast():
                    outputs = model((ncs, ctas))
                    if args.use_wgan:
                        loss_d = -torch.mean(disc(outputs["ct"][-args.d_nums:])) + torch.mean(disc(outputs["st"][-args.d_nums:]))
                    else:
                        # loss_d = (criterion_d(disc(outputs["ct"][-args.d_nums:]), one_label) + criterion_d(disc(outputs["st"][-args.d_nums:]), zero_label)) / 2
                        if args.up:
                            loss_d = (criterion_d(disc(outputs["up_ct"][-args.d_nums:]), one_label) + criterion_d(disc(outputs["up_st"][-args.d_nums:]), zero_label)) / 2
                        elif args.bi:
                            loss_ct = (criterion_d(disc(outputs["up_ct"][-args.d_nums:]), one_label) + criterion_d(disc(outputs["ct"][-args.d_nums:]), one_label)) / 2
                            loss_st = (criterion_d(disc(outputs["up_st"][-args.d_nums:]), zero_label) + criterion_d(disc(outputs["st"][-args.d_nums:]), zero_label)) / 2
                            loss_d = (loss_ct + loss_st) / 2
                        else:
                            loss_d = (criterion_d(disc(outputs["ct"][-args.d_nums:]), one_label) + criterion_d(disc(outputs["st"][-args.d_nums:]), zero_label)) / 2

                optimizer_d.zero_grad()
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
                # loss_d.backward()
                # optimizer_d.step()
                train_loss_d_meter.update(loss_d.item())

                if args.use_wgan:
                    for p in disc.parameters():
                        p.data.clamp_(-clip_value, clip_value)

        run.log({"train_loss": train_loss_meter.avg, "train_loss_d_meter": train_loss_d_meter.avg, "train_loss_m_meter": train_loss_m_meter.avg}, step=iter_num)
        pbar.set_description("train_loss %.4f; Best epoch %d, mse %.4f, pnsr %.4f, ssim %.4f" % (train_loss_meter.avg, best_evaluation["epoch"], 
                                    best_evaluation["mse"], best_evaluation["pnsr"], best_evaluation["ssim"]))
        train_loss_meter.reset()
        train_loss_d_meter.reset()
        train_loss_m_meter.reset()

        # 验证
        model.eval()
        scores_st, scores_stct = test(args, test_iter, model)
        run.log({"mse_st": scores_st["mse"], "pnsr_st": scores_st["pnsr"], "ssim_st": scores_st["ssim"]}, step=iter_num)
        run.log({"mse_stct": scores_stct["mse"], "pnsr_stct": scores_stct["pnsr"], "ssim_stct": scores_stct["ssim"]}, step=iter_num)

        # lr_scheduler.step(scores_st["mse"])
        # lr_scheduler_d.step(scores_st["mse"])

        curr_state = {
            "epoch": iter_num,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "evaluation": {
                "mse_st": scores_st["mse"], 
                "pnsr_st": scores_st["pnsr"], 
                "ssim_st": scores_st["ssim"],
                "mse_stct": scores_stct["mse"], 
                "pnsr_stct": scores_stct["pnsr"], 
                "ssim_stct": scores_stct["ssim"]
            },
        }

        if scores_st["mse"] < best_evaluation["mse"]:
            best_evaluation["epoch"] = iter_num
            best_evaluation["mse"] = scores_st["mse"]
            best_evaluation["pnsr"] = scores_st["pnsr"]
            best_evaluation["ssim"] = scores_st["ssim"]
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
    parser.add_argument('--root', type=str, nargs='?', default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/model/splits', help="dataset csv root")

    parser.add_argument('--in_channels', type=int, default=1, help="in_channels")
    parser.add_argument('--num_classes', type=int, default=1, help="class number")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num works")

    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight_decay")

    parser.add_argument('--num_epoch', type=int, default=50000, help="epoch number")
    parser.add_argument('--resume', type=bool, nargs='?', default=False, help="resume")
    parser.add_argument('--uid', type=str, nargs='?', default=None, help="uid")

    parser.add_argument('--job_type', type=str, nargs='?', default=None, help="job_type")
    parser.add_argument('--d_nums', type=int, default=1, help="d_nums")

    parser.add_argument('--use_disc', action='store_true', default=False)
    parser.add_argument('--use_umap', action='store_true', default=False)
    parser.add_argument('--use_wgan', action='store_true', default=False)
    parser.add_argument('--disc_lambda', type=float, default=1.0, help="disc_lambda")

    parser.add_argument('--use_metric', action='store_true', default=False)
    parser.add_argument('--target_loss', type=str, default="mmd", choices=['MMD', 'LMMD', 'CORAL', 'BNM'],
                            help="the loss function on target domain")
    parser.add_argument('--t_nums', type=int, default=1, help="t_nums")
    parser.add_argument('--metric_lambda', type=float, default=1.0, help="metric_lambda")

    parser.add_argument('--group_name', type=str, nargs='?', default='backbone', help="group name")

    parser.add_argument('--up', action='store_true', default=False)
    parser.add_argument('--bi', action='store_true', default=False)

    parser.add_argument('--save_dir', type=str, default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/adabet3D')
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
    # run = None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)



    train(args, run)




    