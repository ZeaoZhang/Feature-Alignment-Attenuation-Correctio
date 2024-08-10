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


def train(args):
    # define dataLoader
    # args.batch_size = 1
    base_path = './vis_result/adanet+umap'
    if not os.path.exists(base_path):
       os.makedirs(base_path)
    dataLoader = get_dataLoader(args)
    test_iter = dataLoader["test"]

    model = getattr(adanet3D, args.model)(in_channels=args.in_channels, out_channels=args.num_classes).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["model_state"])
    # 验证
    model.eval()
    with torch.no_grad():
        running_metrics_st = runningScore()
        running_metrics_stct = runningScore()
        for _ in range(len(test_iter)):
            ncs, acs, ctas, studyUIDs = next(test_iter)
            # print(studyUID)
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
            for nc, ac, cta, umap_st, umap_stct, ac_st, ac_stct, studyUID in  zip(ncs, acs, ctas, outputs["final_st"], outputs["final_ct"], acs_st, acs_stct, studyUIDs):
                save_path = os.path.join(base_path, studyUID)
                if not os.path.exists(save_path):
                  os.makedirs(save_path)
                np.save(os.path.join(save_path, 'nc.npy'), nc.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'ac.npy'), ac.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'cta.npy'), cta.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'umap_st.npy'), umap_st.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'umap_stct.npy'), umap_stct.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'ac_st.npy'), ac_st.squeeze().cpu().numpy())
                np.save(os.path.join(save_path, 'ac_stct.npy'), ac_stct.squeeze().cpu().numpy())
    # run.log({"mse": scores["mse"], "pnsr": scores["pnsr"], "ssim": scores["ssim"]}, step=iter_num)
    print(running_metrics_st.get_scores(), running_metrics_stct.get_scores())

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
    parser.add_argument('--d_nums', type=int, default=2, help="d_nums")

    parser.add_argument('--use_disc', action='store_true', default=False)
    parser.add_argument('--use_umap', action='store_true', default=True)
    parser.add_argument('--use_wgan', action='store_true', default=False)
    parser.add_argument('--disc_lambda', type=float, default=0.25, help="disc_lambda")

    parser.add_argument('--use_metric', action='store_true', default=False)
    parser.add_argument('--target_loss', type=str, default="mmd", choices=['MMD', 'LMMD', 'CORAL', 'BNM'],
                            help="the loss function on target domain")
    parser.add_argument('--t_nums', type=int, default=5, help="t_nums")
    parser.add_argument('--metric_lambda', type=float, default=0.125, help="metric_lambda")

    parser.add_argument('--group_name', type=str, nargs='?', default='backbone', help="group name")

    parser.add_argument('--up', action='store_true', default=False)
    parser.add_argument('--bi', action='store_true', default=False)
    parser.add_argument('--model', type=str, default="Adanet4")    
    parser.add_argument('--n_critic', type=int, default=3, help="disc critic every n step")   # n_critic = 3

    parser.add_argument('--save_dir', type=str, default='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/adabet3D')
    args = parser.parse_args()

    args.resume = '/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/AC_final/checkpoints/adabet3D/best/4/best_checkpoint_model.pth'

    # run = wandb.init(config=args, project="AC_final_new", name=args.uid, group="unet3D", job_type=args.job_type, reinit=True, id=args.uid)
    # run = None

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train(args)



    