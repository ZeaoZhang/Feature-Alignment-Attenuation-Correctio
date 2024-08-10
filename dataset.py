import os
import re
import cv2
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms, utils
from utils import ForeverDataIterator
import random

# class MyocardialPerfusionLoader(data.Dataset):
#     def __init__(self, root, fold=0, split='train'):
#         # 初始化参数
#         self.root = root
#         self.fold=fold
#         self.split=split
#         assert split in ['train', 'val'], f"split: {split} is not valid"
#         assert fold in [0, 1, 2, 3, 4], f"fold: {fold} is not valid"
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
        
#         if split == 'train':
#             dfs = []
#             for i in range(5):
#                 _df = pd.read_csv("{}/fold_{}.csv".format(root, i))
#                 if i != fold:
#                     dfs.append(_df)
#             df = pd.concat(dfs)
#         else:
#             df = pd.read_csv("{}/fold_{}.csv".format(root, fold))

#         self.dataList = []
#         for _, row in df.iterrows():
#             self.dataList.append(row)

#     def __getitem__(self, index):
#         # 'nc': nc_path, 'ac':ac_path, 'sla_nc': sla_nc_path, 'sla_ac': sla_ac_path, 
#         # 'vla_nc': vla_nc_path, 'vla_ac': vla_ac_path, 'sa_nc': sa_nc_path, 'sa_ac': sa_ac_path
#         data = self.dataList[index]

#         nc_paths = [data['nc'], data['sla_nc'], data['vla_nc'], data['sa_nc']]
#         ac_paths = [data['ac'], data['sla_ac'], data['vla_ac'], data['sa_ac']]
#         choice = random.randint(0, 3)

#         nc = self.loadData(nc_paths[choice]).unsqueeze(0)
#         ac = self.loadData(ac_paths[choice]).unsqueeze(0)
#         cta = self.loadData(data["cta_path"]).unsqueeze(0)

#         nc = nc * (nc < 1200) * (nc > 0)
#         nc = nc / 1200
#         ac = ac * (ac < 3600) * (ac > 0)
#         ac = ac / 3600
#         cta = cta * (cta < 4800) * (cta > 0)
#         cta = cta / 4800
#         # nc = (nc - nc.min()) / (nc.max() - nc.min())
#         # ac = (ac - ac.min()) / (ac.max() - ac.min())
#         # cta = (cta - cta.min()) / (cta.max() - cta.min())

#         # nc = nc * (nc < 4800) * (nc > 0)
#         # nc = nc / 4800
#         # ac = ac * (ac < 4800) * (ac > 0)
#         # ac = ac / 4800
#         # cta = cta * (cta < 4800) * (cta > 0)
#         # cta = cta / 4800
#         return nc, ac, cta #, data['studyUID']

#     def __len__(self):
#         return len(self.dataList)

#     def loadData(self, path):
#         img = np.load(path)
#         img = self.transform(img)

#         return img

class MyocardialPerfusionLoader(data.Dataset):
    def __init__(self, root, fold=0, split='train'):
        # 初始化参数
        self.root = root
        self.fold=fold
        self.split=split
        assert split in ['train', 'val'], f"split: {split} is not valid"
        assert fold in [0, 1, 2, 3, 4], f"fold: {fold} is not valid"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if split == 'train':
            dfs = []
            for i in range(5):
                _df = pd.read_csv("{}/fold_{}.csv".format(root, i))
                if i != fold:
                    dfs.append(_df)
            df = pd.concat(dfs)
        else:
            df = pd.read_csv("{}/fold_{}.csv".format(root, fold))

        self.dataList = []
        for _, data in df.iterrows():
            nc_paths = [data['nc'], data['sla_nc'], data['vla_nc'], data['sa_nc']]
            ac_paths = [data['ac'], data['sla_ac'], data['vla_ac'], data['sa_ac']]   
            for nc_path, ac_path in zip(nc_paths, ac_paths):
                self.dataList.append({'ac_path': ac_path, 'nc_path': nc_path, 'cta_path':data["cta_path"], 'studyUID': data['studyUID']})

    def __getitem__(self, index):
        # 'nc': nc_path, 'ac':ac_path, 'sla_nc': sla_nc_path, 'sla_ac': sla_ac_path, 
        # 'vla_nc': vla_nc_path, 'vla_ac': vla_ac_path, 'sa_nc': sa_nc_path, 'sa_ac': sa_ac_path
        data = self.dataList[index]

        nc = self.loadData(data["nc_path"]).unsqueeze(0)
        ac = self.loadData(data["ac_path"]).unsqueeze(0)
        cta = self.loadData(data["cta_path"]).unsqueeze(0)

        nc = nc * (nc < 1200) * (nc > 0)
        nc = nc / 1200
        ac = ac * (ac < 3600) * (ac > 0)
        ac = ac / 3600
        cta = cta * (cta < 4800) * (cta > 0)
        cta = cta / 4800
        # nc = (nc - nc.min()) / (nc.max() - nc.min())
        # ac = (ac - ac.min()) / (ac.max() - ac.min())
        # cta = (cta - cta.min()) / (cta.max() - cta.min())

        # nc = nc * (nc < 4800) * (nc > 0)
        # nc = nc / 4800
        # ac = ac * (ac < 4800) * (ac > 0)
        # ac = ac / 4800
        # cta = cta * (cta < 4800) * (cta > 0)
        # cta = cta / 4800
        return nc, ac, cta # , data['studyUID']

    def __len__(self):
        return len(self.dataList)

    def loadData(self, path):
        img = np.load(path)
        img = self.transform(img)

        return img

def get_dataLoader(args):
    loader_train = MyocardialPerfusionLoader(root=args.root, split="train", fold=args.fold)
    loader_test = MyocardialPerfusionLoader(root=args.root, split="val", fold=args.fold)
    iter_train = ForeverDataIterator(data.DataLoader(loader_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True))
    iter_test = ForeverDataIterator(data.DataLoader(loader_test, batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False))

    return {"train": iter_train, "test": iter_test}

if __name__ == '__main__':
    loader_train = MyocardialPerfusionLoader(root='/root/workspace/zhangzeao/Attenuation-Correction/zhangzeao/model/splits', split="train")
    iter_train = ForeverDataIterator(data.DataLoader(loader_train, batch_size=1, num_workers=1, shuffle=True, drop_last=True))

    # count = 0
    # for iteration, (ncs, acs, ctas) in enumerate(iter_train):
    #     print(iteration, ncs.size(), acs.size())
    #     count += np.sum(acs.numpy())
    #     print('{}/{}->{}%'.format(count, 8*(iteration+1), int(10000*count/(8*(iteration+1)))/100))

    print(len(iter_train))
    count = 0
    ratio = []
    for iteration in range(len(iter_train)):
        ncs, acs, ctas = next(iter_train)
        # ratio.append(acs.max().numpy()/ncs.max().numpy())
        # print(iteration, ncs.max().numpy(), acs.max().numpy(), acs.max().numpy()/ncs.max().numpy(), np.mean(ratio))
        print(iteration, ncs.max().numpy(), acs.max().numpy(), ctas.max().numpy())
        print(iteration, ncs.max().numpy(), acs.max().numpy(), ctas.max().numpy())
 


