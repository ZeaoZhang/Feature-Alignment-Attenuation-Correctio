import torch.nn as nn
import torch.nn.functional as F
import torch
from .cbam import *

class unetConv3D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv3D, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, 3, 1, 1), nn.BatchNorm3d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, 3, 1, 1), nn.BatchNorm3d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp3D(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp3D, self).__init__()
        self.conv = unetConv3D(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        padding = (0, outputs2.size()[4] - inputs1.size()[4], 
                0, outputs2.size()[3] - inputs1.size()[3], 
                0, outputs2.size()[2] - inputs1.size()[2])
        
        outputs1 = F.pad(inputs1, padding)
        outputs = torch.cat([outputs1, outputs2], 1)
        return self.conv(outputs)

class Adanet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128, 256], is_deconv=True,  is_batchnorm=True):
        super(Adanet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        
        # ST feature extract
        self.conv1_st = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_st = nn.MaxPool3d(kernel_size=2)
        self.conv2_st = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_st = nn.MaxPool3d(kernel_size=2)
        self.conv3_st = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_st = nn.MaxPool3d(kernel_size=2)
        self.conv4_st = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_st = nn.MaxPool3d(kernel_size=2)
        self.center_st = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # STCT feature extract
        self.conv1_ct = unetConv3D(in_channels*2, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)

    def forward(self, inputs):
        ncs, ctas = inputs

        # feature extract
        # ST
        conv1_st = self.conv1_st(ncs)
        maxpool1_st = self.maxpool1_st(conv1_st)
        conv2_st = self.conv2_st(maxpool1_st)
        maxpool2_st = self.maxpool2_st(conv2_st)
        conv3_st = self.conv3_st(maxpool2_st)
        maxpool3_st = self.maxpool3_st(conv3_st)
        conv4_st = self.conv4_st(maxpool3_st)
        maxpool4_st = self.maxpool4_st(conv4_st)
        center_st = self.center_st(maxpool4_st)

        # STCT
        conv1_ct = self.conv1_ct(torch.cat([ncs, ctas], 1))
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)

        # upsample
        up4_st = self.up_concat4(conv4_st, center_st)
        up3_st = self.up_concat3(conv3_st, up4_st)
        up2_st = self.up_concat2(conv2_st, up3_st)
        up1_st = self.up_concat1(conv1_st, up2_st)
        final_st = self.final(up1_st)

        up4_ct = self.up_concat4(conv4_ct, center_ct)
        up3_ct = self.up_concat3(conv3_ct, up4_ct)
        up2_ct = self.up_concat2(conv2_ct, up3_ct)
        up1_ct = self.up_concat1(conv1_ct, up2_ct)
        final_ct = self.final(up1_ct)

        output = {
            "st": [conv1_st, conv2_st, conv3_st, conv4_st, center_st],
            "up_st": [up1_st, up2_st, up3_st, up4_st, center_st], 
            "final_st": final_st,

            "ct": [conv1_ct, conv2_ct, conv3_ct, conv4_ct, center_ct],
            "up_ct": [up1_ct, up2_ct, up3_ct, up4_ct, center_ct], 
            "final_ct": final_ct,
        }

        return output

class Discriminator(nn.Module):
    def __init__(self, nums=2, filters=[16, 32, 64, 128, 256],  in_channels=8192, is_batchnorm=True):
        super(Discriminator, self).__init__()
        self.nums = nums
        self.lens = 5
        self.is_batchnorm = is_batchnorm

        blocks, blocks_f = [], []
        for index in range(self.lens-1):
            blocks.append(nn.Sequential(nn.Conv3d(filters[index], filters[index+1], 3, 1, 1), nn.BatchNorm3d(filters[index+1]), nn.ReLU(), nn.MaxPool3d(kernel_size=2)))
            blocks_f.append(nn.Sequential(nn.Conv3d(filters[index+1]*2, filters[index+1], 3, 1, 1), nn.BatchNorm3d(filters[index+1]), nn.ReLU()))

        self.blocks = nn.ModuleList(blocks)
        self.blocks_f = nn.ModuleList(blocks_f)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(in_channels, 1),
            # nn.Sigmoid(),
        )
# ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]), torch.Size([
# 16, 256, 2, 4, 4]))
# up_ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]))
# st.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]), torch.Size([
# 16, 256, 2, 4, 4]))
# ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]))


    def forward(self, inputs):
        for index in range(self.nums):
            if index == 0:
                x = inputs[0]
            else:
                x = torch.cat([x, inputs[index]], 1)
                x = self.blocks_f[self.lens-self.nums+index-1](x)
            # 如果到达最后一层，就合并了就行
            if index < self.nums - 1:
                # print(x.shape)
                x = self.blocks[self.lens-self.nums+index](x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.final(x)

        return output

class DiscriminatorF(nn.Module):
    def __init__(self, nums=2, filters=[16, 32, 64, 128, 256],  is_batchnorm=True):
        super(DiscriminatorF, self).__init__()
        self.nums = nums
        self.lens = 5
        self.is_batchnorm = is_batchnorm

        self.input_f = nn.Sequential(nn.Conv3d(filters[self.lens-self.nums]*2, filters[self.lens-self.nums], 3, 1, 1), nn.BatchNorm3d(filters[self.lens-self.nums]), nn.ReLU())

        blocks, blocks_f = [], []
        for index in range(self.lens-1):
            blocks.append(nn.Sequential(nn.Conv3d(filters[index], filters[index+1], 3, 1, 1), nn.BatchNorm3d(filters[index+1]), nn.ReLU(), nn.MaxPool3d(kernel_size=2)))
            blocks_f.append(nn.Sequential(nn.Conv3d(filters[index+1]*3, filters[index+1], 3, 1, 1), nn.BatchNorm3d(filters[index+1]), nn.ReLU()))

        self.blocks = nn.ModuleList(blocks)
        self.blocks_f = nn.ModuleList(blocks_f)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(8192, 1),
            # nn.Sigmoid(),
        )

    def forward(self, inputs):
        downs, ups = inputs
        for index in range(self.nums):
            if index == 0:
                # x = downs[0]
                x = torch.cat([downs[0], ups[0]], 1)
                x = self.input_f(x)
            else:
                # print(x.size(), downs[index].size(), ups[index].size())
                x = torch.cat([x, downs[index], ups[index]], 1)
                x = self.blocks_f[self.lens-self.nums+index-1](x)
            # 如果到达最后一层，就合并了就行
            if index < self.nums - 1:
                x = self.blocks[self.lens-self.nums+index](x)

        x = torch.flatten(x, 1)
        output = self.final(x)

        return output
    

class Adanet1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128, 256], is_deconv=True,  is_batchnorm=True):
        super(Adanet1, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # SPECT feature extract
        self.conv1_spect = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_spect = nn.MaxPool3d(kernel_size=2)
        self.conv2_spect = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_spect = nn.MaxPool3d(kernel_size=2)
        self.conv3_spect = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_spect = nn.MaxPool3d(kernel_size=2)
        self.conv4_spect = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_spect = nn.MaxPool3d(kernel_size=2)
        self.center_spect = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # CT feature extract
        self.conv1_ct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # feature fusion 
        self.fusion1 = unetConv3D(filters[0]*2, filters[0], self.is_batchnorm)
        self.fusion2 = unetConv3D(filters[1]*2, filters[1], self.is_batchnorm)
        self.fusion3 = unetConv3D(filters[2]*2, filters[2], self.is_batchnorm)
        self.fusion4 = unetConv3D(filters[3]*2, filters[3], self.is_batchnorm)
        self.fusionC = unetConv3D(filters[4]*2, filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        ncs, ctas = inputs

        # downsample
        # feature extract
        # SPECT
        conv1_st = self.conv1_spect(ncs)
        maxpool1_st = self.maxpool1_spect(conv1_st)
        conv2_st = self.conv2_spect(maxpool1_st)
        maxpool2_st = self.maxpool2_spect(conv2_st)
        conv3_st = self.conv3_spect(maxpool2_st)
        maxpool3_st = self.maxpool3_spect(conv3_st)
        conv4_st = self.conv4_spect(maxpool3_st)
        maxpool4_st = self.maxpool4_spect(conv4_st)
        center_st = self.center_spect(maxpool4_st)
        # CT
        conv1_ct = self.conv1_ct(ctas)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)

        f1 = self.fusion1(torch.cat([conv1_st, conv1_ct], 1))
        f2 = self.fusion2(torch.cat([conv2_st, conv2_ct], 1))
        f3 = self.fusion3(torch.cat([conv3_st, conv3_ct], 1))
        f4 = self.fusion4(torch.cat([conv4_st, conv4_ct], 1))
        fC = self.fusionC(torch.cat([center_st, center_ct], 1))

        # output_f
        up4_f = self.up_concat4(f4, fC)
        up3_f = self.up_concat3(f3, up4_f)
        up2_f = self.up_concat2(f2, up3_f)
        up1_f = self.up_concat1(f1, up2_f)
        final_f = self.final(up1_f)

        # output_st
        up4_st = self.up_concat4(conv4_st, center_st)
        up3_st = self.up_concat3(conv3_st, up4_st)
        up2_st = self.up_concat2(conv2_st, up3_st)
        up1_st = self.up_concat1(conv1_st, up2_st)
        final_st = self.final(up1_st)

        output = {
            "st": [conv1_st, conv2_st, conv3_st, conv4_st, center_st],
            "up_st": [up1_st, up2_st, up3_st, up4_st, center_st], 
            "final_st": final_st,

            "ct": [f1, f2, f3, f4, fC],
            "up_ct": [up1_f, up2_f, up3_f, up4_f, fC], 
            "final_ct": final_f,
        }

        return output

class Adanet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128, 256], is_deconv=True,  is_batchnorm=True):
        super(Adanet2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        # SPECT feature extract
        self.conv1_spect = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_spect = nn.MaxPool3d(kernel_size=2)
        self.conv2_spect = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_spect = nn.MaxPool3d(kernel_size=2)
        self.conv3_spect = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_spect = nn.MaxPool3d(kernel_size=2)
        self.conv4_spect = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_spect = nn.MaxPool3d(kernel_size=2)
        self.center_spect = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # CT feature extract
        self.conv1_ct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # feature fusion 
        self.fusion1 = unetConv3D(filters[0]*2, filters[0], self.is_batchnorm)
        self.fusion2 = unetConv3D(filters[1]*2, filters[1], self.is_batchnorm)
        self.fusion3 = unetConv3D(filters[2]*2, filters[2], self.is_batchnorm)
        self.fusion4 = unetConv3D(filters[3]*2, filters[3], self.is_batchnorm)
        self.fusionC = unetConv3D(filters[4]*2, filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        ncs, ctas = inputs

        # downsample
        # feature extract
        # SPECT
        conv1_st = self.conv1_spect(ncs)
        maxpool1_st = self.maxpool1_spect(conv1_st)
        conv2_st = self.conv2_spect(maxpool1_st)
        maxpool2_st = self.maxpool2_spect(conv2_st)
        conv3_st = self.conv3_spect(maxpool2_st)
        maxpool3_st = self.maxpool3_spect(conv3_st)
        conv4_st = self.conv4_spect(maxpool3_st)
        maxpool4_st = self.maxpool4_spect(conv4_st)
        center_st = self.center_spect(maxpool4_st)
        # CT
        conv1_ct = self.conv1_ct(ctas)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)

        f1 = self.fusion1(torch.cat([conv1_st, conv1_ct], 1))
        f2 = self.fusion2(torch.cat([conv2_st, conv2_ct], 1))
        f3 = self.fusion3(torch.cat([conv3_st, conv3_ct], 1))
        f4 = self.fusion4(torch.cat([conv4_st, conv4_ct], 1))
        fC = self.fusionC(torch.cat([center_st, center_ct], 1))

        conv1_st = self.fusion1(torch.cat([conv1_st, conv1_st], 1))
        conv2_st = self.fusion2(torch.cat([conv2_st, conv2_st], 1))
        conv3_st = self.fusion3(torch.cat([conv3_st, conv3_st], 1))
        conv4_st = self.fusion4(torch.cat([conv4_st, conv4_st], 1))
        center_st = self.fusionC(torch.cat([center_st, center_st], 1))

        # output_f
        up4_f = self.up_concat4(f4, fC)
        up3_f = self.up_concat3(f3, up4_f)
        up2_f = self.up_concat2(f2, up3_f)
        up1_f = self.up_concat1(f1, up2_f)
        final_f = self.final(up1_f)

        # output_st
        up4_st = self.up_concat4(conv4_st, center_st)
        up3_st = self.up_concat3(conv3_st, up4_st)
        up2_st = self.up_concat2(conv2_st, up3_st)
        up1_st = self.up_concat1(conv1_st, up2_st)
        final_st = self.final(up1_st)

        output = {
            "ct": [f1, f2, f3, f4, fC],
            "up_ct": [up1_f, up2_f, up3_f, up4_f],
            "final_ct": final_f,

            "st": [conv1_st, conv2_st, conv3_st, conv4_st, center_st],
            "up_st": [up1_st, up2_st, up3_st, up4_st], 
            "final_st": final_st,
        }

        return output
    
class Adanet3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128, 256], is_deconv=True,  is_batchnorm=True):
        super(Adanet3, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        
        # SPECT feature extract
        self.conv1_st = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_st = nn.MaxPool3d(kernel_size=2)
        self.conv2_st = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_st = nn.MaxPool3d(kernel_size=2)
        self.conv3_st = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_st = nn.MaxPool3d(kernel_size=2)
        self.conv4_st = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_st = nn.MaxPool3d(kernel_size=2)
        self.center_st = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # CT feature extract
        self.conv1_ct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # ST -> CT 
        self.conv1_stct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_stct = nn.MaxPool3d(kernel_size=2)
        self.conv2_stct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_stct = nn.MaxPool3d(kernel_size=2)
        self.conv3_stct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_stct = nn.MaxPool3d(kernel_size=2)
        self.conv4_stct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_stct = nn.MaxPool3d(kernel_size=2)
        self.center_stct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # feature fusion 
        self.fusion1 = unetConv3D(filters[0]*2, filters[0], self.is_batchnorm)
        self.fusion2 = unetConv3D(filters[1]*2, filters[1], self.is_batchnorm)
        self.fusion3 = unetConv3D(filters[2]*2, filters[2], self.is_batchnorm)
        self.fusion4 = unetConv3D(filters[3]*2, filters[3], self.is_batchnorm)
        self.fusionC = unetConv3D(filters[4]*2, filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        ncs, ctas = inputs

        # feature extract
        # SPECT
        conv1_st = self.conv1_st(ncs)
        maxpool1_st = self.maxpool1_st(conv1_st)
        conv2_st = self.conv2_st(maxpool1_st)
        maxpool2_st = self.maxpool2_st(conv2_st)
        conv3_st = self.conv3_st(maxpool2_st)
        maxpool3_st = self.maxpool3_st(conv3_st)
        conv4_st = self.conv4_st(maxpool3_st)
        maxpool4_st = self.maxpool4_st(conv4_st)
        center_st = self.center_st(maxpool4_st)
        # if tag == 0:
        # CT
        conv1_ct = self.conv1_ct(ctas)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)

        f1 = self.fusion1(torch.cat([conv1_st, conv1_ct], 1)) + conv1_st
        f2 = self.fusion2(torch.cat([conv2_st, conv2_ct], 1)) + conv2_st
        f3 = self.fusion3(torch.cat([conv3_st, conv3_ct], 1)) + conv3_st
        f4 = self.fusion4(torch.cat([conv4_st, conv4_ct], 1)) + conv4_st
        fC = self.fusionC(torch.cat([center_st, center_ct], 1)) + center_st

        # upsample
        up4_f = self.up_concat4(f4, fC)
        up3_f = self.up_concat3(f3, up4_f)
        up2_f = self.up_concat2(f2, up3_f)
        up1_f = self.up_concat1(f1, up2_f)
        final_f = self.final(up1_f)

        # output = final_f
        # else:
        # # ST -> CT
        conv1_stct = self.conv1_stct(ncs)
        maxpool1_stct = self.maxpool1_stct(conv1_stct)
        conv2_stct = self.conv2_stct(maxpool1_stct)
        maxpool2_stct = self.maxpool2_stct(conv2_stct)
        conv3_stct = self.conv3_stct(maxpool2_stct)
        maxpool3_stct = self.maxpool3_stct(conv3_stct)
        conv4_stct = self.conv4_stct(maxpool3_stct)
        maxpool4_stct = self.maxpool4_stct(conv4_stct)
        center_stct = self.center_stct(maxpool4_stct)

        conv1_st = self.fusion1(torch.cat([conv1_st, conv1_stct], 1)) + conv1_st
        conv2_st = self.fusion2(torch.cat([conv2_st, conv2_stct], 1)) + conv2_st
        conv3_st = self.fusion3(torch.cat([conv3_st, conv3_stct], 1)) + conv3_st
        conv4_st = self.fusion4(torch.cat([conv4_st, conv4_stct], 1)) + conv4_st
        center_st = self.fusionC(torch.cat([center_st, center_stct], 1)) + center_st

        # output_1
        up4_st = self.up_concat4(conv4_st, center_st)
        up3_st = self.up_concat3(conv3_st, up4_st)
        up2_st = self.up_concat2(conv2_st, up3_st)
        up1_st = self.up_concat1(conv1_st, up2_st)
        final_st = self.final(up1_st)

        output = final_st

        output = {
            "ct": [f1, f2, f3, f4, fC],
            "up_ct": [up1_f, up2_f, up3_f, up4_f],
            "final_ct": final_f,

            "st": [conv1_st, conv2_st, conv3_st, conv4_st, center_st],
            "up_st": [up1_st, up2_st, up3_st, up4_st], 
            "final_st": final_st,

            # "ct": [conv1_ct, conv2_ct, conv3_ct, conv4_ct, center_ct],
            # "stct": [conv1_stct, conv2_stct, conv3_stct, conv4_stct, center_stct],
        }

        return output
    
class featureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super(featureFusion, self).__init__()
        
        self.cbam1 = CBAM(in_channels)
        self.cbam2 = CBAM(in_channels)
        self.conv = unetConv3D(in_channels*2, out_channels, is_batchnorm)

    def forward(self, inputs):
        feature1, feature2 = inputs
#         # 1. 
#         feature = self.cbam(torch.cat([feature1, feature2], 1))
#         outputs = self.conv(feature)
        # 2. 
#         print(feature1.size(), feature2.size())
#         print(self.cbam1)
        feature1 = self.cbam1(feature1)
        feature2 = self.cbam2(feature2)
#         outputs = torch.stack([feature1, feature2], dim=0).sum(dim=0)
        outputs = self.conv(torch.cat([feature1, feature2], 1))
        
        return outputs


class Adanet4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128, 256], is_deconv=True,  is_batchnorm=True):
        super(Adanet4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        
        # SPECT feature extract
        self.conv1_st = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_st = nn.MaxPool3d(kernel_size=2)
        self.conv2_st = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_st = nn.MaxPool3d(kernel_size=2)
        self.conv3_st = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_st = nn.MaxPool3d(kernel_size=2)
        self.conv4_st = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_st = nn.MaxPool3d(kernel_size=2)
        self.center_st = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # CT feature extract
        self.conv1_ct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # ST -> CT 
        self.conv1_stct = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1_stct = nn.MaxPool3d(kernel_size=2)
        self.conv2_stct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_stct = nn.MaxPool3d(kernel_size=2)
        self.conv3_stct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_stct = nn.MaxPool3d(kernel_size=2)
        self.conv4_stct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_stct = nn.MaxPool3d(kernel_size=2)
        self.center_stct = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # feature fusion 
        self.fusion1 = featureFusion(filters[0], filters[0], self.is_batchnorm)
        self.fusion2 = featureFusion(filters[1], filters[1], self.is_batchnorm)
        self.fusion3 = featureFusion(filters[2], filters[2], self.is_batchnorm)
        self.fusion4 = featureFusion(filters[3], filters[3], self.is_batchnorm)
        self.fusionC = featureFusion(filters[4], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)


    def forward(self, inputs):
        ncs, ctas = inputs

        # feature extract
        # SPECT
        conv1_st = self.conv1_st(ncs)
        maxpool1_st = self.maxpool1_st(conv1_st)
        conv2_st = self.conv2_st(maxpool1_st)
        maxpool2_st = self.maxpool2_st(conv2_st)
        conv3_st = self.conv3_st(maxpool2_st)
        maxpool3_st = self.maxpool3_st(conv3_st)
        conv4_st = self.conv4_st(maxpool3_st)
        maxpool4_st = self.maxpool4_st(conv4_st)
        center_st = self.center_st(maxpool4_st)
        # if tag == 0:
        # CT
        conv1_ct = self.conv1_ct(ctas)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)

        f1 = self.fusion1((conv1_st, conv1_ct)) + conv1_st
        f2 = self.fusion2((conv2_st, conv2_ct)) + conv2_st
        f3 = self.fusion3((conv3_st, conv3_ct)) + conv3_st
        f4 = self.fusion4((conv4_st, conv4_ct)) + conv4_st
        fC = self.fusionC((center_st, center_ct)) + center_st

        # upsample
        up4_f = self.up_concat4(f4, fC)
        up3_f = self.up_concat3(f3, up4_f)
        up2_f = self.up_concat2(f2, up3_f)
        up1_f = self.up_concat1(f1, up2_f)
        final_f = self.final(up1_f)

        # output = final_f
        # else:
        # # ST -> CT
        conv1_stct = self.conv1_stct(ncs)
        maxpool1_stct = self.maxpool1_stct(conv1_stct)
        conv2_stct = self.conv2_stct(maxpool1_stct)
        maxpool2_stct = self.maxpool2_stct(conv2_stct)
        conv3_stct = self.conv3_stct(maxpool2_stct)
        maxpool3_stct = self.maxpool3_stct(conv3_stct)
        conv4_stct = self.conv4_stct(maxpool3_stct)
        maxpool4_stct = self.maxpool4_stct(conv4_stct)
        center_stct = self.center_stct(maxpool4_stct)

        conv1_st = self.fusion1((conv1_st, conv1_stct)) + conv1_st
        conv2_st = self.fusion2((conv2_st, conv2_stct)) + conv2_st
        conv3_st = self.fusion3((conv3_st, conv3_stct)) + conv3_st
        conv4_st = self.fusion4((conv4_st, conv4_stct)) + conv4_st
        center_st = self.fusionC((center_st, center_stct)) + center_st

        # output_1
        up4_st = self.up_concat4(conv4_st, center_st)
        up3_st = self.up_concat3(conv3_st, up4_st)
        up2_st = self.up_concat2(conv2_st, up3_st)
        up1_st = self.up_concat1(conv1_st, up2_st)
        final_st = self.final(up1_st)

        output = final_st

        output = {
            "ct": [f1, f2, f3, f4, fC],
            "up_ct": [up1_f, up2_f, up3_f, up4_f, fC],
            "final_ct": final_f,

            "st": [conv1_st, conv2_st, conv3_st, conv4_st, center_st],
            "up_st": [up1_st, up2_st, up3_st, up4_st, center_st], 
            "final_st": final_st,

            # "ct": [conv1_ct, conv2_ct, conv3_ct, conv4_ct, center_ct],
            # "stct": [conv1_stct, conv2_stct, conv3_stct, conv4_stct, center_stct],
        }
        # print(f"ct.shape: {f1.shape, f2.shape, f3.shape, f4.shape, fC.shape}")
        # print(f"up_ct.shape: {up1_f.shape, up2_f.shape, up3_f.shape, up4_f.shape}")
        # print(f"st.shape: {conv1_st.shape, conv2_st.shape, conv3_st.shape, conv4_st.shape, center_st.shape}")
        # print(f"ct.shape: {up1_st.shape, up2_st.shape, up3_st.shape, up4_st.shape}")
#         ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]), torch.Size([
# 16, 256, 2, 4, 4]))
# up_ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]))
# st.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]), torch.Size([
# 16, 256, 2, 4, 4]))
# ct.shape: (torch.Size([16, 16, 32, 64, 64]), torch.Size([16, 32, 16, 32, 32]), torch.Size([16, 64, 8, 16, 16]), torch.Size([16, 128, 4, 8, 8]))

        return output

if __name__ == '__main__':
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    model = Adanet1(in_channels=1, out_channels=1).cuda()
    disc = DiscriminatorF2(nums=2).cuda()
    # for i in range(100):
    ncs = torch.rand(4, 1, 32, 64, 64).cuda()
    acs = torch.rand(4, 1, 32, 64, 64).cuda()
    outputs = model((ncs, acs))
    d = disc((outputs["st"][-2:], outputs["up_st"][-2:]))
    print(d.size())



