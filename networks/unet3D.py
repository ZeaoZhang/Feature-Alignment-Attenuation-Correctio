import torch.nn as nn
import torch
import torch.nn.functional as F
from .cbam import CBAM

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

class unet3D(nn.Module):
    def __init__(self, feature_scale=4, is_deconv=True, in_channels=2, out_channels=1, is_batchnorm=True):
        super(unet3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # downsampling
        self.conv1 = unetConv3D(in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)
        self.center = unetConv3D(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], out_channels, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
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
    

class unet3Dcta(nn.Module):
    def __init__(self, feature_scale=4, is_deconv=True, in_channels=[32, 32], out_channels=19, is_batchnorm=True):
        super(unet3Dcta, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # channel scale
        # self.input_scale_spect = nn.Conv2d(in_channels[0], 32, 1)
        # self.input_scale_ct = nn.Conv2d(in_channels[1], 32, 1)
        
        # SPECT feature extract
        self.conv1_spect = unetConv3D(1, filters[0], self.is_batchnorm)
        self.maxpool1_spect = nn.MaxPool3d(kernel_size=2)
        self.conv2_spect = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_spect = nn.MaxPool3d(kernel_size=2)
        self.conv3_spect = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_spect = nn.MaxPool3d(kernel_size=2)
        self.conv4_spect = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_spect = nn.MaxPool3d(kernel_size=2)
        self.center_spect = unetConv3D(filters[3], filters[4], self.is_batchnorm)
        # CT feature extract
        self.conv1_ct = unetConv3D(1, filters[0], self.is_batchnorm)
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=2)
        self.conv2_ct = unetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=2)
        self.conv3_ct = unetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=2)
        self.conv4_ct = unetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=2)
        self.center_ct = unetConv3D(filters[3], filters[4], self.is_batchnorm)
        # feature fusion 
        self.fusion1 = featureFusion(filters[0], filters[0], self.is_batchnorm)
        self.fusion2 = featureFusion(filters[1], filters[1], self.is_batchnorm)
        self.fusion3 = featureFusion(filters[2], filters[2], self.is_batchnorm)
        self.fusion4 = featureFusion(filters[3], filters[3], self.is_batchnorm)
        self.fusion_center= featureFusion(filters[4], filters[4], self.is_batchnorm)       

        # upsampling
        self.up_concat4 = unetUp3D(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3D(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3D(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3D(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], 1, 1)
        # self.ouput_scale = nn.Conv2d(32, out_channels, 1)

    def forward(self, inputs):
        nc, cta = inputs
        # downsample
        # feature extract
        # SPECT
        conv1_spect = self.conv1_spect(nc)
        maxpool1_spect = self.maxpool1_spect(conv1_spect)
        conv2_spect = self.conv2_spect(maxpool1_spect)
        maxpool2_spect = self.maxpool2_spect(conv2_spect)
        conv3_spect = self.conv3_spect(maxpool2_spect)
        maxpool3_spect = self.maxpool3_spect(conv3_spect)
        conv4_spect = self.conv4_spect(maxpool3_spect)
        maxpool4_spect = self.maxpool4_spect(conv4_spect)
        center_spect = self.center_spect(maxpool4_spect)
        # CT
        conv1_ct = self.conv1_ct(cta)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)
        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)
        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)
        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)
        center_ct = self.center_ct(maxpool4_ct)
        # feature fusion 
        fusion1 = self.fusion1((conv1_spect, conv1_ct))
        fusion2 = self.fusion2((conv2_spect, conv2_ct))
        fusion3 = self.fusion3((conv3_spect, conv3_ct))
        fusion4 = self.fusion4((conv4_spect, conv4_ct))
        fusion_center = self.fusion_center((center_spect, center_ct))
        
        up4 = self.up_concat4(fusion4, fusion_center)
        up3 = self.up_concat3(fusion3, up4)
        up2 = self.up_concat2(fusion2, up3)
        up1 = self.up_concat1(fusion1, up2)

        final = self.final(up1)

        return final
    
    
if __name__ == '__main__':
    import os
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model = unet3D(in_channels=1, out_channels=1).cuda()
    # for i in range(100):
    inputs = torch.rand(8, 1, 64, 64, 32).cuda()
    output = model(inputs)
    print(output.size())


