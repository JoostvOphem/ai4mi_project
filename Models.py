# This file consists of models that could be applied to the SEGTHOR challenge.
import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import UNETR
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


# Inspired by https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        dec3 = self.dec3(torch.cat([self.upsample(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        return self.final_conv(dec1)
    
    def init_weights(self):
        pass


class UNETR_monai(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Parameters optimized for medical imaging:
        # - feature_size: smaller than default since we have clear anatomical structures
        # - hidden_size: reduced to prevent overfitting on medical data
        # - num_heads: adjusted for the hidden_size
        # - dropout_rate: increased slightly to prevent overfitting
        
        self.unetr = UNETR(
            in_channels=input_dim,      # 1 for CT scans
            out_channels=output_dim,    # 5 for your classes
            img_size=(128, 128, 64),    # Your target shape
            feature_size=32,            # Increased from 16 for better feature extraction
            hidden_size=384,            # Reduced from 768 to prevent overfitting
            mlp_dim=1536,              # 4x hidden_size
            num_heads=6,               # Reduced from 12 to match hidden_size
            pos_embed="conv",          # Better for medical images than learned
            norm_name="instance",      # Good for varying intensity in medical images
            conv_block=True,           
            res_block=True,            # Helps with gradient flow
            dropout_rate=0.1           # Slight regularization
        )

    def forward(self, x):
        return self.unetr(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


import torch.nn as nn


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


class shallowCNN(nn.Module):
    def __init__(self, nin, nout, nG=4):
        super(shallowCNN, self).__init__()
        self.conv0 = convBatch(nin, nG * 4)
        self.conv1 = convBatch(nG * 4, nG * 4)
        self.conv2 = convBatch(nG * 4, nout)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        return x2

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


# Example usage:
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1, 1, 128, 128, 64)
    
    # Initialize the model
    model = UNETR_monai(in_channels=1, out_channels=5, img_size=(128, 128, 64))
    
    # Initialize weights
    model.init_weights()
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")