# This file consists of models that could be applied to the SEGTHOR challenge.

import torch
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

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



# !!! AI Code below !!!

class SAM(nn.Module):
    def __init__(self, num_classes, checkpoint_path, model_type="vit_h"):
        super(SAM, self).__init__()
        self.num_classes = num_classes
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.sam)
        self.conv = nn.Conv2d(1, num_classes, kernel_size=1)  # Add a learnable layer
        
    def forward(self, x):
        # x should be a batch of images (B, C, H, W) or (B, H, W)
        if x.dim() == 3:
            B, H, W = x.shape
            x = x.unsqueeze(1)  # Add channel dimension
        else:
            B, C, H, W = x.shape
        
        device = x.device
        
        # Process each image in the batch
        masks = []
        for img in x:
            # Ensure the image is in the correct format for SAM (H, W, C)
            img_np = img.cpu().numpy()
            if img_np.shape[0] == 1:  # If it's a single-channel image
                img_np = np.repeat(img_np, 3, axis=0)  # Repeat the channel 3 times
            img_np = img_np.transpose(1, 2, 0)
            
            # Set image for SAM
            self.predictor.set_image(img_np)
            
            # Use center of image as prompt
            input_point = np.array([[H//2, W//2]])
            input_label = np.array([1])
            
            # Generate mask
            mask, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False
            )
            masks.append(torch.from_numpy(mask).float())
        
        # Stack masks and move to correct device
        masks = torch.stack(masks).to(device)  # Shape: (B, 1, H, W)
        
        # Ensure masks have the correct shape (B, 1, H, W)
        if masks.dim() == 5:
            masks = masks.squeeze(1)
        
        # Pass through a learnable layer to get class probabilities
        out = self.conv(masks)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(out, dim=1)
        
        return probs

    def init_weights(self):
        # Initialize the conv layer
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)