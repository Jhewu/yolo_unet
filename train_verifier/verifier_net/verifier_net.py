import torch 
import torch.nn as nn 
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.resnet import resnet18, ResNet18_Weights

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class VerificationNet(nn.Module):
    def __init__(self, num_extra_features=0, input_ch=4):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Replace the first conv layer to handle 4 channels
        if input_ch != 3:
            old_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                input_ch, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            # Initialize with average of original weights across channels
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True).expand(-1, input_ch, -1, -1))
            
            self.backbone.features[0][0] = new_conv

        self.backbone.classifier = nn.Identity()  # Remove classifier head
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.3)

        # Use a slightly deeper classifier for better fitting
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.features[-1].out_channels + num_extra_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(self, x, extra_features=None):
        feats = self.backbone(x)
        if len(feats.shape) > 2:
            feats = feats.mean(dim=[2, 3])  # Global average pooling
        feats = self.dropout(feats)
        if extra_features is not None:
            feats = torch.cat([feats, extra_features], dim=1)
        return self.classifier(feats)