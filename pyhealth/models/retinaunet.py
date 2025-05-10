import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel

class RetinaUNet(BaseModel):
    """
    Retina UNet model recreated from the paper
    Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection
    https://arxiv.org/abs/1811.08661

    This model uses an initial resnet model to encode input data as a base model. Resnet acts as an encoder that is followed by serveral convultion transsposers
    and their own convolution layers. The result is an approximation of identifiable areas as a mask of the region.
    The model aims to create more efficient
    labeling from a small set of label data.

    """  
    def __init__(self):
        super(RetinaUNet, self).__init__()
        resnet = models.resnet18(pretrained=False)

        # Encoder layers (for skip connections)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder with skip connections
        x = self.upconv1(x5)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.conv1(x))

        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.conv2(x))

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv3(x))

        x = self.upconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv4(x))

        x = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=False)
        return self.final_conv(x)
