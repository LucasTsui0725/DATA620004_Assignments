import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, down_sample):
        super(ResBlock, self).__init__()
        self.down_sample = down_sample
        if down_sample:
            self.conv_layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
            self.sample_layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(output_channel)
            )
        else:
            self.conv_layer = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
            self.sample_layer = nn.Sequential()
        
        self.block_layer = nn.Sequential(
            nn.BatchNorm2d(output_channel), 
            nn.ReLU(), 
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(output_channel)
        )
    
    def forward(self, x):
        out = self.sample_layer(x)
        x = self.conv_layer(x)
        x = self.block_layer(x)
        
        return nn.ReLU()(x + out)

class ResNet18(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            # nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=3),
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            ResBlock(64, 64, down_sample=False),
            ResBlock(64, 64, down_sample=False),
        )
        self.block3 = nn.Sequential(
            ResBlock(64, 128, down_sample=True), 
            ResBlock(128, 128, down_sample=False)
        )
        self.block4 = nn.Sequential(
            ResBlock(128, 256, down_sample=True), 
            ResBlock(256, 256, down_sample=False)
        )
        self.block5 = nn.Sequential(
            ResBlock(256, 512, down_sample=True), 
            ResBlock(512, 512, down_sample=False)
        )
        self.classifier = nn.Linear(512, output_channel)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x