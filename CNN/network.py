import torch
import torch.nn.functional as F
from torch import nn

# class Network(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.full = nn.Linear(16 * 72 * 72, num_classes)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.full = nn.Linear( 64 * 72 * 72, 128)
#         self.full1 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.pool1(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.pool2(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.full(x)
#         x = self.relu(x)
#         x = self.full1(x)
#         return x
    
# class Network(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
#         self.full = nn.Linear(16 * 72 * 72, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.full(x)
#         return x

class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()
        
        # Conv Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv Block 1: 128x128 -> 64x64
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Conv Block 2: 64x64 -> 32x32
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Conv Block 3: 32x32 -> 16x16
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Conv Block 4: 16x16 -> 8x8
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # Global Average Pooling: 8x8 -> 1x1
        x = self.global_avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    