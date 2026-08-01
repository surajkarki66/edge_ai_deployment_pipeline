from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

# Model architecture
class Net(nn.Module):
    # Structure of the NN
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 16x28x28 | 3
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)  # 16x28x28 | 5
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x14x14 | 6
        self.drop1 = nn.Dropout2d(p=0.1)
        # Block 2
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 32x14x14 | 10
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)  # 32x14x14 | 14
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x7x7 | 16
        self.drop2 = nn.Dropout2d(p=0.1)
        # Block 3
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0)  # 16x7x7 | 16
        self.bn5 = nn.BatchNorm2d(num_features=16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)  # 16x7x7 | 24
        self.bn6 = nn.BatchNorm2d(num_features=16)
        # Head
        self.gap1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 16x1x1 | 48
        self.fc1 = nn.Linear(in_features=16, out_features=10)  # 10x1x1 | 48

    # Feed forward of the NN
    def forward(self, x):
        x = self.drop1(self.pool1(self.bn2(F.relu(self.conv2(self.bn1(F.relu(self.conv1(x))))))))
        x = self.drop2(self.pool2(self.bn4(F.relu(self.conv4(self.bn3(F.relu(self.conv3(x))))))))
        x = self.bn6(F.relu(self.conv6(self.bn5(F.relu(self.conv5(x))))))
        x = self.gap1(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)