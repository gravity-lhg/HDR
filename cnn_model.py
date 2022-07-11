# Author: Haoguang Liu
# Date: 2022.07.11 19:47
# E-mail: Liu.gravity@gmail.com

import torch.nn as nn 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        # create network's backbone
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # create network's classification head
        self.fc = nn.Sequential(
            nn.Linear(64 * 49, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, input):
        x = input
        
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x