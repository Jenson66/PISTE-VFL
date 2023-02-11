# -*-coding:utf-8-*-
from torch import nn
import numpy as np
import torch


class Client_LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2= 128, hideen3=128, num_classes=2):
        super(Client_LeNet, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen2)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Server_LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_LeNet, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3, hideen4),
            act(),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(hideen4, hideen5),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hideen5, num_classes),
        )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        out1 = self.fc2(x)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)

        return  out4



class general_LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2= 128, hideen3=128, num_classes=2):
        super(general_LeNet, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen2),
            nn.Linear(hideen2, hideen2),
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out



class Client_LeNet_linear(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2= 128, hideen3=128, num_classes=2):
        super(Client_LeNet_linear, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),    
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),   
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen2)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Client_LeNet_linear4(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2= 128, hideen3=128, num_classes=2):
        super(Client_LeNet_linear4, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),    
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),   
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hideen1, hideen2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen2, hideen2)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


