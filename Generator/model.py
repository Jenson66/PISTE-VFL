# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class a_1(nn.Module):
    def __init__(self):
        super(a_1, self).__init__()

    def forward(self, output):
        a = torch.ones(size=output.shape).to(device)
        a[:,:,:,:16]=0 

        return output*a

class a_2(nn.Module):
    def __init__(self):
        super(a_2, self).__init__()

    def forward(self, output):
        a = torch.ones(size=output.shape).to(device)
        a[:,:,:,16:]=0 

        return output*a

class Generator(nn.Module):
    def __init__(self, channel=3, shape_img=32, n_hidden_1=2048, batchsize=6, g_in=128, d = 32, iters=0):
        super(Generator, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.iters = iters

        self.layer1 = nn.Sequential(nn.Linear(g_in, n_hidden_1),
                                   )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, g_in),
                                  )

        block_num = int(np.log2(shape_img) - 3)
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d*pow(2,block_num) * 2, 4, 1, 0),
            GLU()
        )
        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        self.a_output_1 = nn.Sequential(a_1())
        self.a_output_2 = nn.Sequential(a_2())


    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                             padding=1, bias=False)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes*2),
            nn.BatchNorm2d(out_planes*2),
            GLU()
        )
        return block


    # forward method
    def forward(self, x, batch_size, iters):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(batch_size, self.g_in, 1, 1)
        output = self.block0(x)
        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        if iters%2 ==0:
            output = self.a_output_1(output)
        else:
            output = self.a_output_2(output)
        
        return output