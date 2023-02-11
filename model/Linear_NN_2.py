from torch import nn
import numpy as np
import torch

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


class Client_LeakyreluNet_4_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_4_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                     nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(47, n_hidden_1),
                                     nn.LeakyReLU())
              
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                     nn.LeakyReLU() )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,  n_hidden_2),
                                    )
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_2,  n_hidden_2),
                                    )


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3, 11, 3, 2, 10, 5]
            for i in range(10):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer4_out


class Server_LeakyreluNet_4(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2):
        super(Server_LeakyreluNet_4, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.LeakyReLU())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
                                   )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer6_out




class Client_LinearNet_2_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LinearNet_2_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    )

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(47, n_hidden_1),
                                    )
              
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3, 11, 3, 2, 10, 5]
            for i in range(10):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_LeakyreluNet_2_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_2_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(47, n_hidden_1),
                                    nn.LeakyReLU())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3, 11, 3, 2, 10, 5]
            for i in range(10):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_LeakyreluNet_3_em_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_3_em_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(10, n_hidden_1),
                                    nn.LeakyReLU())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(47, n_hidden_1),
                                    nn.LeakyReLU())


        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_2),
                                    nn.LeakyReLU())


    def forward(self, x, client):
        if client ==2:
            z = (x-1).long()
            c = [2, 7, 2, 2, 3, 11, 3, 2, 10, 5]
            for i in range(10):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a,1)
                if i ==0:
                    x = b
                else:
                    x = torch.cat([x, b],1)
    
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out



