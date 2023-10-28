import torch
import torch.nn as nn

from mmcv.cnn import Linear
from mmcv.runner import BaseModule

from ..builder import BACKBONES


@BACKBONES.register_module()
class MLP(BaseModule):
    def __init__(self, in_channel, out_channel ,init_cfg = None):
        super().__init__(init_cfg)

        self.in_channel = in_channel
        self.out_channel = out_channel

        amp_Hdim_head = []
        amp_Hdim_head.append(Linear(30, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        amp_Hdim_head.append(nn.ReLU())
        amp_Hdim_head.append(Linear(512, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        amp_Hdim_head.append(nn.ReLU())
        self.amp_Hdim_head = nn.Sequential(*amp_Hdim_head)

        amp_Ldim_head = []
        amp_Ldim_head.append(Linear(30, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        amp_Ldim_head.append(nn.ReLU())
        amp_Ldim_head.append(Linear(512, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        amp_Ldim_head.append(nn.ReLU())
        self.amp_Ldim_head = nn.Sequential(*amp_Ldim_head)

        phd_head = []
        phd_head.append(Linear(30, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        phd_head.append(nn.ReLU())
        phd_head.append(Linear(512, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        phd_head.append(nn.ReLU())
        self.phd_head = nn.Sequential(*phd_head)

        head = []
        head.append(Linear(512*3, 512))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        head.append(nn.ReLU())
        head.append(Linear(512, 256))
        amp_Hdim_head.append(nn.BatchNorm1d(512))
        head.append(nn.ReLU())
        self.head = nn.Sequential(*head)

    def forward(self, x):
        bs, _, _, _, channel = x.shape
        x = x.reshape(-1, channel)
        
        amp_Hdim = x[...,:30]
        amp_Ldim = x[...,30:60]
        phd = x[...,60:]

        amp_Hdim = self.amp_Hdim_head(amp_Hdim)
        amp_Ldim = self.amp_Ldim_head(amp_Ldim)
        phd = self.phd_head(phd)
        x = torch.cat((amp_Hdim, amp_Ldim, phd), -1)
        x = self.head(x)
        x = x.reshape(bs, -1, 256)
        
        return x
