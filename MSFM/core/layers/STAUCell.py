import typing_extensions
import torch
import torch.nn as nn
import math
from core.layers.MulWFT import MulWFT


class STAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(STAUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)
        
        # Frequency domain module and adaptive fusion weight
        self.mul_wft = MulWFT(in_channels=num_hidden)
        self.alpha_t = nn.Parameter(torch.zeros(1)) 
        self.alpha_s = nn.Parameter(torch.zeros(1)) 

    def forward(self, T_t, S_t, t_att, s_att, t_spatial, s_spatial):
        s_next = self.conv_s_next(S_t)
        t_next = self.conv_t_next(T_t)
        weights_list = []
        weights_list_t = []
        
        for i in range(self.tau):
            weights_list.append(
                (s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        for j in range(t_spatial.shape[0]):
            weights_list_t.append(
                (t_spatial[j]*t_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
                
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(
            weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        weights_list_t = torch.stack(weights_list_t, dim=0)
        weights_list_t = torch.reshape(
            weights_list_t, (*weights_list_t.shape, 1, 1, 1))
        weights_list_t = self.softmax(weights_list_t)

        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0)

        S_trend = s_spatial * weights_list_t
        S_trend = S_trend.sum(dim=0)

        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend

        s_att_gate = torch.sigmoid(s_next)
        S_fusion = S_t * s_att_gate + (1 - s_att_gate) * S_trend

        # Dual stream frequency domain enhancement
        # Extract frequency domain features of time and spatial streams
        T_freq = self.mul_wft(T_fusion)
        S_freq = self.mul_wft(S_fusion)

        T_fusion = T_fusion + self.alpha_t * T_freq
        S_fusion = S_fusion + self.alpha_s * S_freq
        # ==========================

        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_fusion)

        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        
        if self.cell_mode == 'residual':
            S_new = S_new + S_t

        return T_new, S_new