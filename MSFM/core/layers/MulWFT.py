import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowWFTBlock(nn.Module):
    def __init__(self, in_channels, win_size):
        super(WindowWFTBlock, self).__init__()
        self.win_size = win_size
        self.padding = win_size // 2
        
        # Register 2D Hanning Window
        hann_1d = torch.hann_window(win_size)
        hann_2d = hann_1d.outer(hann_1d)
        self.register_buffer('window', hann_2d.view(1, 1, win_size, win_size, 1)) 
        
        # Frequency domain size after rfft2
        self.freq_size = win_size * (win_size // 2 + 1)

        self.conv = nn.Conv2d(in_channels * 3 * self.freq_size, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8 if in_channels >= 8 else 1, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reflection filling
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        
        #  (B, C, win, win, H*W)
        x_unfold = F.unfold(x_pad, kernel_size=self.win_size, stride=1)
        x_unfold = x_unfold.view(B, C, self.win_size, self.win_size, H * W)
        
        # Hann
        x_unfold = x_unfold * self.window
        
        # 2D Fourier Transform
        fft_result = torch.fft.rfft2(x_unfold, dim=(-3, -2)) 
        
        # Extract amplitude, real and imaginary parts
        real_part = fft_result.real
        imag_part = fft_result.imag
        amplitude = torch.sqrt(torch.clamp(real_part**2 + imag_part**2,min=1e-8))

        freq_feat = torch.cat([amplitude,real_part,imag_part],dim=1)
        
        # (B, C*3*freq_size, H, W)
        freq_feat = freq_feat.permute(0, 1, 2, 3, 4).reshape(B, -1, H * W)
        freq_feat = freq_feat.view(B, -1, H, W)
        
        # LayerNorm
        freq_feat = F.layer_norm(freq_feat, freq_feat.shape[1:])
        
        # GroupNorm
        out = self.conv(freq_feat)
        out = self.norm(out)
        return out


class DualPathFrequencyAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(DualPathFrequencyAttentionGate, self).__init__()
        
        # Frequency domain attention
        self.freq_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        
        # channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        )
        
        # Conv2D 1x1
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        freq_feat = self.freq_attn(x)
        channel_feat = self.channel_attn(x)
        
        attn_weight = self.sigmoid(freq_feat + channel_feat)
        bottom_feat = self.bottom_conv(x)
        
        return attn_weight * bottom_feat


class MulWFT(nn.Module):
    def __init__(self, in_channels):
        super(MulWFT, self).__init__()
        self.branch_3x3 = WindowWFTBlock(in_channels, win_size=3)
        self.branch_5x5 = WindowWFTBlock(in_channels, win_size=5)
        self.branch_7x7 = WindowWFTBlock(in_channels, win_size=7)
        
        self.concat_channels = in_channels * 3
        self.fusion_gate = DualPathFrequencyAttentionGate(self.concat_channels)
        self.out_conv = nn.Conv2d(self.concat_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out_3 = self.branch_3x3(x)
        out_5 = self.branch_5x5(x)
        out_7 = self.branch_7x7(x)
        
        x_concat = torch.cat([out_3, out_5, out_7], dim=1)
        gated_feat = self.fusion_gate(x_concat)
        out = self.out_conv(gated_feat)
        
        return out + x