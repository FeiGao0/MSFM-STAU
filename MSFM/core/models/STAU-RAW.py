import torch
import torch.nn as nn
from core.layers.STAUCell import STAUCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError

        # ================= 修改点 1：动态计算各层尺寸 =================
        # 基础尺寸（不除 sr_size）
        base_h = configs.img_height // configs.patch_size
        base_w = configs.img_width // configs.patch_size

        n = int(math.log2(configs.sr_size))

        # 计算编码器每一层的输出尺寸
        encoder_sizes = [(base_h, base_w)]
        h, w = base_h, base_w
        for _ in range(n):
            h = (h + 1) // 2  # 等价于 ceil(h/2)
            w = (w + 1) // 2
            encoder_sizes.append((h, w))

        # 最底层特征图的尺寸，用于 STAUCell
        final_h, final_w = encoder_sizes[-1]
        self.final_h = final_h
        self.final_w = final_w

        # 计算解码器需要的 output_padding
        decoder_paddings = []
        for i in range(n):
            # 解码器第 i 层的输入尺寸是 encoder_sizes[n-i]，输出尺寸是 encoder_sizes[n-i-1]
            in_h, in_w = encoder_sizes[n - i]
            out_h, out_w = encoder_sizes[n - i - 1]
            # 公式: out = (in - 1)*stride - 2*padding + kernel + output_padding
            # 此处 stride=2, padding=1, kernel=3
            op_h = out_h - (2 * in_h - 1)
            op_w = out_w - (2 * in_w - 1)
            decoder_paddings.append((op_h, op_w))
        # ==============================================================

        cell_list = []
        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            # 修改点 2：使用动态计算出的最底层尺寸
            cell_list.append(
                STAUCell(in_channel, num_hidden[i], final_h, final_w, configs.filter_size,
                         configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []
        for i in range(n - 1):
            op_h, op_w = decoder_paddings[i]  # 修改点 3：使用动态 output_padding
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(op_h, op_w)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            op_h, op_w = decoder_paddings[-1]
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(op_h, op_w)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(n - 1),
                               module=nn.LeakyReLU(0.2))  # 原版这里漏了 relu，我顺手帮你加上了，防止最后一层特征突变
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel,
                      kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames, mask_true):
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]

        # 修改点 4：Forward 中的初始尺寸直接用 init 中算好的最终尺寸
        height = self.final_h
        width = self.final_w

        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None

        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.configs.total_length - 1):
            S_lower = []
            T_lower = []
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + \
                      (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(
                        self.configs.device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                S_lower.append(S_t)
                T_lower.append(T_t[i])
                s_spatial = S_lower[max(len(S_lower) - self.configs.tau, 0):]
                s_spatial = torch.stack(s_spatial, dim=0)
                t_spatial = T_lower[max(len(T_lower) - self.configs.tau, 0):]
                t_spatial = torch.stack(t_spatial, dim=0)
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att, t_spatial, s_spatial)
                T_pre[i].append(T_t[i])

            out = S_t
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(
            1, 0, 2, 3, 4).contiguous()
        return next_frames



# import torch
# import torch.nn as nn
# from core.layers.STAUCell import STAUCell
# import math
#
#
# class RNN(nn.Module):
#     def __init__(self, num_layers, num_hidden, configs):
#         super(RNN, self).__init__()
#         # print(configs.srcnn_tf)
#         self.configs = configs
#         self.frame_channel = configs.patch_size * \
#             configs.patch_size * configs.img_channel
#         self.num_layers = num_layers
#         self.num_hidden = num_hidden
#         self.tau = configs.tau
#         self.cell_mode = configs.cell_mode
#         self.states = ['recall', 'normal']
#         if not self.configs.model_mode in self.states:
#             raise AssertionError
#         # self.time = 2
#         cell_list = []
#
#         width = configs.img_width // configs.patch_size // configs.sr_size
#         height = configs.img_height // configs.patch_size // configs.sr_size
#         # print(width)
#
#         for i in range(num_layers):
#             in_channel = num_hidden[i - 1]
#             cell_list.append(
#                 STAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
#                            configs.stride, self.tau, self.cell_mode)
#             )
#         self.cell_list = nn.ModuleList(cell_list)
#
#         # Encoder
#         n = int(math.log2(configs.sr_size))
#         encoders = []
#         encoder = nn.Sequential()
#         encoder.add_module(name='encoder_t_conv{0}'.format(-1),
#                            module=nn.Conv2d(in_channels=self.frame_channel,
#                                             out_channels=self.num_hidden[0],
#                                             stride=1,
#                                             padding=0,
#                                             kernel_size=1))
#         encoder.add_module(name='relu_t_{0}'.format(-1),
#                            module=nn.LeakyReLU(0.2))
#         encoders.append(encoder)
#         for i in range(n):
#             encoder = nn.Sequential()
#             encoder.add_module(name='encoder_t{0}'.format(i),
#                                module=nn.Conv2d(in_channels=self.num_hidden[0],
#                                                 out_channels=self.num_hidden[0],
#                                                 stride=(2, 2),
#                                                 padding=(1, 1),
#                                                 kernel_size=(3, 3)
#                                                 ))
#             # self.encoder_t.add_module(name='gn_t{0}'.format(i),
#             #                           module=nn.GroupNorm(4, self.frame_channel))
#             encoder.add_module(name='encoder_t_relu{0}'.format(i),
#                                module=nn.LeakyReLU(0.2))
#             encoders.append(encoder)
#         self.encoders = nn.ModuleList(encoders)
#
#         # Decoder
#         decoders = []
#
#         for i in range(n - 1):
#             decoder = nn.Sequential()
#             decoder.add_module(name='c_decoder{0}'.format(i),
#                                module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
#                                                          out_channels=self.num_hidden[-1],
#                                                          stride=(2, 2),
#                                                          padding=(1, 1),
#                                                          kernel_size=(3, 3),
#                                                          output_padding=(1, 1)
#                                                          ))
#             # self.decoder_s.add_module(name='gn_decoder_s{0}'.format(i),
#             #                           module=nn.GroupNorm(4, self.frame_channel))
#             decoder.add_module(name='c_decoder_relu{0}'.format(i),
#                                module=nn.LeakyReLU(0.2))
#             decoders.append(decoder)
#
#         if n > 0:
#             decoder = nn.Sequential()
#             decoder.add_module(name='c_decoder{0}'.format(n - 1),
#                                module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
#                                                          out_channels=self.num_hidden[-1],
#                                                          stride=(2, 2),
#                                                          padding=(1, 1),
#                                                          kernel_size=(3, 3),
#                                                          output_padding=(1, 1)
#                                                          ))
#             decoders.append(decoder)
#         self.decoders = nn.ModuleList(decoders)
#
#         self.srcnn = nn.Sequential(
#             nn.Conv2d(self.num_hidden[-1], self.frame_channel,
#                       kernel_size=1, stride=1, padding=0)
#         )
#         self.merge = nn.Conv2d(
#             self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
#         self.conv_last_sr = nn.Conv2d(
#             self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, frames, mask_true):
#         # print('ok')
#         mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
#         batch_size = frames.shape[0]
#         height = frames.shape[3] // self.configs.sr_size
#         width = frames.shape[4] // self.configs.sr_size
#         frame_channels = frames.shape[2]
#         next_frames = []
#         T_t = []
#         T_pre = []
#         S_pre = []
#         # features = {}
#         # H_t = []
#         x_gen = None
#         for layer_idx in range(self.num_layers):
#             tmp_t = []
#             tmp_s = []
#             if layer_idx == 0:
#                 in_channel = self.num_hidden[layer_idx]
#             else:
#                 in_channel = self.num_hidden[layer_idx - 1]
#             for i in range(self.tau):
#                 tmp_t.append(torch.zeros(
#                     [batch_size, in_channel, height, width]).to(self.configs.device))
#                 tmp_s.append(torch.zeros(
#                     [batch_size, in_channel, height, width]).to(self.configs.device))
#             T_pre.append(tmp_t)
#             S_pre.append(tmp_s)
#
#         for t in range(self.configs.total_length - 1):
#             S_lower = []
#             T_lower = []
#             if t < self.configs.input_length:
#                 net = frames[:, t]
#             else:
#                 time_diff = t - self.configs.input_length
#                 net = mask_true[:, time_diff] * frames[:, t] + \
#                     (1 - mask_true[:, time_diff]) * x_gen
#             frames_feature = net
#             frames_feature_encoded = []
#             for i in range(len(self.encoders)):
#                 frames_feature = self.encoders[i](frames_feature)
#                 frames_feature_encoded.append(frames_feature)
#             if t == 0:
#                 for i in range(self.num_layers):
#                     zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(
#                         self.configs.device)
#                     T_t.append(zeros)
#                     # print('ok')
#             S_t = frames_feature
#             for i in range(self.num_layers):
#                 S_lower.append(S_t)
#                 T_lower.append(T_t[i])
#                 s_spatial = S_lower[max(len(S_lower)-self.configs.tau,0):]
#                 s_spatial = torch.stack(s_spatial, dim=0)
#                 t_spatial = T_lower[max(len(T_lower)-self.configs.tau,0):]
#                 t_spatial = torch.stack(t_spatial, dim=0)
#                 t_att = T_pre[i][-self.tau:]
#                 t_att = torch.stack(t_att, dim=0)
#                 s_att = S_pre[i][-self.tau:]
#                 s_att = torch.stack(s_att, dim=0)
#                 S_pre[i].append(S_t)
#                 T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att, t_spatial, s_spatial)
#                 # if(i==self.num_layers-1 and t>=self.configs.tau):
#                 #     features[str(t)+str(i)] = features_cur_unit
#                 T_pre[i].append(T_t[i])
#                 # S_pre[i].append(S_t)
#             out = S_t
#             # out = self.merge(torch.cat([T_t[-1], S_t], dim=1))
#             frames_feature_decoded = []
#             for i in range(len(self.decoders)):
#                 out = self.decoders[i](out)
#                 # print("ok")
#                 if self.configs.model_mode == 'recall':
#                     # print('unet')
#                     out = out + frames_feature_encoded[-2 - i]
#             # out = self.decoder(out)
#
#             x_gen = self.srcnn(out)
#             next_frames.append(x_gen)
#         next_frames = torch.stack(next_frames, dim=0).permute(
#             1, 0, 2, 3, 4).contiguous()
#         return next_frames
