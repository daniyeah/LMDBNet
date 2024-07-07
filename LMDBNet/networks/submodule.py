import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np


class up(nn.Module):
    def __init__(self, scale=(2, 2)):
        super(up, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x



class Deep_conv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1, dilation=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1, bias=False, group=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, dilation=dilation, bias=bias, groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Cross_cat(nn.Module):
    # 一层一层拼接
    def __init__(self, in_c):
        super().__init__()
        self.conv = Conv(in_c * 2, in_c, group=in_c)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        output = torch.stack([x1, x2], dim=2, ).reshape(b, -1, h, w)  # 作用与前一行相似，速度快一点，但是要是想输入三个张量，务必验证！！！
        output = self.conv(output)
        return output


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = Conv(inp, init_channels, kernel_size, kernel_size // 2, )
        self.cheap_operation = Conv(init_channels, new_channels, dw_size, padding=dw_size // 2, group=init_channels)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]



class Global_feature(nn.Module):
    # FSIM
    def __init__(self, mid_ch, c_list, ):
        super(Global_feature, self).__init__()
        self.up2 = up((2, 2))
        self.up4 = up((4, 4))
        self.t1_conv3 = nn.Sequential(nn.MaxPool2d(4, 4), Conv(c_list[0], mid_ch, ), )
        self.t2_conv3 = nn.Sequential(nn.MaxPool2d(2, 2), Conv(c_list[1], mid_ch, ), )
        self.t3_conv3 = Conv(c_list[2], mid_ch, )
        self.t4_conv3 = Conv(c_list[3], mid_ch, )
        self.last_conv = Conv(mid_ch * 4, mid_ch, )

        self.global_t1 = nn.Sequential(Conv(mid_ch, c_list[0]), up((4, 4)))
        self.global_t2 = nn.Sequential(Conv(mid_ch, c_list[1]), up((2, 2)))
        self.global_t3 = Conv(mid_ch, c_list[2])
        self.global_t4 = nn.Sequential(nn.MaxPool2d(2, 2), Conv(mid_ch, c_list[3]))

        self.deep_conv = Deep_conv(mid_ch)

    def forward(self, t1, t2, t3, t4):
        t1_mid = self.t1_conv3(t1)
        t2_mid = self.t2_conv3(t2)
        t3_mid = self.t3_conv3(t3)
        t4_mid = self.t4_conv3(self.up2(t4))
        global_feature = self.last_conv(torch.cat([t1_mid, t2_mid, t3_mid, t4_mid], 1))

        global_t1 = self.global_t1(global_feature) + t1
        global_t2 = self.global_t2(global_feature) + t2
        global_t3 = self.global_t3(global_feature) + t3
        global_t4 = self.global_t4(global_feature) + t4

        deep_out = self.deep_conv(self.up4(global_feature))
        return deep_out, global_t1, global_t2, global_t3, global_t4



class Decoder(nn.Module):
    def __init__(self, c_list):
        super(Decoder, self).__init__()
        self.decoder1 = DoubleConv(c_list[3], c_list[2])
        self.decoder2 = DoubleConv(c_list[3], c_list[1])
        self.decoder3 = DoubleConv(c_list[2], c_list[0])
        self.decoder4 = DoubleConv(c_list[1], c_list[0] // 2)
        self.decoder5 = Deep_conv(c_list[0] // 2)
        self.up = up()

    def forward(self, t1, t2, t3, t4):
        out4 = self.decoder1(t4)
        cat_out4 = torch.cat([self.up(out4), t3], dim=1)

        out3 = self.decoder2(cat_out4)
        cat_out3 = torch.cat([self.up(out3), t2], dim=1)

        out2 = self.decoder3(cat_out3)
        cat_out2 = torch.cat([self.up(out2), t1], dim=1)

        out1 = self.decoder4(cat_out2)
        out = self.decoder5(out1)
        return out, out1, out2, out3, out4

class LastConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LastConv, self).__init__()
        self.Conv1 = Conv(in_ch, out_ch // 4)
        self.Conv3 = Conv(in_ch, out_ch // 4, dilation=3, padding=3)
        self.Conv5 = Conv(in_ch, out_ch // 4, dilation=5, padding=5)
        self.Conv7 = Conv(in_ch, out_ch // 4, dilation=7, padding=7)
        self.last_conv = Conv(out_ch, out_ch, kernel=1, padding=0)

    def forward(self, input):
        x1 = self.Conv1(input)
        x3 = self.Conv3(input)
        x5 = self.Conv5(input)
        x7 = self.Conv7(input)
        out = self.last_conv(torch.cat([x1, x3, x5, x7], 1))
        return out



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        # Ghost
        self.Conv = nn.Sequential(
            GhostModule(in_ch, out_ch),
            GhostModule(out_ch, out_ch), )

    def forward(self, input):
        return self.Conv(input)
