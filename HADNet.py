import torch
import torch.nn as nn
from .ResNeSt import Bottleneck, ResNetCt
from .DySample import DySample
from .DPGN_Conv import *


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.BatchNorm2d(dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ECAAttention()
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class DPGN(nn.Module):
    def __init__(self):
        super().__init__()
        stem_width = 8
        self.stem = nn.Sequential(
            nn.BatchNorm2d(1, affine=False),
            nn.Conv2d(1, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width * 2),
            nn.PReLU()
        )

        self.d11_conv = Conv_d11()
        self.d12_conv = Conv_d12()
        self.d13_conv = Conv_d13()
        self.d14_conv = Conv_d14()
        self.d21_conv = Conv_d21()
        self.d22_conv = Conv_d22()
        self.d23_conv = Conv_d23()
        self.d24_conv = Conv_d24()

        self.conv_dir1 = DepthwiseSeparableConv2d(16, 16, 3, stride=1, padding=1)
        self.conv_dir2 = DepthwiseSeparableConv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.act_final = nn.PReLU()
        self.act_mid1 = nn.Sigmoid()
        self.act_mid2 = nn.Sigmoid()

    def forward(self, x):
        d11 = self.d11_conv(x)
        d12 = self.d12_conv(x)
        d13 = self.d13_conv(x)
        d14 = self.d14_conv(x)
        d21 = self.d21_conv(x)
        d22 = self.d22_conv(x)
        d23 = self.d23_conv(x)
        d24 = self.d24_conv(x)

        md = d11 * d13 + d12 * d14
        md_2 = d21 * d22 + d23 * d24
        md = self.act_mid1(md)
        md_2 = self.act_mid2(-md_2)

        x_stem = self.stem(x)
        stem_base = x_stem

        x_dir1 = self.conv_dir1(x_stem * md)
        x_dir2 = self.conv_dir2(x_stem * md_2)
        x_dir1 = self.bn1(x_dir1)
        x_dir2 = self.bn2(x_dir2)

        out = x_dir1 + x_dir2 + stem_base
        out = self.act_final(out)
        return out


class EncoderEnhance(nn.Module):
    def __init__(self, inp_num=1, layers=[1, 2, 4, 8], channels=[16, 32, 64, 128],
                 bottleneck_width=16, stem_width=8, **kwargs):
        super().__init__()
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                             radix=2, groups=4, bottleneck_width=bottleneck_width,
                             deep_stem=True, stem_width=stem_width, avg_down=True,
                             avd=True, avd_first=False, layer_parms=channels, **kwargs)

        self.eca_x2 = ECAAttention()
        self.eca_x3 = ECAAttention()
        self.eca_x4 = ECAAttention()

    def forward(self, x):
        feats = self.down(x)
        f1, f2, f3, f4 = feats

        f2_enh = self.eca_x2(f2) + f2
        f3_enh = self.eca_x3(f3) + f3
        f4_enh = self.eca_x4(f4) + f4

        return (f1, f2, f3, f4), (f2_enh, f3_enh, f4_enh)


class ANDU(nn.Module):
    def __init__(self, nb_filter, channels):
        super().__init__()
        self.channels = channels

        self.up_path1_stage1 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        self.up_path1_stage2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.up_path1_stage3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.upsample1_stage1 = DySample(channels[2])
        self.upsample1_stage2 = DySample(channels[1])
        self.upsample1_stage3 = DySample(channels[0])
        self.conv_block1 = self._build_conv_block(ResCBAMBlock, nb_filter[0] * 2, nb_filter[0])

        self.up_path2_stage2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.up_path2_stage3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.upsample2_stage2 = DySample(channels[1])
        self.upsample2_stage3 = DySample(channels[0])
        self.conv_block2 = self._build_conv_block(ResCBAMBlock, nb_filter[0] * 3, nb_filter[0])

        self.up_path3_stage3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.upsample3_stage3 = DySample(channels[0])
        self.conv_block3 = self._build_conv_block(ResCBAMBlock, nb_filter[0] * 4, nb_filter[0])

    def _build_conv_block(self, block, in_ch, out_ch, blocks=1):
        layers = [block(in_ch, out_ch)]
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, base_features, enhanced_features):
        f1, f2, f3, f4 = base_features
        f2_enh, f3_enh, f4_enh = enhanced_features

        path1 = self.up_path1_stage1(f4_enh)
        path1 = f3_enh + self.upsample1_stage1(path1)
        path1_mid = path1

        path1 = self.up_path1_stage2(path1)
        path1 = f2_enh + self.upsample1_stage2(path1)
        path1 = self.up_path1_stage3(path1)
        path1 = torch.cat([f1, self.upsample1_stage3(path1)], dim=1)
        path1_out = self.conv_block1(path1)

        path2 = self.up_path2_stage2(path1_mid)
        path2 = f2_enh + self.upsample2_stage2(path2)
        path2_mid = path2

        path2 = self.up_path2_stage3(path2)
        path2 = torch.cat([f1, path1_out, self.upsample2_stage3(path2)], dim=1)
        path2_out = self.conv_block2(path2)

        path3 = self.up_path3_stage3(path2_mid)
        path3 = torch.cat([f1, path1_out, path2_out, self.upsample3_stage3(path3)], dim=1)
        path3_out = self.conv_block3(path3)

        return path1_out, path2_out, path3_out


class Segment(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.head1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        )

    def forward(self, x1, x2, x3):
        out1 = self.head1(x1)
        out2 = self.head2(x2)
        out3 = self.head3(x3)
        return out1, out2, out3


class HADNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dpgn = DPGN()
        self.encoder_enhance = EncoderEnhance(
            layers=[1, 2, 4, 8],
            channels=[16, 32, 64, 128],
            bottleneck_width=16
        )
        self.andu = ANDU(
            nb_filter=[64, 128, 256, 512],
            channels=[64, 128, 256, 512]
        )
        self.segment = Segment(in_channels=64)

    def forward(self, x):
        dpgn_out = self.dpgn(x)

        base_features, enhanced_features = self.encoder_enhance(dpgn_out)

        andu_out1, andu_out2, andu_out3 = self.andu(base_features, enhanced_features)

        seg1, seg2, seg3 = self.segment(andu_out1, andu_out2, andu_out3)

        return seg1, seg2, seg3
