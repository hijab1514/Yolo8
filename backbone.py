import torch
import torch.nn as nn

# Conv Block
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 2.1 Bottleneck: stack of 2 conv with shortcut connection (True/false)  
# Bottleneck Block
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity if self.shortcut else x

# 2.2 c2f: conv + bottleneck + conv 
# C2f Block
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcuts=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.mid_channels, self.mid_channels, shortcut=shortcuts)
            for _ in range(num_bottlenecks)
        ])
        self.conv2 = Conv((2 + num_bottlenecks) * self.mid_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        x1, x2 = torch.split(y, self.mid_channels, dim=1)
        outputs = [x1, x2]
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
            outputs.append(x1)
        y = torch.cat(outputs, dim=1)
        return self.conv2(y)

# SPPF Block
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

# YOLO Scaling Parameters
def yolo_params(version):
    if version == 'n':
        return 1/3, 1/4, 2.0
    elif version == 's':
        return 1/3, 1/2, 2.0
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0

# YOLOv8 Backbone Definition
class YOLOv8Backbone(nn.Module):
    def __init__(self, version='n', in_channel=3, shortcut=True):
        super().__init__()
        d, w, r = yolo_params(version)
        self.conv_0 = Conv(in_channel, int(64 * w), stride=2)
        self.conv_1 = Conv(int(64 * w), int(128 * w), stride=2)
        self.c2f_2 = C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcuts=shortcut)
        self.conv_3 = Conv(int(128 * w), int(256 * w), stride=2)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcuts=shortcut)
        self.conv_5 = Conv(int(256 * w), int(512 * w), stride=2)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcuts=shortcut)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), stride=2)
        self.c2f_8 = C2f(int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d), shortcuts=shortcut)
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x)  # 80x80 for 640 input
        x = self.conv_5(out1)
        out2 = self.c2f_6(x)  # 40x40
        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)   # 20x20
        return out1, out2, out3

# Run a sanity check
if __name__ == "__main__":
    x = torch.randn(1, 3, 640, 640)
    model = YOLOv8Backbone(version='n')
    o1, o2, o3 = model(x)
    print("Output shapes:")
    print("out1:", o1.shape)
    print("out2:", o2.shape)
    print("out3:", o3.shape)
