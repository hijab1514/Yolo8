import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, groups=1, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activate else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

#c2f BLOCK
   # 2.1 Bottleneck: stack of 2 conv with shortcut connection (True/false)      

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x_in = x  # save original input for residual connection
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut and self.in_channels == self.out_channels:
            x = x + x_in  # element-wise addition for residual
        return x

# 2.2 c2f: conv + bottleneck + conv 
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcuts=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Sequence of Bottleneck layers
        self.m = nn.ModuleList([
            Bottleneck(self.mid_channels, self.mid_channels, shortcut=shortcuts)
            for _ in range(num_bottlenecks)
        ])

        self.conv2 = Conv((num_bottlenecks + 2) * self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)  # Shape: [B, C_out, H, W]
        
        # Split into two halves along channel dimension
        x1, x2 = torch.split(y, self.mid_channels, dim=1)  # each: [B, C_out/2, H, W]

        outputs = [x1, x2]

        for bottleneck in self.m:
            x1 = bottleneck(x1)  # Apply bottleneck to x1
            outputs.append(x1)   # Accumulate outputs

        y = torch.cat(outputs, dim=1)  # Concat along channel dimension
        out = self.conv2(y)           # Final 1x1 Conv to fuse channels

        return out
# sanity check
# Create the C2f module
c2f = C2f(in_channels=64, out_channels=128, num_bottlenecks=2)

# Print total number of parameters (in millions)
print(f"{sum(p.numel() for p in c2f.parameters()) / 1e6:.2f} million parameters")

# Create dummy input of shape (batch=1, channels=64, height=244, width=244)
dummy_input = torch.rand((1, 64, 244, 244))

# Pass through the model
dummy_output = c2f(dummy_input)

# Print output shape
print("output shape", dummy_output.shape)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        # kernel_size = size of maxpool
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # concatenate outputs of maxpool and feed to conv2
        self.conv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # maxpool is applied at 3 different scales
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)

        # applying maxpooling at different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # concatenate
        y = torch.cat([x, y1, y2, y3], dim=1)

        # final conv
        y = self.conv2(y)
        return y

    #sanity check 
import torch
import torch.nn as nn

# YOLO version scaling parameters
def yolo_parmas(version):
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

# Backbone definition
class YOLOv8Backbone(nn.Module):
    def __init__(self, version, in_channel=3, shortcut=True):
        super().__init__()
        d, w, r = yolo_parmas(version)

        # Convolutional layers
        self.conv_0 = Conv(in_channel, int(64 * w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1)

        # C2f layers
        self.c2f_2 = C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcuts=shortcut)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcuts=shortcut)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcuts=shortcut)
        self.c2f_8 = C2f(int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d), shortcuts=shortcut)

        # SPPF
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x)  # stage-1 output

        x = self.conv_5(out1)
        out2 = self.c2f_6(x)  # stage-2 output

        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)   # stage-3 output

        return out1, out2, out3

# Choose the version (e.g., 'n' for nano, 's' for small, etc.)
model = YOLOv8Backbone(version='n')

# Create dummy input with shape (batch=1, channels=3, height=640, width=640)
dummy_input = torch.rand((1, 3, 640, 640))

# Pass through the model
out1, out2, out3 = model(dummy_input)

# Print the output shapes
print("Stage 1 Output (Small):", out1.shape)  # Expected ~ [1, C1, 80, 80]
print("Stage 2 Output (Medium):", out2.shape) # Expected ~ [1, C2, 40, 40]
print("Stage 3 Output (Large):", out3.shape)  # Expected ~ [1, C3, 20, 20]
