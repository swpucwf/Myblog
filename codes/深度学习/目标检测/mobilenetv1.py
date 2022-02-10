import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(Conv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2d(x)


class ConvDw(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ConvDw, self).__init__()
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                      stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_dw(x)


class MobilenetV1(nn.Module):
    def __init__(self):
        super(MobilenetV1, self).__init__()
        self.feature = nn.Sequential(
            Conv(3, 32, 2),
            ConvDw(32, 64, 1),
            ConvDw(64, 128, 2),
            ConvDw(128, 128, 1),
            ConvDw(128, 256, 2),
            ConvDw(256, 256, 1),
            ConvDw(256, 512, 2),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 1024, 2),
            ConvDw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    net = MobilenetV1()
    res = net(a)
    print(res.shape)
