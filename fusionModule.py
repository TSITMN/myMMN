import torch
from torch import nn

class senet(nn.Module):
    def __init__(self, channel, ration=16):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ration, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c ,h, w --> b, c, 1, 1
        avg = self.avg_pool(x).view([b,c])
        fc = self.fc(avg).view([b, c, 1, 1])

        return x * fc



#通道注意力
class channel_attention(nn.Module):
    def __init__(self, channel, ration=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel//ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])
        max_pool = self.max_pool(x).view([b, c])

        avg_fc = self.fc(avg_pool)
        max_fc = self.fc(max_pool)

        out = self.sigmoid(max_fc+avg_fc).view([b, c, 1, 1])
        return x * out

#空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #通道的最大池化
        max_pool = torch.max(x, dim=1, keepdim=True).values
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool_out)
        out = self.sigmoid(conv)

        return out * x

#将通道注意力和空间注意力进行融合
class CBAM(nn.Module):
    def __init__(self, channel, ration=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)

        return out


class eca_block(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+  b)/gamma))
        kernel_size = kernel_size if kernel_size % 2  else kernel_size+1
        padding = kernel_size//2
        self.avg_pool =nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #变成序列的形式
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return  out * x


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        #b,c,h,w
        _, _, h, w = x.size()
        #(b, c, h, w) --> (b, c, h, 1)  --> (b, c, 1, h)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        #(b, c, h, w) --> (b, c, 1, w)
        x_w = torch.mean(x, dim=2, keepdim=True)
        #(b, c, 1, w) cat (b, c, 1, h) --->  (b, c, 1, h+w)
        #(b, c, 1, h+w) ---> (b, c/r, 1, h+w)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h,x_w), 3))))
        #(b, c/r, 1, h+w) ---> (b, c/r, 1, h)  、 (b, c/r, 1, w)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h,w], 3)
        #(b, c/r, 1, h) ---> (b, c, h, 1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        #(b, c/r, 1, w) ---> (b, c, 1, w)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        #s_h往宽方向进行扩展， s_w往高方向进行扩展
        out = (s_h.expand_as(x) * s_w.expand_as(x)) * x

        return out


