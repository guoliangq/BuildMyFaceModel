import torch
import torch.nn as nn
import torch.nn.functional as F

#将一个张量展平
class Flatten_Block(nn.Module):

    def forward(self,x):
        return x.view(x.shape[0],-1)

#卷积操作再加上一个BN层
class ConvBn_Block(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1,1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self,x):
        return self.net(x)

#就是ConvBn块再加上一个PReLU激活层
class ConvBnPrelu_Block(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1,1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn_Block(in_c, out_c, kernel, stride,padding=padding, groups=groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)

#由ConvBnPrelu块加上ConvBn组成
#按通道进行卷积，实现了高效计算
class DepthWise_Block(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3,3), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu_Block(in_c, groups, kernel=(1,1), stride=1, padding=0),
            #groups = groups 按通道进行卷积操作，实现了高校计算
            ConvBnPrelu_Block(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn_Block(groups, out_c, kernel=(1,1), stride=1, padding=0)
        )

    def forward(self,x):
        return self.net(x)

#在DepthWise块的基础上加了原始输入
class DepthWiseRes_Block(nn.Module):

    def __init__(self, in_c, out_c, kernel=(3,3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise_Block(in_c, out_c, kernel, stride, padding, groups)

    def forward(self,x):
        return x + self.net(x)

#由多个DepthWiseRes_Block组成，个数有传入的num_block决定
#由于DepthWiseRes的输入输出通道个数一样的，所以堆积多少个都不会变化
class MultiDepthWiseRes_Block(nn.Module):

    def __init__(self, num_block, channels, kernel=(3,3), stride=1, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(*[
            DepthWiseRes_Block(channels,channels,kernel,stride,padding,groups)
            for _ in range(num_block)
        ])

    def forward(self,x):
        return self.net(x)

#由上面的六种块最后拼接成最后的cnn网络
class FaceNet(nn.Module):

    def __init__(self,embedding_size):
        super().__init__()
        self.conv1 = ConvBnPrelu_Block(1,64,kernel=(3,3),stride=2,padding=1)
        self.conv2 = ConvBn_Block(64,64,kernel=(3,3),stride=1,padding=1,groups=64)
        self.conv3 = DepthWise_Block(64,64,kernel=(3,3),stride=2,padding=1,groups=128)
        self.conv4 = MultiDepthWiseRes_Block(num_block=4,channels=64,kernel=3,stride=1,padding=1,groups=128)
        self.conv5 = DepthWise_Block(64,128,kernel=(3,3),stride=2,padding=1,groups=256)
        self.conv6 = MultiDepthWiseRes_Block(num_block=6,channels=128,kernel=(3,3),stride=1,padding=1,groups=256)
        self.conv7 = DepthWise_Block(128,128,kernel=(3,3),stride=2,padding=1,groups=512)
        self.conv8 = MultiDepthWiseRes_Block(num_block=2,channels=128,kernel=(3,3),stride=1,padding=1,groups=256)
        self.conv9 = ConvBnPrelu_Block(128,512,kernel=(1,1))
        self.conv10 = ConvBn_Block(512,512,groups=512,kernel=(7,7))
        self.flatten = Flatten_Block()
        self.linear = nn.Linear(2048,embedding_size,bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out

#真假脸识别
class SpoofFaceNet(nn.Module):

    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.conv1 = ConvBnPrelu_Block(1,64,kernel=(3,3),stride=2,padding=1)
        self.conv2 = ConvBn_Block(64,64,kernel=(3,3),stride=1,padding=1,groups=64)
        self.conv3 = DepthWise_Block(64,64,kernel=(3,3),stride=2,padding=1,groups=128)
        self.conv4 = MultiDepthWiseRes_Block(num_block=4,channels=64,kernel=3,stride=1,padding=1,groups=128)
        self.conv5 = DepthWise_Block(64,128,kernel=(3,3),stride=2,padding=1,groups=256)
        self.conv6 = MultiDepthWiseRes_Block(num_block=6,channels=128,kernel=(3,3),stride=1,padding=1,groups=256)
        self.conv7 = DepthWise_Block(128,128,kernel=(3,3),stride=2,padding=1,groups=512)
        self.conv8 = MultiDepthWiseRes_Block(num_block=2,channels=128,kernel=(3,3),stride=1,padding=1,groups=256)
        self.conv9 = ConvBnPrelu_Block(128,512,kernel=(1,1))
        self.conv10 = ConvBn_Block(512,512,groups=512,kernel=(7,7))
        self.flatten = Flatten_Block()
        self.linear = nn.Linear(2048,embedding_size,bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out

#测试
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    x = Image.open("D:\\data\\CASIA-WebFace\\0000045\\001.jpg").convert('L')
    x = x.resize((128,128))
    x = np.asarray(x,dtype=np.float32)
    x = x[None,None,...]
    x = torch.from_numpy(x)
    net = FaceNet(512)
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)