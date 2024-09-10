from math import gcd
import time
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MultiAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp(x)
        x = identity + x
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x
           
class TriScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(TriScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=5,padding=5),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=3,padding=3),
            nn.PReLU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel*4,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        out = torch.cat([x,x1,x2,x3],dim=1)
        out = self.merge(out)
        return x+out

class DualScaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(DualScaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=4,padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.merge(out)
        return x + out if self.res else out
    
class DynamicConv(nn.Module):
    def __init__(self, inchannels, mode='highPass', dilation=0, kernel_size=3, stride=1, kernelNumber=8):
        super(DynamicConv, self).__init__()
        self.stride = stride
        self.mode = mode
        self.kernel_size = kernel_size
        self.kernelNumber = inchannels
        self.conv = nn.Conv2d(inchannels, self.kernelNumber*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernelNumber*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.unfoldMask = []
        self.unfoldSize = kernel_size + dilation * (kernel_size - 1)
        self.pad = nn.ReflectionPad2d(self.unfoldSize//2)
        if mode == 'lowPass':
            for i in range(self.unfoldSize):
                for j in range(self.unfoldSize):
                    if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
                        self.unfoldMask.append(i * self.unfoldSize + j)
        elif mode != 'highPass':
            raise ValueError("Invalid mode. Expected 'lowPass' or 'highPass'.")
        
    def forward(self, x):
        copy = x
        filter = self.ap(x)
        filter = self.conv(filter)
        filter = self.bn(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.unfoldSize).reshape(n, self.kernelNumber, c//self.kernelNumber, self.unfoldSize**2, h*w)
        if self.mode == 'lowPass':
            x = x[:,:,:,self.unfoldMask,:]
        n,c1,p,q = filter.shape
        filter = filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        filter = self.act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out, copy - out

class localFusionBlock(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusionBlock, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels*3,in_channels*6,kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels*6,in_channels,kernel_size=1)
    def forward(self,x,a,b):
        out = self.conv1(torch.cat([x,a,b],dim=1))
        out = self.act(out)
        out = self.conv2(out)
        return x + out
    
class localFusion(nn.Module):  
    def __init__(self, in_channels):  
        super(localFusion, self).__init__() 
        self.conv1 = nn.ModuleList([
            CCAM(in_channels) for i in range(8)
        ])

    def forward(self,x0,x1,x2,x3,x4,x5,x6,x7):
        (x0,x1,x2,x3,x4,x5,x6,x7) =  (self.conv1[0](x0,x1,x2)+x0,self.conv1[1](x1,x0,x2)+x1,
                                      self.conv1[2](x2,x1,x3)+x2,self.conv1[3](x3,x2,x4)+x3,
                                      self.conv1[4](x4,x3,x5)+x4,self.conv1[5](x5,x4,x6)+x5,
                                      self.conv1[6](x6,x5,x7)+x6,self.conv1[7](x7,x6,x5)+x7
                                      )

        out = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7],dim=1)
        return out

class CCAM(nn.Module):
    def __init__(self,inchannels):
        super(CCAM, self).__init__()
        self.inchannels=inchannels
        self.fc = nn.Linear(inchannels*3, inchannels*3*inchannels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, z):
        m = torch.cat((x, y, z), dim=1)  # n, 12, h, w
        gap = F.adaptive_avg_pool2d(m, (1, 1))  # n, 12, 1, 1
        gap = gap.view(m.size(0), self.inchannels*3)  # n, 12
        fc_out = self.fc(gap)  # n, 48
        conv1d_input = fc_out.unsqueeze(1)  # n, 1, 48
        conv1d_out = self.conv1d(conv1d_input)  # n, 1, 48
        conv1d_out = conv1d_out.view(m.size(0), self.inchannels*3, self.inchannels)  # n, 12, 4
        softmax_out = self.softmax(conv1d_out)  # n, 12, 4
        out = torch.einsum('nchw,ncm->nmhw', (m, softmax_out))  # n, 4, h, w
        
        return out

class DFS(nn.Module):
    def __init__(self, in_channels,outchannel,basechannel, mergeNum,res = 1,attn=False):
        super(DFS, self).__init__()
        self.mergeNum = mergeNum
        self.low_pass_filter = DynamicConv(in_channels,mode='highPass')
        self.enlarger = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1)
        )
        self.fe = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,basechannel//8,kernel_size=1),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
            TriScaleConv(basechannel//8,basechannel//8,res=True),
        )
    def forward(self,x):
        low,high = self.low_pass_filter(x)
        low = self.fe(low)
        high = self.enlarger(high)
        return high,low

class DFLSBlock(nn.Module):  
    def __init__(self, in_channels,split=3):  
        super(DFLSBlock, self).__init__()  

        self.split = split
        self.frequency_enlarge = DualScaleConv(in_channels,in_channels,res=True) 

        self.blocks = nn.ModuleList([
            DFS(in_channels,in_channels*7//8,in_channels,0,res=1),
            DFS(in_channels*7//8,in_channels*6//8,in_channels,in_channels*2//8,res=1,attn=True),
            DFS(in_channels*6//8,in_channels*5//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*5//8,in_channels*4//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*4//8,in_channels*3//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*3//8,in_channels*2//8,in_channels,in_channels*3//8,res=1),
            DFS(in_channels*2//8,in_channels*1//8,in_channels,in_channels*3//8,res=1),
        ])

        self.local = localFusion(in_channels//8)

        self.synthesizer = nn.Sequential(
            MultiAttn(in_channels),
        )
        self.merger = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
        self.merger2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
          
    def forward(self, m):  
        m0 = self.frequency_enlarge(m) 

        m1,x0 = self.blocks[0](m0)
        m1,x1 = self.blocks[1](m1)
        m1,x2 = self.blocks[2](m1)
        m1,x3 = self.blocks[3](m1)
        m1,x4 = self.blocks[4](m1)
        m1,x5 = self.blocks[5](m1)
        x7,x6 = self.blocks[6](m1)

        m2 = self.local(x0,x1,x2,x3,x4,x5,x6,x7)
        m2 = self.merger(m2)
        m2 = m2 + m0
        out = self.synthesizer(m2)
        out = self.merger2(out)
        out = out + m2
        return out

class DFLSNet(nn.Module):
    def __init__(self, num_res=4):
        super(DFLSNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            DFLSBlock(base_channel),
            DFLSBlock(base_channel*2),
            DFLSBlock(base_channel*4),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DFLSBlock(base_channel * 4),
            DFLSBlock(base_channel * 2),
            DFLSBlock(base_channel)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs
    
def build_net():
    return DFLSNet()
