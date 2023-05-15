#-*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import os
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self ,num_class=10 ,head_payload=False):
        super(CNN ,self).__init__()
        if head_payload:
            self.avg_kernel_size = 6
            self.i_size = 22
        else:
            self.avg_kernel_size = 4
            self.i_size = 16
        self.num_class = num_class
        self.input_space = None
        self.input_size = (self.i_size ,self.i_size ,1)

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(1 ,16 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 16*16*16
            nn.BatchNorm2d(16 ,eps=1e-05 ,momentum=0.9 ,affine=True),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16 ,32 ,kernel_size=3 ,stride=2 ,dilation=1 ,padding=1 ,bias=True)  ,  # 8*8*32
            nn.BatchNorm2d(32 ,eps=1e-05 ,momentum=0.9 ,affine=True),
            nn.ReLU(),
        )

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(32 ,64 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 8*8*64
            nn.BatchNorm2d(64 ,eps=1e-05 ,momentum=0.9 ,affine=True),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 ,128 ,kernel_size=3 ,stride=2 ,dilation=1 ,padding=1 ,bias=True)  ,  # 4*4*128
            nn.BatchNorm2d(128 ,eps=1e-05 ,momentum=0.9 ,affine=True),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size ,stride=2 ,ceil_mode=False  )  # 1*1*128

        self.fc0 = nn.Sequential(
            nn.BatchNorm1d( 1 * 1 *128),
            nn.Dropout(0.5),
            nn.Linear( 1 * 1 *128 ,self.num_class ,bias=True),
            # nn.Linear( 1 * 1 *128 ,2 ,bias=True),
        )

        self.plot = nn.Sequential(   # 将维数压到2维，用于绘图
            nn.BatchNorm1d(1 * 1 * 128),
            nn.Dropout(0.5),
            # nn.Linear(1 * 1 * 128, self.num_class, bias=True),
            nn.Linear( 1 * 1 *128 ,2 ,bias=True),
        )
        self.fc1 = nn.Conv2d(128 ,self.num_class ,kernel_size=1 ,stride=1 ,dilation=1 ,padding=0 ,bias=True)  ,  # 4*4*128

    def features(self ,input_data):
        x = self.conv1_0(input_data)
        x = self.conv1(x)
        x = self.conv2_0(x)
        x = self.conv2(x)
        return x

    def logits(self ,input_data):
        x = self.avg_pool(input_data)
        out = x.view(x.size(0) ,-1)
        x = self.fc0(out)
        plot = self.plot(out)
        smax = nn.Softmax(1)
        plot = smax(plot)
        return x,plot

    def forward(self ,input_data):
        x = self.features(input_data)
        x,plot = self.logits(x)
        return x


class CROSS_CNN(nn.Module):
    """docstring for CROSS_CNN"""
    def __init__(self,num_class=12,head_payload=False):
        super(CROSS_CNN,self).__init__()
        if head_payload:
            self.avg_kernel_size = 3
            self.i_size = 22
        else:
            self.avg_kernel_size = 2
            self.i_size = 16
        self.num_class = num_class
        self.input_size = None
        self.input_size = (self.i_size,self.i_size,1)
        self.p = 1 if self.i_size == 22 else 0

        self.layer_A_conv1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*16
            nn.BatchNorm2d(16,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )
        self.layer_A_pool1 = self.layer_A_pool1 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=2,dilation=1,padding=1,bias=True),#8*8*32
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )
        self.layer_A_conv2 = nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*96
            nn.BatchNorm2d(96,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )
        self.layer_A_pool2 = nn.Sequential(
            nn.Conv2d(192,256,kernel_size=3,stride=2,dilation=1,padding=1,bias=True),#4*4*256
            nn.BatchNorm2d(256,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )

        self.layer_B_conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*32
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )
        self.layer_B_pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#8*8*32
        self.layer_B_conv2 = nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*96
            nn.BatchNorm2d(96,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )
        self.layer_B_pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=self.p,)#4*4*192

        self.global_conv = nn.Sequential(
            nn.Conv2d(448,896,kernel_size=3,stride=2,dilation=1,padding=1,bias=True),#2*2*896
            nn.BatchNorm2d(896,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size,stride=2,ceil_mode=False)#1*1*896

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1*1*896),
            nn.Dropout(0.5),
            nn.Linear(1*1*896,self.num_class,bias=True)
            )
        self.bn = nn.BatchNorm1d(self.num_class)

    def features(self,input_data):
        x_A_conv1 = self.layer_A_conv1(input_data)#16*16*16
        x_B_conv1 = self.layer_B_conv1(input_data)#16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)#16*16*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)#8*8*32

        x_A_cat1 = torch.cat((x_A_pool1,x_B_pool1),1)#8*8*64
        x_B_cat1 = torch.cat((x_B_pool1,x_A_pool1),1)#8*8*64

        x_A_conv2 = self.layer_A_conv2(x_A_cat1)#8*8*96
        x_B_conv2 = self.layer_B_conv2(x_B_cat1)#8*8*96

        x_A_cat2 = torch.cat((x_A_conv2,x_B_conv2),1)#8*8*192
        x_B_cat2 = torch.cat((x_B_conv2,x_A_conv2),1)#8*8*192

        x_A_pool2 = self.layer_A_pool2(x_A_cat2)#4*4*256
        x_B_pool2 = self.layer_B_pool2(x_B_cat2)#4*4*192

        x_global_cat = torch.cat((x_A_pool2,x_B_pool2),1)#4*4*448(256+192)
        x_global_conv = self.global_conv(x_global_cat)#2*2*896 or (3*3*896)

        return x_global_conv

    def forward(self,input_data):
        x = self.features(input_data)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.bn(x)
        return x

# 三级平行模型
class TPCNN(nn.Module):
    def __init__(self, num_class=10, head_payload=False):
        super(TPCNN, self).__init__()
        # 上
        self.uconv1 = nn.Sequential(  #
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.uconv2 = nn.Sequential(  #
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        # 中
        self.mconv1 = nn.Sequential(  #
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        # 下
        self.dconv1 = nn.Sequential(  #
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.uconv3 = nn.Sequential(  #
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.mconv2 = nn.Sequential(  #
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.dconv2 = nn.Sequential(  #
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.uconv4 = nn.Sequential(  #
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.globalconv1 = nn.Sequential(
            nn.Conv2d(896, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU()
        )

        self.dmaxpool = nn.MaxPool2d(kernel_size=2,padding=1)

        #         self.lstm1 = nn.LSTM(256,512, 2)
        #         self.lstm2 = nn.LSTM(self.i_size*2,self.i_size*2, 2)

        self.avpool = nn.AdaptiveAvgPool2d(2)
        #         self.globallstm = nn.LSTM(512, 256, 1)

        self.fc1 = nn.Linear(1024*2*2, 512)
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        # 上
        uout = self.uconv1(x)
        uout = self.uconv2(uout)

        # 中
        mout = self.mconv1(x)

        # 下
        dout = self.dconv1(x)

        # 连接
        # print("uout", uout.shape)
        # print("mout", mout.shape)
        # print("dout", dout.shape)

        out = torch.cat((uout, mout, dout), dim=1)
        # print("out", out.shape)

        # 上
        uout = self.uconv3(out)

        # 中
        mout = self.mconv2(out)
        # 下
        dout = self.dconv2(out)

        # 连接
        # print("uout", uout.shape)
        # print("dout", dout.shape)

        out = torch.cat((uout, dout), dim=1)
        # print("out", out.shape)

        # 上
        uout = self.uconv4(out)

        # 中

        # 下
        dout = self.dmaxpool(out)

        # 连接
        # print("uout", uout.shape)
        # print("mout", mout.shape)
        # print("dout", dout.shape)

        out = torch.cat((uout, mout, dout), dim=1)

        # 最后的网络
        # print("out", out.shape)
        out = self.globalconv1(out)
        out = self.avpool(out)

        # 全连接层
        # print("out", out.shape)
        out=out.view(-1,1024*2*2)
        out = self.fc1(out)
        out = self.fc2(out)

        return out



class CROSS_CNN_LSTM(nn.Module):  # PCCN-LSTM

    def __init__(self, num_class=10, head_payload=False):
        super(CROSS_CNN_LSTM, self).__init__()
        if head_payload:
            self.avg_kernel_size = 3
            self.i_size = 22
        else:
            self.avg_kernel_size = 2
            self.i_size = 16
        self.num_class = num_class
        self.input_size = None
        self.input_size = (self.i_size, self.i_size, 1)
        self.p = 1 if self.i_size == 22 else 0

        self.layer_A_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool1 = self.layer_A_pool1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 8*8*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_conv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*96
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 4*4*256
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.layer_B_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*32
        self.layer_B_conv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*96
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=self.p, )  # 4*4*192

        self.global_conv = nn.Sequential(
            nn.Conv2d(448, 896, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 2*2*896 or 3*3*896
            nn.BatchNorm2d(896, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        # 最后接一个LSTM,输入16，输出64
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=2,batch_first=True)
        self.lstm2= nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*896


        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 896),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 896, self.num_class, bias=True)
        )

    def features(self, input_data):
        x_A_conv1 = self.layer_A_conv1(input_data)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(input_data)  # 16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)  # 8*8*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)  # 8*8*32

        x_A_cat1 = torch.cat((x_A_pool1, x_B_pool1), 1)  # 8*8*64
        x_B_cat1 = torch.cat((x_B_pool1, x_A_pool1), 1)  # 8*8*64

        x_A_conv2 = self.layer_A_conv2(x_A_cat1)  # 8*8*96
        x_B_conv2 = self.layer_B_conv2(x_B_cat1)  # 8*8*96

        x_A_cat2 = torch.cat((x_A_conv2, x_B_conv2), 1)  # 8*8*192
        x_B_cat2 = torch.cat((x_B_conv2, x_A_conv2), 1)  # 8*8*192

        x_A_pool2 = self.layer_A_pool2(x_A_cat2)  # 4*4*256
        x_B_pool2 = self.layer_B_pool2(x_B_cat2)  # 4*4*192

        x_global_cat = torch.cat((x_A_pool2, x_B_pool2), 1)  # 4*4*448(256*192)

        x_global_conv = self.global_conv(x_global_cat)  # 2*2*896 or 3*3*896
        # print(x_global_conv.shape)

        return x_global_conv

    def forward(self, input_data, ):
        x = self.features(input_data)
        x = self.avg_pool(x)   # 1*1*896
        # print(x.shape)
        x_lstm = x.view(x.size(0), -1, 16)  # 56*16
        # print(x_lstm.shape)
        out1, _ = self.lstm(x_lstm)# 56*16
        out2, _ = self.lstm(x_lstm)
        # print(out1.shape)
        # print(out2.shape)
        out=torch.cat((out1,out2),dim=1)
        x = x.view(out.size(0), -1)
        # print(x.shape)
        x = self.fc(x)

        return x

##对比网络之一
class CDLSTM(nn.Module):  # PCCN-LSTM

    def __init__(self, num_class=10, head_payload=False):
        super(CDLSTM, self).__init__()
        # 上
        self.Aconv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.Aconv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.Aconv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.Aconv4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        # 下
        self.Bconv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation = 1, padding = 1, bias = True),  # 16*16*16
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.Bconv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.Bmaxpool = nn.MaxPool2d(kernel_size=2)

        # 最后
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)

        self.global_conv = nn.Sequential(
            nn.Conv2d(756, 896, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 2*2*896 or 3*3*896
            nn.BatchNorm2d(896, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, ceil_mode=False)  # 1*1*896

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 896),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 896, num_class, bias=True)
        )
        self.fc_plot = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 896),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 896, 2, bias=True)
        )




    def forward(self, input_data,):
        Aout = self.Aconv1(input_data)
        # print("1", Aout.shape)
        Aout = self.Aconv2(Aout)
        # print("2", Aout.shape)
        Bout = self.Bconv1(input_data)
        # print("3", Bout.shape)

        out = torch.cat((Aout, Bout), dim=1)
        # print("4", out.shape)

        Aout = self.Aconv3(out)
        # print("5", Aout.shape)
        Bout = self.Bconv2(out)
        # print("6", Bout.shape)

        out = torch.cat((Aout, Bout), dim=1)
        # print("7", out.shape)

        Aout = self.Aconv4(out)
        # print("8", Aout.shape)
        Bout = self.Bmaxpool(out)
        # print("9", Bout.shape)

        out1 = torch.cat((Aout, Bout,Aout), dim=1)
        # print("10", out1.shape)
        out2 = torch.cat((Aout, Bout,Bout), dim=1)
        # print("11", out2.shape)

        out1=out1.view(out1.size(0),-1,16)
        out2=out2.view(out2.size(0),-1,16)
        Aout,_= self.lstm(out1)
        # print("12", Aout.shape)
        Bout,_= self.lstm2(out2)
        # print("13", Bout.shape)

        out = torch.cat((Aout, Bout), dim=1)
        # print("14", out.shape)
        out=torch.reshape(out,[out.size(0),-1,4,4])
        out = self.global_conv(out)
        # print("15", out.shape)
        out = self.avg_pool(out)
        # print("16", out.shape)
        out=out.view(-1,896*1*1)
        plot = self.fc_plot(out)
        smax = nn.Softmax(1)
        plot = smax(plot)
        out = self.fc(out)


        return out,plot

# 平行网络
class p_lstm_cnn(nn.Module):
    def __init__(self,num_class=10,head_payload=False):
        super(p_lstm_cnn, self).__init__()
        if head_payload:
            self.avg_kernel_size = 3
            self.i_size = 22
        else:
            self.avg_kernel_size = 2
            self.i_size = 16
        self.num_class = num_class
        self.num_layer = 2
        self.conv1 = nn.Sequential(    #
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,dilation=1,bias=True),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.globalconv1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512,eps=1e-05,momentum=0.9,affine=True),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(self.i_size,self.i_size*2, 2)
        self.lstm2 = nn.LSTM(self.i_size*2,self.i_size*2, 2)


        self.avpool = nn.AdaptiveAvgPool2d(2)
        self.globallstm = nn.LSTM(512, 256, 1)

        self.fc1 = nn.Linear(256 * 4, 128)
        self.fc2 = nn.Linear(128, self.num_class)

    def feature(self,input_data,):

        # 第一层 3个CNN卷积层
        top_conv1 = self.conv1(input_data)
        top_conv2 = self.conv2(top_conv1)
        top_conv3 = self.conv3(top_conv2)

        # 第二层LSTM
        x = input_data.view(256, -1, self.i_size)
        out,_ = self.lstm1(x)
        out,_ = self.lstm2(out)

        # 将lstm和CNN合并
        bottom_lstm = torch.reshape(out, [top_conv3.size(0), -1, 2, 2])
        # print(bottom_lstm.shape)
        # print(top_conv3.shape)
        x = torch.cat((top_conv3, bottom_lstm), dim=1)

        return x

    def forward(self, input_data,):
        x = self.feature(input_data,)
        x = self.globalconv1(x)

        x = self.avpool(x)
        #         print("x",x.shape)
        x = x.view(16, -1, 512)
        x, _ = self.globallstm(x)
        #print("x",x.shape)
        x = x.view(-1, 256 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


#双向lstm对比网络2
class DILSTM(nn.Module):
 def __init__(self,num_class=12,head_payload=False,n_layers=2,bidirectional=True,drop_prob=0.5):
    super(DILSTM,self).__init__()
    self.n_layers = n_layers
    self.bidirectional = bidirectional
    if head_payload==True:
        self.input_size = 484
    else:
        self.input_size = 256
    self.lstm1= nn.LSTM(self.input_size,self.input_size, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)
    self.lstm2= nn.LSTM(self.input_size*2, 64, n_layers,
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)
    self.dropout = nn.Dropout(0.3)
    if bidirectional:
       self.fc = nn.Linear(64*2, num_class)
    else:
       self.fc = nn.Linear(64, num_class)
    self.sig = nn.Sigmoid()
 def forward(self,x,):
    # print(x.shape)
    x=x.view(x.shape[0],-1,self.input_size)
    out,_=self.lstm1(x)
    # print("out",out.shape)
    out,_=self.lstm2(out)
    # print("out",out.shape)
    out=self.dropout(out)
    out=out.view(-1,1*128)
    out=self.fc(out)
    # print("out",out.shape)
    return out

#双向lstm对比网络2
class LSTM(nn.Module):
    def __init__(self,num_class=12,head_payload=False,n_layers=2,bidirectional=True,drop_prob=0.5):
        super(LSTM,self).__init__()
        self.n_layers = n_layers
        if head_payload==True:
            input_size = 484
        else:
            input_size = 196
        self.lstm1= nn.LSTM(input_size,input_size, n_layers,
                                dropout=drop_prob, batch_first=True,
                                bidirectional=bidirectional)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_class)
        self.sig = nn.Sigmoid()
    def forward(self,x,):
        # print(x.shape)
        x=x.view(256,-1,484)
        out,_=self.lstm1(x)
        # print("out",out.shape)
        # out,_=self.lstm2(out)
        # print("out",out.shape)
        out=self.dropout(out)
        out=out.view(-1,1*128)
        out=self.fc(out)
        # print("out",out.shape)
        return out


class TPCNN_C(nn.Module):
    def __init__(self, num_class=10, head_payload=False):
        super(TPCNN_C, self).__init__()
        # 上
        self.uconv1 = nn.Sequential(  #
         nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
         nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )
        self.uconv2 = nn.Sequential(  #
         nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, dilation=1, bias=True),
         nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )
        # 中
        self.mconv1 = nn.Sequential(  #
         nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0, dilation=1, bias=True),
         nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )
        # 下
        self.dconv1 = nn.Sequential(  #
         nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
         nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=2,padding=0)
        )
        # 上
        self.uconv3 = nn.Sequential(  #
         nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
         nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )
        # 中
        #         self.mconv2 = nn.Sequential(    #
        #             nn.Conv2d(96,128, kernel_size=3, stride=2, padding=1,dilation=1,bias=True),
        #             nn.BatchNorm2d(128,eps=1e-05,momentum=0.9,affine=True),
        #             nn.ReLU(),
        #             nn.MaxPool2d(kernel_size=2)
        #         )
        self.mlstm = nn.LSTM(48, 8, 2)

        # 下
        self.dconv2 = nn.Sequential(  #
         nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
         nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )

        self.uconv4 = nn.Sequential(  #
         nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, dilation=1, bias=True),  ###______
         nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU(),
        )
        self.globalconv1 = nn.Sequential(
         nn.Conv2d(912, 1024, kernel_size=3, stride=1, padding=1),
         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.9, affine=True),
         nn.ReLU()
        )

        self.dmaxpool = nn.MaxPool2d(kernel_size=2)

        #         self.lstm1 = nn.LSTM(256,512, 2)
        #         self.lstm2 = nn.LSTM(self.i_size*2,self.i_size*2, 2)

        self.avpool = nn.AdaptiveAvgPool2d(2)
        #         self.globallstm = nn.LSTM(512, 256, 1)

        self.fc1 = nn.Linear(1024 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        # print(x.shape)
        # 上
        uout = self.uconv1(x)
        uout = self.uconv2(uout)
        # 中
        mout = self.mconv1(x)

        # 下
        dout = self.dconv1(x)

        # 连接
        # print("uout", uout.shape)
        # print("mout", mout.shape)
        # print("dout", dout.shape)
        out = torch.cat((uout, mout, dout), dim=1)
        # print("out", out.shape)

        # 上
        uout = self.uconv3(out)

        # 中
        m = out.view(out.size(0), -1, 48)
        mout, _ = self.mlstm(m)
        # 下
        dout = self.dconv2(out)

        # 连接
        # print("uout",uout.shape)
        # print("mout", mout.shape)
        # print("dout", dout.shape)

        out = torch.cat((uout, dout), dim=1)
        # print("out", out.shape)

        # 上
        uout = self.uconv4(out)
        # print("uout",uout.shape)
        # 中

        # 下
        dout = self.dmaxpool(out)
        # print("dout",dout.shape)
        # 连接
        #

        mout = torch.reshape(mout, [mout.size(0), -1, 2, 2])
        dout = torch.reshape(dout, [mout.size(0), -1, 2, 2])
        uout = torch.reshape(uout, [mout.size(0), -1, 2, 2])
        out = torch.cat((uout, mout, dout), dim=1)

        # 最后的网络
        # print("out", out.shape)
        out = self.globalconv1(out)
        out = self.avpool(out)

        # 全连接层
        # print("out", out.shape)
        out = out.view(-1, 1024 * 2 * 2)
        out = self.fc1(out)
        out = self.fc2(out)

        return out





class HPM(nn.Module):  # HPM
    """
    docstring for PARALLEL_CROSS_CNN_ADD
    """

    def __init__(self, num_class=10, head_payload=False):
        super(HPM, self).__init__()
        if head_payload:
            self.avg_kernel_size = 3
            self.i_size = 22
        else:
            self.avg_kernel_size = 2
            self.i_size = 16
        self.num_class = num_class
        self.input_size = None
        self.input_size = (16, 16, 1)
        self.p = 1 if self.i_size == 22 else 0

        self.layer_A_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*16
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool1 = self.layer_A_pool1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 8*8*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 4*4*128
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.layer_B_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*32
        self.layer_B_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=self.p, )  # 4*4*128

        self.global_conv = nn.Sequential(
            nn.Conv2d(1152, 256, kernel_size=2, stride=2, dilation=1, padding=1, bias=True),  # 2*2*256
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*256
        self.point_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True)


        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 256),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 256, self.num_class, bias=True)
        )

    def features(self, input_data):
        x_A_conv1 = self.layer_A_conv1(input_data)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(input_data)  # 16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)  # 8*8*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)  # 8*8*32

        x_A_add1 = x_A_pool1 + x_B_pool1  # 8*8*32
        x_B_add1 = x_A_pool1 + x_B_pool1  # 8*8*32

        x_A_conv2 = self.layer_A_conv2(x_A_add1)  # 8*8*64
        x_B_conv2 = self.layer_B_conv2(x_B_add1)  # 8*8*64

        x_A_add2 = x_A_conv2 + x_B_conv2  # 8*8*64
        x_B_add2 = x_A_conv2 + x_B_conv2  # 8*8*64

        x_B_add2 = self.point_conv1(x_B_add2)
        x_A_pool2 = self.layer_A_pool2(x_A_add2)  # 4*4*128
        x_B_pool2 = self.layer_B_pool2(x_B_add2)  # 4*4*128

        x_add = x_A_pool2 + x_B_pool2  # 4*4*128
        out1 = torch.cat((x_A_pool2,x_add),dim=1)
        out2 = torch.cat((x_B_pool2,x_add),dim=1)

        out1 = out1.view(out1.size(0), -1, 16)
        out2 = out2.view(out2.size(0), -1, 16)
        Aout, _ = self.lstm(out1)
        # print("12", Aout.shape)
        Bout, _ = self.lstm2(out2)

        out = torch.cat((Aout, Bout), dim=1)
        out = torch.reshape(out, [out.size(0), -1, 4, 4])

        x_gconv = self.global_conv(out)  # 2*2*256

        return x_gconv

    def forward(self, input_data):
        x = self.features(input_data)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

