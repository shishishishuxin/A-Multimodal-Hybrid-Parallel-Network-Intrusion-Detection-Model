#-*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import os
import torch
import torch.nn as nn

#2017label文件
class CNN_NORMAL(nn.Module):
    """docstring for CNN_NORMAL"""
    def __init__(self ,num_class=10 ,head_payload=False):
        super(CNN_NORMAL, self).__init__()
        if head_payload:
            self.avg_kernel_size = 2
            self.i_size = 22
        else:
            self.avg_kernel_size = 2
            self.i_size = 8
        self.num_class = num_class
        self.input_space = None
        self.input_size = (self.i_size ,self.i_size ,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1 ,32 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 8*8*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2 ,padding=0 ,  )  # 4*4*16
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32 ,128 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 4*4*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2 ,padding=0 ,  )  # 2*2*128
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size ,stride=2 ,ceil_mode=False  )  # 1*1*128

        self.fc = nn.Sequential(
            nn.BatchNorm1d( 1 * 1 *128),
            nn.Dropout(0.5),
            nn.Linear( 1 * 1 *128 ,self.num_class ,bias=True)
        )

    def features(self ,input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)

        return x

    def logits(self ,input_data):
        x = self.avg_pool(input_data)
        x = x.view(x.size(0) ,-1)
        x = self.fc(x)

        return x

    def forward(self ,input_data):
        x = self.features(input_data)
        x = self.logits(x)

        return x


class LSTM(nn.Module):
    """docstring for LSTM"""
    def __init__(self, num_class=10 ,head_payload=False):
        super(LSTM, self).__init__()
        if head_payload:
            self.i_size = 8
        else:
            self.i_size = 8
        self.num_class = num_class
        self.input_size = (self.i_size ,self.i_size ,1)
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=self.i_size ,hidden_size=16 ,num_layers=self.num_layers ,batch_first=True
                            ,dropout=0.5)
        self.classifier = nn.Linear(16 ,self.num_class ,bias=True)

    def forward(self ,input_data):
        x = input_data.view(input_data.size(0) ,self.i_size ,self.i_size)
        out ,_ = self.lstm(x)
        out = out[: ,-1 ,:]
        out = self.classifier(out)

        return out

class CNN_LSTM(nn.Module):
	"""docstring for ClassName"""
	def __init__(self, num_class=10,head_payload=False):
		super(CNN_LSTM, self).__init__()
		if head_payload:
			self.i_size = 8
			self.avg_kernel_size = 2
		else:
			self.i_size = 8
			self.avg_kernel_size = 2
		self.num_class = num_class
		self.input_space = None
		self.input_size = (self.i_size,self.i_size,1)
		self.num_layers = 2
		self.conv1 = nn.Sequential(
			nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*16
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#4*4*16
			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#4*4*64
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#2*2*64
			)

		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.avg_kernel_size*self.avg_kernel_size*16,self.i_size*self.i_size,bias=True)  # 输出16*16
			)

		self.lstm = nn.LSTM(input_size=self.i_size,hidden_size=16,num_layers=self.num_layers,batch_first=True,dropout=0.5)
		self.classifier = nn.Linear(16,self.num_class,bias=True)

	def features(self,input_data):
		x = self.conv1(input_data)
		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		x = self.fc(x)
		return x

	def forward(self,input_data):
		'''
		input_data shape: 16*16
		'''
		x = self.features(input_data)
		x = x.view(x.size(0),self.i_size,self.i_size)
		out,_ = self.lstm(x)
		out = out[:,-1,:]
		out = self.classifier(out)

		return out



class Multimoding2017(nn.Module):  # HPM
    """
    docstring for PARALLEL_CROSS_CNN_ADD
    """

    def __init__(self, num_class=10, head_payload=False):
        super(Multimoding2017, self).__init__()
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

        self.convlabel1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1,
                      bias=True),  # 8*8*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 4*4*16
        )
        self.convlabel2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, dilation=1, padding=1,
                      bias=True),  # 4*4*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 2*2*128
        )
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,
                                     stride=2, ceil_mode=False)  # 1*1*128

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 384),     #128+256 =384
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 384, self.num_class, bias=True)
        )

    def featurespcap(self, input_data):
        x = input_data[:,:,:484]
        x = x.view(x.shape[0],1,22,22)
        x_A_conv1 = self.layer_A_conv1(x)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(x)  # 16*16*32
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

    def featureslabel(self,input_data):
        input_data=input_data
        x= input_data[:,:,484:]
        x = x.view(x.shape[0],1,8,8)
        x = self.convlabel1(x)   #4*4*16
        x = self.convlabel2(x)   #2*2*128
        return x

    def forward(self, input_data):

        x2 = self.featureslabel(input_data)
        x1 = self.featurespcap(input_data)
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool2(x2)
        x = torch.cat((x1,x2),dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Multimoding2017noFCN(nn.Module):  # HPM
    """
    docstring for PARALLEL_CROSS_CNN_ADD
    """

    def __init__(self, num_class=10, head_payload=False):
        super(Multimoding2017noFCN, self).__init__()
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
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )
        self.layer_A_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=self.p, )  # 4*4*128

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

        self.convlabel1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1,
                      bias=True),  # 8*8*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 4*4*16
        )
        self.convlabel2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, dilation=1, padding=1,
                      bias=True),  # 4*4*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 2*2*128
        )
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,
                                     stride=2, ceil_mode=False)  # 1*1*128

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 384),     #128+256 =384
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 384, self.num_class, bias=True)
        )

    def featurespcap(self, input_data):
        x = input_data[:,:,:484]
        x = x.view(x.shape[0],1,22,22)
        x_A_conv1 = self.layer_A_conv1(x)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(x)  # 16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)  # 8*8*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)  # 8*8*32

        x_A_add1 = x_A_pool1 + x_B_pool1  # 8*8*32
        x_B_add1 = x_A_pool1 + x_B_pool1  # 8*8*32

        x_A_conv2 = self.layer_A_conv2(x_A_add1)  # 8*8*64
        x_B_conv2 = self.layer_B_conv2(x_B_add1)  # 8*8*64

        x_A_add2 = x_A_conv2 + x_B_conv2  # 8*8*64
        x_B_add2 = x_A_conv2 + x_B_conv2  # 8*8*64
        x_A_add2 = self.point_conv1(x_A_add2)
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

    def featureslabel(self,input_data):
        input_data=input_data
        x= input_data[:,:,484:]
        x = x.view(x.shape[0],1,8,8)
        x = self.convlabel1(x)   #4*4*16
        x = self.convlabel2(x)   #2*2*128
        return x

    def forward(self, input_data):

        x2 = self.featureslabel(input_data)
        x1 = self.featurespcap(input_data)
        x1 = self.avg_pool(x1)
        x2 = self.avg_pool2(x2)
        x = torch.cat((x1,x2),dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
