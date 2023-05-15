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
            nn.Linear( 1 * 1 *128 ,self.num_class ,bias=True)
        )

        self.fc1 = nn.Conv2d(128 ,num_class ,kernel_size=1 ,stride=1 ,dilation=1 ,padding=0 ,bias=True)  ,  # 4*4*128

    def features(self ,input_data):
        x = self.conv1_0(input_data)
        x = self.conv1(x)
        x = self.conv2_0(x)
        x = self.conv2(x)
        return x

    def logits(self ,input_data):
        x = self.avg_pool(input_data)
        x = x.view(x.size(0) ,-1)
        x = self.fc0(x)
        return x

    def forward(self ,input_data):
        x = self.features(input_data)
        x = self.logits(x)
        return x

class CNN_NORMAL(nn.Module):
    """docstring for CNN_NORMAL"""
    def __init__(self ,num_class=10 ,head_payload=False):
        super(CNN_NORMAL, self).__init__()
        if head_payload:
            self.avg_kernel_size = 6
            self.i_size = 22
        else:
            self.avg_kernel_size = 4
            self.i_size = 16
        self.num_class = num_class
        self.input_space = None
        self.input_size = (self.i_size ,self.i_size ,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1 ,32 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 16*16*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2 ,padding=0 ,  )  # 8*8*16
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32 ,128 ,kernel_size=3 ,stride=1 ,dilation=1 ,padding=1 ,bias=True)  ,  # 8*8*128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 ,stride=2 ,padding=0 ,  )  # 4*4*128
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
            self.i_size = 22
        else:
            self.i_size = 16
        self.num_class = num_class
        self.input_size = (self.i_size ,self.i_size ,1)
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=self.i_size ,hidden_size=64 ,num_layers=self.num_layers ,batch_first=True
                            ,dropout=0.5)
        self.classifier = nn.Linear(64 ,self.num_class ,bias=True)

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
			self.i_size = 22
			self.avg_kernel_size = 5
		else:
			self.i_size = 16
			self.avg_kernel_size = 4
		self.num_class = num_class
		self.input_space = None
		self.input_size = (self.i_size,self.i_size,1)
		self.num_layers = 2
		self.conv1 = nn.Sequential(
			nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*16
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#8*8*16
			)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*64
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#4*4*64
			)

		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(self.avg_kernel_size*self.avg_kernel_size*64,self.i_size*self.i_size,bias=True)  # 输出16*16
			)

		self.lstm = nn.LSTM(input_size=self.i_size,hidden_size=64,num_layers=self.num_layers,batch_first=True,dropout=0.5)
		self.classifier = nn.Linear(64,self.num_class,bias=True)

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

class PARALLELNETS(nn.Module):  # 单纯的并行
	"""docstring for PARALLELNETS"""
	def __init__(self, num_class=10,head_payload=False):
		super(PARALLELNETS, self).__init__()
		if head_payload:
			self.avg_kernel_size = 6
			self.i_size = 22
		else:
			self.avg_kernel_size = 4
			self.i_size = 16
		self.num_class = num_class
		self.input_space = None
		self.input_size = (self.i_size,self.i_size,1)
		self.p = 1 if self.i_size == 22 else 0

		self.layer_A_conv1 = nn.Sequential(
			nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*16
			nn.BatchNorm2d(16,eps=1e-05,momentum=0.9,affine=True),
			nn.ReLU(),
			)

		self.layer_A_pool1 = nn.Sequential(
			nn.Conv2d(16,32,kernel_size=3,stride=2,dilation=1,padding=1,bias=True),#8*8*32
			nn.BatchNorm2d(32,eps=1e-05,momentum=0.9,affine=True),
			nn.ReLU(),
			)

		self.layer_A_conv2 = nn.Sequential(
			nn.Conv2d(32,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*64
			nn.BatchNorm2d(64,eps=1e-05,momentum=0.9,affine=True),
			nn.ReLU(),
			)

		self.layer_A_pool2 = nn.Sequential(
			nn.Conv2d(64,128,kernel_size=3,stride=2,dilation=1,padding=1,bias=True),#4*4*128
			nn.BatchNorm2d(128,eps=1e-05,momentum=0.9,affine=True),
			nn.ReLU(),
			)

		self.layer_B_conv1 = nn.Sequential(
			nn.Conv2d(1,32,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*32
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#8*8*16
			)

		self.layer_B_conv2 = nn.Sequential(
			nn.Conv2d(32,128,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*128
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2,padding=self.p,)#4*4*128
			)

		self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size,stride=2,ceil_mode=False)#1*1*128

		self.fc0 = nn.Sequential(
			nn.BatchNorm1d(1*1*128),
			nn.Dropout(0.5),
			nn.Linear(1*1*128,self.num_class,bias=True)
			)

		self.fc1 = nn.Conv2d(128,num_class,kernel_size=1,stride=1,dilation=1,padding=0,bias=True)#4*4*128


	def features(self,input_data):
		"""
		"""
		x_A = self.layer_A_conv1(input_data)
		x_A = self.layer_A_pool1(x_A)
		x_A = self.layer_A_conv2(x_A)
		x_A = self.layer_A_pool2(x_A) #4*4*128

		x_B = self.layer_B_conv1(input_data)
		x_B = self.layer_B_conv2(x_B) #4*4*128

		x = x_A + x_B

		return x

	def logits(self,input_data):
		x = self.avg_pool(input_data)
		x = x.view(x.size(0),-1)
		x = self.fc0(x)

		return x


	def forward(self,input_data):
		x = self.features(input_data)
		x = self.logits(x)

		return x


class PARALLEL_CROSS_CNN(nn.Module):  # PPCCN
    """
    docstring for PARALLEL_CROSS_CNN
    """

    def __init__(self, num_class=10, head_payload=False):
        super(PARALLEL_CROSS_CNN, self).__init__()
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 4*4*256
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
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
            nn.Conv2d(128, 160, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 2*2*160
            nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.point_conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.point_conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.point_conv3 = nn.Conv2d(160, 128, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*160

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 160),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 160, self.num_class, bias=True)
        )

    def features(self, input_data):
        x_A_conv1 = self.layer_A_conv1(input_data)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(input_data)  # 16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)  # 16*16*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)  # 8*8*32

        x_A_cat1 = torch.cat((x_A_pool1, x_B_pool1), 1)  # 8*8*64
        x_B_cat1 = torch.cat((x_B_pool1, x_A_pool1), 1)  # 8*8*64

        x_A_point_conv1 = self.point_conv1(x_A_cat1)  # 8*8*32
        x_B_point_conv1 = self.point_conv1(x_B_cat1)  # 8*8*32

        x_A_conv2 = self.layer_A_conv2(x_A_point_conv1)  # 8*8*64
        x_B_conv2 = self.layer_B_conv2(x_B_point_conv1)  # 8*8*64

        x_A_cat2 = torch.cat((x_A_conv2, x_B_conv2), 1)  # 8*8*128
        x_B_cat2 = torch.cat((x_B_conv2, x_A_conv2), 1)  # 8*8*128

        x_A_point_conv2 = self.point_conv2(x_A_cat2)  # 8*8*64
        x_B_point_conv2 = self.point_conv2(x_B_cat2)  # 8*8*64

        x_A_pool2 = self.layer_A_pool2(x_A_point_conv2)  # 4*4*96
        x_B_pool2 = self.layer_B_pool2(x_B_point_conv2)  # 4*4*64

        x_global_cat = torch.cat((x_A_pool2, x_B_pool2), 1)  # 4*4*160(96+64)
        x_global_cat = self.point_conv3(x_global_cat)  # 4*4*128
        x_global_conv = self.global_conv(x_global_cat)  # 2*2*160

        return x_global_conv

    def forward(self, input_data):
        x = self.features(input_data)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  #


class DPARALLEL_CROSS_NET(nn.Module):  # DPCCN
    """
    docstring for DPARALLEL_CROSS_NET
    """

    def __init__(self, num_class=10, head_payload=False):
        super(DPARALLEL_CROSS_NET, self).__init__()
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_A_pool2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 4*4*256
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.layer_B_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),  # 16*16*32
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*32
        self.layer_B_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, padding=2, bias=True),  # 8*8*64
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )
        self.layer_B_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=self.p, )  # 4*4*128

        self.global_conv = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 2*2*160
            nn.BatchNorm2d(160, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.point_conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.point_conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)
        self.point_conv3 = nn.Conv2d(160, 128, kernel_size=3, stride=1, dilation=2, padding=2, bias=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*160

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1 * 1 * 160),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 160, self.num_class, bias=True)
        )

    def features(self, input_data):
        x_A_conv1 = self.layer_A_conv1(input_data)  # 16*16*16
        x_B_conv1 = self.layer_B_conv1(input_data)  # 16*16*32
        x_A_pool1 = self.layer_A_pool1(x_A_conv1)  # 16*16*32
        x_B_pool1 = self.layer_B_pool1(x_B_conv1)  # 8*8*32

        x_A_cat1 = torch.cat((x_A_pool1, x_B_pool1), 1)  # 8*8*64
        x_B_cat1 = torch.cat((x_B_pool1, x_A_pool1), 1)  # 8*8*64

        x_A_point_conv1 = self.point_conv1(x_A_cat1)  # 8*8*32
        x_B_point_conv1 = self.point_conv1(x_B_cat1)  # 8*8*32

        x_A_conv2 = self.layer_A_conv2(x_A_point_conv1)  # 8*8*64
        x_B_conv2 = self.layer_B_conv2(x_B_point_conv1)  # 8*8*64

        x_A_cat2 = torch.cat((x_A_conv2, x_B_conv2), 1)  # 8*8*128
        x_B_cat2 = torch.cat((x_B_conv2, x_A_conv2), 1)  # 8*8*128

        x_A_point_conv2 = self.point_conv2(x_A_cat2)  # 8*8*64
        x_B_point_conv2 = self.point_conv2(x_B_cat2)  # 8*8*64

        x_A_pool2 = self.layer_A_pool2(x_A_point_conv2)  # 4*4*96
        x_B_pool2 = self.layer_B_pool2(x_B_point_conv2)  # 4*4*64

        x_global_cat = torch.cat((x_A_pool2, x_B_pool2), 1)  # 4*4*160(96+64)
        x_global_cat = self.point_conv3(x_global_cat)  # 4*4*128
        x_global_conv = self.global_conv(x_global_cat)  # 2*2*160

        return x_global_conv

    def forward(self, input_data):
        x = self.features(input_data)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  # D  #


class PARALLEL_CROSS_CNN_ADD(nn.Module):  # APCCN
    """
    docstring for PARALLEL_CROSS_CNN_ADD
    """

    def __init__(self, num_class=10, head_payload=False):
        super(PARALLEL_CROSS_CNN_ADD, self).__init__()
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),  # 2*2*256
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=True),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*256
        self.point_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)

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

        x_add = x_A_pool2 + x_A_pool2  # 4*4*128
        x_gconv = self.global_conv(x_add)  # 2*2*256

        return x_gconv

    def forward(self, input_data):
        x = self.features(input_data)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 平行网络
class p_lstm_cnn(nn.Module):
    def __init__(self):
        super(p_lstm_cnn, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2)

        self.lstm1 = nn.LSTM(22, 44, 2)
        self.lstm2 = nn.LSTM(44, 44, 2)

        self.globalconv1 = nn.Conv2d(854, 1456, kernel_size=3, stride=1, padding=1)
        self.globalbn = nn.BatchNorm2d(1456)
        self.avpool = nn.AdaptiveAvgPool2d(2)
        self.globallstm = nn.LSTM(1456, 512, 1)

        self.fc1 = nn.Linear(512 * 4, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        ux = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        ux = self.maxpool(F.relu(self.bn2(self.conv2(ux))))
        #         print(ux.shape)
        ux = self.maxpool(F.relu(self.bn3(self.conv3(ux))))
        #         print("ux",ux.shape)

        x = x.view(16, -1, 22)
        dx, _ = self.lstm1(x)
        dx, _ = self.lstm2(dx)

        #         print("dx",dx.shape)
        dx = torch.reshape(dx, [16, -1, 2, 2])
        x = torch.cat([ux, dx], dim=1)
        #         print("x",x.shape)

        x = self.avpool(F.relu(self.globalbn(self.globalconv1(x))))
        #         print("x",x.shape)
        x = x.view(16, -1, 1456)
        x, _ = self.globallstm(x)
        #         print("x",x.shape)
        x = x.view(-1, 512 * 4)
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x



if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    import numpy as np

    # font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']
    labels = ['ant', 'bird', 'cat']
    maxtrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.matshow(maxtrix)
    plt.colorbar()
    plt.xlabel('predict type')
    plt.ylabel('actual type')
    plt.xticks(np.arange(maxtrix.shape[1]), labels)
    plt.yticks(np.arange(maxtrix.shape[1]), labels)
    plt.show()

    my_models = {
        'FCNN': my_new_nets.CNN(num_class=NUM_CLASSES, head_payload=header_payload),
        'CNN_NORMAL': my_new_nets.CNN_NORMAL(num_class=NUM_CLASSES, head_payload=header_payload),
        'LSTM': my_new_nets.LSTM(num_class=NUM_CLASSES, head_payload=header_payload),
        'CNN_LSTM': my_new_nets.CNN_LSTM(num_class=NUM_CLASSES, head_payload=header_payload),
        'PARALLELNETS': my_new_nets.PARALLELNETS(num_class=NUM_CLASSES, head_payload=header_payload),
        'CROSS_CNN': my_new_nets.CROSS_CNN(num_class=NUM_CLASSES, head_payload=header_payload),
        'PARALLEL_CROSS_CNN': my_new_nets.PARALLEL_CROSS_CNN(num_class=NUM_CLASSES, head_payload=header_payload),
        'DPARALLEL_CROSS_NET': my_new_nets.DPARALLEL_CROSS_NET(num_class=NUM_CLASSES, head_payload=header_payload),
        'PARALLEL_CROSS_CNN_ADD': my_new_nets.PARALLEL_CROSS_CNN_ADD(num_class=NUM_CLASSES, head_payload=header_payload)
    }

    import shutil

    shutil.rmtree(r"E:\seafile\downloadfile\project\USTC-TK2016-master\2_Session\L7\testbed-16jun-L7")



