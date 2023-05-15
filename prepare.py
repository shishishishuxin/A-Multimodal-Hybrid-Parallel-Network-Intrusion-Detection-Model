#-*- coding:utf-8 -*-
#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np 
import time
import pandas as pd 
import torch
import tqdm
import shutil
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# import image

def train(train_loader,model,metric,loss_function,optimizer,epoch):

	batch_time =AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()
	batch_size = 256
	embedding_dim = 2
	train_batchs = 668  # 每一轮输入多少次batch
	# 用于更新每次训练后的参数，后面进行覆盖,train_loader.sampler.num_samples是获得训练集的样本数量
	nlabels = np.zeros((train_loader.sampler.num_samples,), dtype=np.int32)
	# embeddings = np.zeros((train_loader.sampler.num_samples, embedding_dim), dtype=np.float32)
	# switch to train mode
	model.train()
	loss_acc = {'loss':[],'accuracy':[]}
	end = time.time()

	for step,(feature,label) in enumerate(train_loader):
		feature = Variable(feature).cuda()
		label = Variable(label).cuda()
		# input feature to train，得到预测标签y_pred
		y_pred = model(feature)
		# y_pred= metric(y_pred, label)   # arcmargin loss
		# 将输出值输入到损失函数进行处理
		# squeeze()默认将所有的1的维度删除，此时输入的label是[256,1]，squeeze后是[256]，相当于都存放在一个数据里面了
		loss = loss_function(y_pred,label.squeeze())
		losses.update(loss.item(),feature.size(0))
		# compute accuracy,batch_size = 256,y_pred 的shape(256,12) 分别判断这256个流量里面属于12个分类的准确率
		# 获得本次输入的准确率
		pred_acc,pred_count = accuracy(y_pred.data,label,topk=(1,1))
		acc.update(pred_acc,pred_count)

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()  # 梯度清0

		batch_time.update(time.time() - end,1)  #更新时间
		end = time.time()

		# 更新绘图参数
		i, j = step * batch_size, (step + 1) * batch_size
		if(j>train_loader.sampler.num_samples):
			j = train_loader.sampler.num_samples
		# embeddings[i:j] = plot.cpu().detach().numpy()   # 将tensor转换成numpy
		t_label = label.cpu().detach().numpy()
		nlabels[i:j] = t_label.reshape(t_label.size)
		if step % 10 == 0:
			print('epoch:[{0}][{1}/{2}]\t'
				  'Time:{batch_time.val:.3f} \t'
				  'Data:{data_time.val:.3f} \t'
				  'Loss:{loss.val:.4f} \t'
				  'Accuracy:{acc.val:.3f} '.format(
				  	epoch,step,len(train_loader),batch_time=batch_time,data_time=data_time,loss=losses,acc=acc
				  	)
				)
			loss_acc['loss'].append(losses.val)
			loss_acc['accuracy'].append(acc.val)

	return loss_acc

def validate(validate_loader,model,metric,loss_function,best_precision,lowest_loss):
	batch_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()

	#switch to evaluable mode,forbidden batchnormalization and dropout
	model.eval()

	end = time.time()
	for step,(feature,label) in enumerate(validate_loader):
		# feature,label = data
		
		feature = Variable(feature).cuda(non_blocking=True)
		label = Variable(label).cuda(non_blocking=True)

		with torch.no_grad():
			y_pred = model(feature)
			# y_pred = metric(y_pred, label)
			loss = loss_function(y_pred,label.squeeze())

		#measure accuracy and record loss
		pred_acc,PRED_COUNT = accuracy(y_pred.data,label,topk=(1,1))
		losses.update(loss.item(),feature.size(0))
		acc.update(pred_acc,PRED_COUNT)

		
		batch_time.update(time.time() - end,1)
		end = time.time()

		if step % 10 == 0:
			print('TrainVal: [{0}/{1}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
					step, len(validate_loader), batch_time=batch_time, loss=losses, acc=acc))

	print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
		' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
	return acc.avg,losses.avg

def test(test_loader,model,metric,num_class,topk=1):

	top1_prob = []
	top1_pred_label = []
	topk_prob = []
	topk_pred_label = []
	actual_label = []
	correct = 0


	model.eval()
	for step,(feature,label) in enumerate(test_loader):
		feature = Variable(feature).cuda(non_blocking=True)
		label = Variable(label).cuda(non_blocking=True)
		
		with torch.no_grad():
			y_pred = model(feature)
			#使用softmax预测结果
			smax = nn.Softmax(1)
			smax_out = smax(y_pred)
			# smax_out = metric(y_pred, label)  # 修改分类器

		probility,pred_label = torch.topk(smax_out,topk)
		p1,l1 = torch.topk(smax_out,1)

		actual_label += label.squeeze().tolist()
		top1_pred_label += l1.tolist()

	# 里面储存的是测试集所有数据的预测标签和预测值

	top1_pred_label = np.array(top1_pred_label)
	actual_label = np.array(actual_label).reshape(-1,1)


	result = (top1_pred_label,actual_label)
	
	return result



def accuracy(y_pred,y_label,topk=(1,)):

	final_acc = 0
	maxk = max(topk)
	# 预测准确的数量
	PRED_COUNT = y_label.size(0)
	PRED_CORRECT_COUNT = 0
	# 获得y_pred 对12个类别的top k值(此时k=1)，返回prob是按大小排序后的值，pred_label是索引
	prob,pred_label = y_pred.topk(maxk,dim=1,largest=True,sorted=True)
	for x in range(pred_label.size(0)):  # 遍历每一行（即每一个数据流）
		if int(pred_label[x]) == y_label[x]:  # 判断标签是否正确
			PRED_CORRECT_COUNT += 1
	
	if PRED_COUNT == 0:
		return final_acc

	final_acc = PRED_CORRECT_COUNT / PRED_COUNT
	return final_acc*100,PRED_COUNT


def adjust_learning_rate(model,metric,weight_decay,base_lr,lr_decay):
	base_lr = base_lr / lr_decay
	return optim.Adam(model.parameters(),base_lr,weight_decay=weight_decay,amsgrad=True)
	# return optim.Adam([{'params': model.parameters()}, {'params': metric.parameters()}],base_lr,weight_decay=weight_decay,amsgrad=True)

def save_checkpoint(state,is_best,is_lowest_loss,filename):
	s_filename = './model/%s/checkpoint.pth.tar' %filename
	torch.save(state,s_filename)
	if is_best:
		shutil.copyfile(s_filename,'./model/%s/model_best.pth.tar' %filename)
	if is_lowest_loss:
		shutil.copyfile(s_filename,'./model/%s/lowest_loss.pth.tar' %filename)


class MDealDataSet2017(Dataset):  #将数据从numpy转换成tensor
	"""docstring for DealDataSet"""
	def __init__(self,data_list,header_payload=False):
		self.x = torch.from_numpy(data_list[:,:-1])
		self.x = self.x.type(torch.FloatTensor)
		# 修改数据类型从LongTensor为FloatTensor
		if header_payload == True:
			self.x = self.x.view(self.x.shape[0],1,548)
		else:
			self.x = self.x.view(self.x.shape[0],1,548)
		self.y = torch.from_numpy(data_list[:,[-1]])
		self.y = self.y.type(torch.LongTensor)
		self.len = self.x.shape[0]
		self.xshape = self.x.shape
		self.yshape = self.y.shape

	def __getitem__(self,index):
		return self.x[index],self.y[index]

	def __len__(self):
		return self.len

class DealDataSet(Dataset):  #将数据从numpy转换成tensor
	"""docstring for DealDataSet"""
	def __init__(self,data_list,header_payload=False):
		self.x = torch.from_numpy(data_list[:,:-1])
		self.x = self.x.type(torch.FloatTensor)
		# 修改数据类型从LongTensor为FloatTensor
		if header_payload == True:
			self.x = self.x.view(self.x.shape[0],1,8,8)
		else:
			self.x = self.x.view(self.x.shape[0],1,8,8)
		self.y = torch.from_numpy(data_list[:,[-1]])
		self.y = self.y.type(torch.LongTensor)
		self.len = self.x.shape[0]
		self.xshape = self.x.shape
		self.yshape = self.y.shape


	def __getitem__(self,index):
		return self.x[index],self.y[index]

	def __len__(self):
		return self.len




class AverageMeter(object):
	"""计算和保存当前值与平均值"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self,val,n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count

class TrainDataSetHeader():
	def __init__(self,data_type=1):
		self.data_type = data_type
		super(TrainDataSetHeader, self).__init__()

	def read_csv(self):
		if self.data_type == 1:
			mydata_botnet =  pd.read_csv('./flow_labeled/labeld_Botnet.csv')#2075
			mydata_DDoS =  pd.read_csv('./flow_labeled/labeld_DDoS.csv')#261226
			mydata_glodeneye =  pd.read_csv('./flow_labeled/labeld_DoS-GlodenEye.csv')#20543
			mydata_hulk =  pd.read_csv('./flow_labeled/labeld_DoS-Hulk.csv')#474656
			mydata_slowhttp =  pd.read_csv('./flow_labeled/labeld_DoS-Slowhttptest.csv')#6786
			mydata_slowloris =  pd.read_csv('./flow_labeled/labeld_DoS-Slowloris.csv')#10537
			mydata_ftppatator =  pd.read_csv('./flow_labeled/labeld_FTP-Patator.csv')#19941
			mydata_heartbleed =  pd.read_csv('./flow_labeled/labeld_Heartbleed-Port.csv')#9859
			mydata_infiltration_2 =  pd.read_csv('./flow_labeled/labeld_Infiltration-2.csv')#5126
			mydata_infiltration_4 =  pd.read_csv('./flow_labeled/labeld_Infiltration-4.csv')#168
			mydata_portscan_1 =  pd.read_csv('./flow_labeled/labeld_PortScan_1.csv')#755
			mydata_portscan_2 =  pd.read_csv('./flow_labeled/labeld_PortScan_2.csv')#318881
			mydata_sshpatator =  pd.read_csv('./flow_labeled/labeld_SSH-Patator.csv')#27545
			mydata_bruteforce =  pd.read_csv('./flow_labeled/labeld_WebAttack-BruteForce.csv')#7716
			mydata_sqlinjection =  pd.read_csv('./flow_labeled/labeld_WebAttack-SqlInjection.csv')#25
			mydata_xss =  pd.read_csv('./flow_labeled/labeld_WebAttack-XSS.csv')#2796
		elif self.data_type == 2:
			mydata_botnet =  pd.read_csv('./payload_labeled/labeld_Botnet_payload.csv')
			mydata_DDoS =  pd.read_csv('./payload_labeled/labeld_DDoS_payload.csv')
			mydata_glodeneye =  pd.read_csv('./payload_labeled/labeld_DoS-GlodenEye_payload.csv')
			mydata_hulk =  pd.read_csv('./payload_labeled/labeld_DoS-Hulk_payload.csv')
			mydata_slowhttp =  pd.read_csv('./payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
			mydata_slowloris =  pd.read_csv('./payload_labeled/labeld_DoS-Slowloris_payload.csv')
			mydata_ftppatator =  pd.read_csv('./payload_labeled/labeld_FTP-Patator_payload.csv')
			mydata_heartbleed =  pd.read_csv('./payload_labeled/labeld_Heartbleed-Port_payload.csv')
			mydata_infiltration_2 =  pd.read_csv('./payload_labeled/labeld_Infiltration-2_payload.csv')
			mydata_infiltration_4 =  pd.read_csv('./payload_labeled/labeld_Infiltration-4_payload.csv')
			mydata_portscan_1 =  pd.read_csv('./payload_labeled/labeld_PortScan_1_payload.csv')
			mydata_portscan_2 =  pd.read_csv('./payload_labeled/labeld_PortScan_2_payload.csv')
			mydata_sshpatator =  pd.read_csv('./payload_labeled/labeld_SSH-Patator_payload.csv')
			mydata_bruteforce =  pd.read_csv('./payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
			mydata_sqlinjection =  pd.read_csv('./payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
			mydata_xss = pd.read_csv('./payload_labeled/labeld_WebAttack-XSS_payload.csv')
		elif self.data_type == 3:
			mydata_botnet =  pd.read_csv('./head_payload_labeled/labeld_Botnet_head_payload.csv')
			mydata_DDoS =  pd.read_csv('./head_payload_labeled/labeld_DDoS_head_payload.csv')
			mydata_glodeneye =  pd.read_csv('./head_payload_labeled/labeld_DoS-GlodenEye_head_payload.csv')
			mydata_hulk =  pd.read_csv('./head_payload_labeled/labeld_DoS-Hulk_head_payload.csv')
			mydata_slowhttp =  pd.read_csv('./head_payload_labeled/labeld_DoS-Slowhttptest_head_payload.csv')
			mydata_slowloris =  pd.read_csv('./head_payload_labeled/labeld_DoS-Slowloris_head_payload.csv')
			mydata_ftppatator =  pd.read_csv('./head_payload_labeled/labeld_FTP-Patator_head_payload.csv')
			mydata_heartbleed =  pd.read_csv('./head_payload_labeled/labeld_Heartbleed-Port_head_payload.csv')
			mydata_infiltration_2 =  pd.read_csv('./head_payload_labeled/labeld_Infiltration-2_head_payload.csv')
			mydata_infiltration_4 =  pd.read_csv('./head_payload_labeled/labeld_Infiltration-4_head_payload.csv')
			mydata_portscan_1 =  pd.read_csv('./head_payload_labeled/labeld_PortScan_1_head_payload.csv')
			mydata_portscan_2 =  pd.read_csv('./head_payload_labeled/labeld_PortScan_2_head_payload.csv')
			mydata_sshpatator =  pd.read_csv('./head_payload_labeled/labeld_SSH-Patator_head_payload.csv')
			mydata_bruteforce =  pd.read_csv('./head_payload_labeled/labeld_WebAttack-BruteForce_head_payload.csv')
			mydata_sqlinjection =  pd.read_csv('./head_payload_labeled/labeld_WebAttack-SqlInjection_head_payload.csv')
			mydata_xss = pd.read_csv('./head_payload_labeled/labeld_WebAttack-XSS_head_payload.csv')
			# mydata_benign = pd.read_csv('./head_payload_labeled/labeld_Monday-Benign.csv')

		# benign = mydata_benign.values[:,1:]
		botnet = mydata_botnet.values[:,1:]
		ddos = mydata_DDoS.values[:,1:]
		glodeneye = mydata_glodeneye.values[:,1:]
		hulk = mydata_hulk.values[:,1:]
		slowhttp = mydata_slowhttp.values[:,1:]
		slowloris = mydata_slowloris.values[:,1:]
		ftp_patator = mydata_ftppatator.values[:,1:]
		heartbleed = mydata_heartbleed.values[:,1:]
		infiltration_2 = mydata_infiltration_2.values[:,1:]
		infiltration_4 = mydata_infiltration_4.values[:,1:]
		portscan_1 = mydata_portscan_1.values[:,1:]
		portscan_2 = mydata_portscan_2.values[:,1:]
		ssh_patator = mydata_sshpatator.values[:,1:]
		bruteforce = mydata_bruteforce.values[:,1:]
		sqlinjection = mydata_sqlinjection.values[:,1:]
		xss = mydata_xss.values[:,1:]

		# return benign,botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss
		return botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss
		# return botnet,glodeneye

	def get_item(self):
		# botnet,glodeneye = self.read_csv()
		botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss = self.read_csv()

		# print('shape of benign: ',benign.shape)
		print('shape of botnet: ',botnet.shape)
		print('shape of DDoS: ',ddos.shape)
		print('shape of glodeneye: ',glodeneye.shape)
		print('shape of hulk: ',hulk.shape)
		print('shape of slowhttp: ',slowhttp.shape)
		print('shape of slowloris: ',slowloris.shape)
		print('shape of ftppatator: ',ftp_patator.shape)
		print('shape of heartbleed: ',heartbleed.shape)
		print('shape of infiltration_2: ',infiltration_2.shape)
		print('shape of infiltration_4: ',infiltration_4.shape)
		print('shape of portscan_1: ',portscan_1.shape)
		print('shape of portscan_2: ',portscan_2.shape)
		print('shape of sshpatator: ',ssh_patator.shape)
		print('shape of brutefoece: ',bruteforce.shape)
		print('shape of sqlinjection: ',sqlinjection.shape)
		print('shape of xss: ',xss.shape)

		# x_benign = benign[:,:-1]
		x_botnet = botnet[:,:-1]
		x_ddos = ddos[:,:-1]
		x_glodeneye = glodeneye[:,:-1]
		x_hulk = hulk[:,:-1]
		x_slowhttp = slowhttp[:,:-1]
		x_slowloris = slowloris[:,:-1]
		x_ftppatator = ftp_patator[:,:-1]
		x_heartbleed = heartbleed[:,:-1]
		x_infiltration_2 = infiltration_2[:,:-1]
		x_infiltration_4 = infiltration_4[:,:-1]
		x_portscan_1 = portscan_1[:,:-1]
		x_portscan_2 = portscan_2[:,:-1]
		x_sshpatator = ssh_patator[:,:-1]
		x_bruteforce = bruteforce[:,:-1]
		x_sqlinjection = sqlinjection[:,:-1]
		x_xss = xss[:,:-1]

		# y_benign = benign[:,-1]
		y_botnet = botnet[:,-1]
		y_ddos = ddos[:,-1]
		y_glodeneye = glodeneye[:,-1]
		y_hulk = hulk[:,-1]
		y_slowhttp = slowhttp[:,-1]
		y_slowloris = slowloris[:,-1]
		y_ftppatator = ftp_patator[:,-1]
		y_heartbleed = heartbleed[:,-1]
		y_infiltration_2 = infiltration_2[:,-1]
		y_infiltration_4 = infiltration_4[:,-1]
		y_portscan_1 = portscan_1[:,-1]
		y_portscan_2 = portscan_2[:,-1]
		y_sshpatator = ssh_patator[:,-1]
		y_bruteforce = bruteforce[:,-1]
		y_sqlinjection = sqlinjection[:,-1]
		y_xss = xss[:,-1]

		# x_tr_benign,x_te_benign,y_tr_benign,y_te_benign = train_test_split(x_benign,y_benign,test_size=0.2,random_state=1)
		x_tr_botnet,x_te_botnet,y_tr_botnet,y_te_botnet = train_test_split(x_botnet,y_botnet,test_size=0.2,random_state=1)
		x_tr_ddos,x_te_ddos,y_tr_ddos,y_te_ddos = train_test_split(x_ddos[:len(x_ddos)//8],y_ddos[:len(x_ddos)//8],test_size=0.2,random_state=1)
		x_tr_glodeneye,x_te_glodeneye,y_tr_glodeneye,y_te_glodeneye = train_test_split(x_glodeneye,y_glodeneye,test_size=0.2,random_state=1)
		x_tr_hulk,x_te_hulk,y_tr_hulk,y_te_hulk = train_test_split(x_hulk[:len(x_ddos)//8],y_hulk[:len(x_ddos)//8],test_size=0.2,random_state=1)
		x_tr_slowhttp,x_te_slowhttp,y_tr_slowhttp,y_te_slowhttp = train_test_split(x_slowhttp,y_slowhttp,test_size=0.2,random_state=1)
		x_tr_slowloris,x_te_slowloris,y_tr_slowloris,y_te_slowloris = train_test_split(x_slowloris,y_slowloris,test_size=0.2,random_state=1)
		x_tr_ftppatator,x_te_ftppatator,y_tr_ftppatator,y_te_ftppatator = train_test_split(x_ftppatator,y_ftppatator,test_size=0.2,random_state=1)
		x_tr_heartbleed,x_te_heartbleed,y_tr_heartbleed,y_te_heartbleed = train_test_split(x_heartbleed,y_heartbleed,test_size=0.2,random_state=1)
		x_tr_infiltration_2,x_te_infiltration_2,y_tr_infiltration_2,y_te_infiltration_2 = train_test_split(x_infiltration_2,y_infiltration_2,test_size=0.2,random_state=1)
		x_tr_infiltration_4,x_te_infiltration_4,y_tr_infiltration_4,y_te_infiltration_4 = train_test_split(x_infiltration_4,y_infiltration_4,test_size=0.2,random_state=1)
		x_tr_portscan_1,x_te_portscan_1,y_tr_portscan_1,y_te_portscan_1 = train_test_split(x_portscan_1[:len(x_ddos)//8],y_portscan_1[:len(x_ddos)//8],test_size=0.2,random_state=1)
		x_tr_portscan_2,x_te_portscan_2,y_tr_portscan_2,y_te_portscan_2 = train_test_split(x_portscan_2[:len(x_ddos)//8],y_portscan_2[:len(x_ddos)//8],test_size=0.2,random_state=1)
		x_tr_sshpatator,x_te_sshpatator,y_tr_sshpatator,y_te_sshpatator = train_test_split(x_sshpatator,y_sshpatator,test_size=0.2,random_state=1)
		x_tr_bruteforce,x_te_bruteforce,y_tr_bruteforce,y_te_bruteforce = train_test_split(x_bruteforce,y_bruteforce,test_size=0.2,random_state=1)
		x_tr_sqlinjection,x_te_sqlinjection,y_tr_sqlinjection,y_te_sqlinjection = train_test_split(x_sqlinjection,y_sqlinjection,test_size=0.2,random_state=1)
		x_tr_xss,x_te_xss,y_tr_xss,y_te_xss = train_test_split(x_xss,y_xss,test_size=0.2,random_state=1)


		x_tr_infiltration = np.concatenate((x_tr_infiltration_2,x_tr_infiltration_4),axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1,x_tr_portscan_2),axis=0)
		x_tr_webattack = np.concatenate((x_tr_bruteforce,x_tr_sqlinjection,x_tr_xss),axis=0)

		y_tr_infiltration = np.concatenate((y_tr_infiltration_2,y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1,y_tr_portscan_2))
		y_tr_webattack = np.concatenate((y_tr_bruteforce,y_tr_sqlinjection,y_tr_xss))

		x_te_infiltration = np.concatenate((x_te_infiltration_2,x_te_infiltration_4),axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1,x_te_portscan_2),axis=0)
		x_te_webattack = np.concatenate((x_te_bruteforce,x_te_sqlinjection,x_te_xss),axis=0)

		y_te_infiltration = np.concatenate((y_te_infiltration_2,y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1,y_te_portscan_2))
		y_te_webattack = np.concatenate((y_te_bruteforce,y_te_sqlinjection,y_te_xss))

		#play label,因为之前合并了某类攻击，所以需要重新定义标签

		y_tr_botnet = np.array([0]*len(y_tr_botnet))
		y_tr_ddos = np.array([1]*len(y_tr_ddos))
		y_tr_glodeneye = np.array([2]*len(y_tr_glodeneye))
		y_tr_hulk = np.array([3]*len(y_tr_hulk))
		y_tr_slowhttp = np.array([4]*len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5]*len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6]*len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7]*len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8]*len(y_tr_infiltration))
		y_tr_portscan = np.array([9]*len(y_tr_portscan))
		y_tr_sshpatator = np.array([10]*len(y_tr_sshpatator))
		y_tr_webattack = np.array([11]*len(y_tr_webattack))
		# y_tr_benign = np.array([12] * len(y_tr_benign))

		y_te_botnet = np.array([0]*len(y_te_botnet))
		y_te_ddos = np.array([1]*len(y_te_ddos))
		y_te_glodeneye = np.array([2]*len(y_te_glodeneye))
		y_te_hulk = np.array([3]*len(y_te_hulk))
		y_te_slowhttp = np.array([4]*len(y_te_slowhttp))
		y_te_slowloris = np.array([5]*len(y_te_slowloris))
		y_te_ftppatator = np.array([6]*len(y_te_ftppatator))
		y_te_heartbleed = np.array([7]*len(y_te_heartbleed))
		y_te_infiltration = np.array([8]*len(y_te_infiltration))
		y_te_portscan = np.array([9]*len(y_te_portscan))
		y_te_sshpatator = np.array([10]*len(y_te_sshpatator))
		y_te_webattack = np.array([11]*len(y_te_webattack))
		# y_te_benign = np.array([12] * len(y_te_benign))

		x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack))
		# x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack,x_tr_benign))
		# y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack,y_tr_benign))
		# x_train = np.concatenate((x_tr_botnet,x_tr_glodeneye))
		# y_train = np.concatenate((y_tr_botnet,y_tr_glodeneye))

		x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye, x_te_hulk, x_te_slowhttp, x_te_slowloris,x_te_ftppatator, x_te_heartbleed, x_te_infiltration, x_te_portscan, x_te_sshpatator,x_te_webattack))
		y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye, y_te_hulk, y_te_slowhttp, y_te_slowloris,y_te_ftppatator, y_te_heartbleed, y_te_infiltration, y_te_portscan, y_te_sshpatator,y_te_webattack))
		# x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye, x_te_hulk, x_te_slowhttp, x_te_slowloris,x_te_ftppatator, x_te_heartbleed, x_te_infiltration, x_te_portscan, x_te_sshpatator,x_te_webattack,x_te_benign))
		# y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye, y_te_hulk, y_te_slowhttp, y_te_slowloris,y_te_ftppatator, y_te_heartbleed, y_te_infiltration, y_te_portscan, y_te_sshpatator,y_te_webattack,y_te_benign))
		# x_test = np.concatenate((x_te_botnet,x_te_glodeneye))
		# y_test = np.concatenate((y_te_botnet,y_te_glodeneye))

		return x_train,y_train,x_test,y_test

#2017row
class TrainDataSet2017():
	def __init__(self, data_type=1):
		self.data_type = data_type
		super(TrainDataSet2017, self).__init__()

	def read_csv(self):
		if self.data_type == 1:
			mydata_botnet = pd.read_csv(
				'./flow_labeled/labeld_Botnet.csv')  # 2075
			mydata_DDoS = pd.read_csv(
				'./flow_labeled/labeld_DDoS.csv')  # 261226
			mydata_glodeneye = pd.read_csv(
				'./flow_labeled/labeld_DoS-GlodenEye.csv')  # 20543
			mydata_hulk = pd.read_csv(
				'./flow_labeled/labeld_DoS-Hulk.csv')  # 474656
			mydata_slowhttp = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowhttptest.csv')  # 6786
			mydata_slowloris = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowloris.csv')  # 10537
			mydata_ftppatator = pd.read_csv(
				'./flow_labeled/labeld_FTP-Patator.csv')  # 19941
			mydata_heartbleed = pd.read_csv(
				'./flow_labeled/labeld_Heartbleed-Port.csv')  # 9859
			mydata_infiltration_2 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-2.csv')  # 5126
			mydata_infiltration_4 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-4.csv')  # 168
			mydata_portscan_1 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_1.csv')  # 755
			mydata_portscan_2 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_2.csv')  # 318881
			mydata_sshpatator = pd.read_csv(
				'./flow_labeled/labeld_SSH-Patator.csv')  # 27545
			mydata_bruteforce = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-BruteForce.csv')  # 7716
			mydata_sqlinjection = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-SqlInjection.csv')  # 25
			mydata_xss = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-XSS.csv')  # 2796
		elif self.data_type == 2:
			mydata_botnet = pd.read_csv(
				'./payload_labeled/labeld_Botnet_payload.csv')
			mydata_DDoS = pd.read_csv(
				'./payload_labeled/labeld_DDoS_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./payload_labeled/labeld_DoS-GlodenEye_payload.csv')
			mydata_hulk = pd.read_csv(
				'./payload_labeled/labeld_DoS-Hulk_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowloris_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./payload_labeled/labeld_FTP-Patator_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./payload_labeled/labeld_Heartbleed-Port_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-2_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-4_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_1_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_2_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./payload_labeled/labeld_SSH-Patator_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
			mydata_xss = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-XSS_payload.csv')
		elif self.data_type == 3:
			mydata_botnet = pd.read_csv(
				'./2017_labeled/labeld_Botnet_head_payload.csv')
			mydata_DDoS = pd.read_csv(
				'2017_labeled/labeld_DDoS_head_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./2017_labeled/labeld_DoS-GlodenEye_head_payload.csv')
			mydata_hulk = pd.read_csv(
				'./2017_labeled/labeld_DoS-Hulk_head_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowhttptest_head_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowloris_head_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./2017_labeled/labeld_FTP-Patator_head_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./2017_labeled/labeld_Heartbleed-Port_head_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-2_head_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-4_head_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_1_head_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_2_head_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./2017_labeled/labeld_SSH-Patator_head_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-BruteForce_head_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-SqlInjection_head_payload.csv')
			mydata_xss = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-XSS_head_payload.csv')
		# mydata_benign = pd.read_csv('./head_payload_labeled/labeld_Monday-Benign.csv')

		# benign = mydata_benign.values[:,1:]
		botnet = mydata_botnet.values[:, :484]
		ddos = mydata_DDoS.values[:, :484]
		glodeneye = mydata_glodeneye.values[:, :484]
		hulk = mydata_hulk.values[:, :484]
		slowhttp = mydata_slowhttp.values[:, :484]
		slowloris = mydata_slowloris.values[:, :484]
		ftp_patator = mydata_ftppatator.values[:, :484]
		heartbleed = mydata_heartbleed.values[:, :484]
		infiltration_2 = mydata_infiltration_2.values[:, :484]
		infiltration_4 = mydata_infiltration_4.values[:, :484]
		portscan_1 = mydata_portscan_1.values[:, :484]
		portscan_2 = mydata_portscan_2.values[:, :484]
		ssh_patator = mydata_sshpatator.values[:, :484]
		bruteforce = mydata_bruteforce.values[:, :484]
		sqlinjection = mydata_sqlinjection.values[:, :484]
		xss = mydata_xss.values[:, :484]

		# return benign,botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss
		return botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss

	# return botnet,glodeneye

	def get_item(self):
		# botnet,glodeneye = self.read_csv()
		botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss = self.read_csv()

		# print('shape of benign: ',benign.shape)
		print('shape of botnet: ', botnet.shape)
		print('shape of DDoS: ', ddos.shape)
		print('shape of glodeneye: ', glodeneye.shape)
		print('shape of hulk: ', hulk.shape)
		print('shape of slowhttp: ', slowhttp.shape)
		print('shape of slowloris: ', slowloris.shape)
		print('shape of ftppatator: ', ftp_patator.shape)
		print('shape of heartbleed: ', heartbleed.shape)
		print('shape of infiltration_2: ', infiltration_2.shape)
		print('shape of infiltration_4: ', infiltration_4.shape)
		print('shape of portscan_1: ', portscan_1.shape)
		print('shape of portscan_2: ', portscan_2.shape)
		print('shape of sshpatator: ', ssh_patator.shape)
		print('shape of brutefoece: ', bruteforce.shape)
		print('shape of sqlinjection: ', sqlinjection.shape)
		print('shape of xss: ', xss.shape)

		# x_benign = benign[:,:-1]
		x_botnet = botnet[:, :]
		x_ddos = ddos[:, :]
		x_glodeneye = glodeneye[:, :]
		x_hulk = hulk[:, :]
		x_slowhttp = slowhttp[:, :]
		x_slowloris = slowloris[:, :]
		x_ftppatator = ftp_patator[:, :]
		x_heartbleed = heartbleed[:, :]
		x_infiltration_2 = infiltration_2[:, :]
		x_infiltration_4 = infiltration_4[:, :]
		x_portscan_1 = portscan_1[:, :]
		x_portscan_2 = portscan_2[:, :]
		x_sshpatator = ssh_patator[:, :]
		x_bruteforce = bruteforce[:, :]
		x_sqlinjection = sqlinjection[:, :]
		x_xss = xss[:, :]

		# y_benign = benign[:,-1]
		y_botnet = botnet[:, -1]
		y_ddos = ddos[:, -1]
		y_glodeneye = glodeneye[:, -1]
		y_hulk = hulk[:, -1]
		y_slowhttp = slowhttp[:, -1]
		y_slowloris = slowloris[:, -1]
		y_ftppatator = ftp_patator[:, -1]
		y_heartbleed = heartbleed[:, -1]
		y_infiltration_2 = infiltration_2[:, -1]
		y_infiltration_4 = infiltration_4[:, -1]
		y_portscan_1 = portscan_1[:, -1]
		y_portscan_2 = portscan_2[:, -1]
		y_sshpatator = ssh_patator[:, -1]
		y_bruteforce = bruteforce[:, -1]
		y_sqlinjection = sqlinjection[:, -1]
		y_xss = xss[:, -1]

		# x_tr_benign,x_te_benign,y_tr_benign,y_te_benign = train_test_split(x_benign,y_benign,test_size=0.2,random_state=1)
		x_tr_botnet, x_te_botnet, y_tr_botnet, y_te_botnet = train_test_split(
			x_botnet, y_botnet, test_size=0.2, random_state=1)
		x_tr_ddos, x_te_ddos, y_tr_ddos, y_te_ddos = train_test_split(
			x_ddos, y_ddos,
			test_size=0.2, random_state=1)
		x_tr_glodeneye, x_te_glodeneye, y_tr_glodeneye, y_te_glodeneye = train_test_split(
			x_glodeneye, y_glodeneye, test_size=0.2, random_state=1)
		x_tr_hulk, x_te_hulk, y_tr_hulk, y_te_hulk = train_test_split(
			x_hulk, y_hulk,
			test_size=0.2, random_state=1)
		x_tr_slowhttp, x_te_slowhttp, y_tr_slowhttp, y_te_slowhttp = train_test_split(
			x_slowhttp, y_slowhttp, test_size=0.2, random_state=1)
		x_tr_slowloris, x_te_slowloris, y_tr_slowloris, y_te_slowloris = train_test_split(
			x_slowloris, y_slowloris, test_size=0.2, random_state=1)
		x_tr_ftppatator, x_te_ftppatator, y_tr_ftppatator, y_te_ftppatator = train_test_split(
			x_ftppatator, y_ftppatator, test_size=0.2, random_state=1)
		x_tr_heartbleed, x_te_heartbleed, y_tr_heartbleed, y_te_heartbleed = train_test_split(
			x_heartbleed, y_heartbleed, test_size=0.2, random_state=1)
		x_tr_infiltration_2, x_te_infiltration_2, y_tr_infiltration_2, y_te_infiltration_2 = train_test_split(
			x_infiltration_2, y_infiltration_2, test_size=0.2, random_state=1)
		x_tr_infiltration_4, x_te_infiltration_4, y_tr_infiltration_4, y_te_infiltration_4 = train_test_split(
			x_infiltration_4, y_infiltration_4, test_size=0.2, random_state=1)
		x_tr_portscan_1, x_te_portscan_1, y_tr_portscan_1, y_te_portscan_1 = train_test_split(
			x_portscan_1, y_portscan_1,
			test_size=0.2, random_state=1)
		x_tr_portscan_2, x_te_portscan_2, y_tr_portscan_2, y_te_portscan_2 = train_test_split(
			x_portscan_2, y_portscan_2,
			test_size=0.2, random_state=1)
		x_tr_sshpatator, x_te_sshpatator, y_tr_sshpatator, y_te_sshpatator = train_test_split(
			x_sshpatator, y_sshpatator, test_size=0.2, random_state=1)
		x_tr_bruteforce, x_te_bruteforce, y_tr_bruteforce, y_te_bruteforce = train_test_split(
			x_bruteforce, y_bruteforce, test_size=0.2, random_state=1)
		x_tr_sqlinjection, x_te_sqlinjection, y_tr_sqlinjection, y_te_sqlinjection = train_test_split(
			x_sqlinjection, y_sqlinjection, test_size=0.2, random_state=1)
		x_tr_xss, x_te_xss, y_tr_xss, y_te_xss = train_test_split(x_xss, y_xss,
																  test_size=0.2,
																  random_state=1)

		x_tr_infiltration = np.concatenate(
			(x_tr_infiltration_2, x_tr_infiltration_4), axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1, x_tr_portscan_2),
									   axis=0)
		x_tr_webattack = np.concatenate(
			(x_tr_bruteforce, x_tr_sqlinjection, x_tr_xss), axis=0)

		y_tr_infiltration = np.concatenate(
			(y_tr_infiltration_2, y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1, y_tr_portscan_2))
		y_tr_webattack = np.concatenate(
			(y_tr_bruteforce, y_tr_sqlinjection, y_tr_xss))

		x_te_infiltration = np.concatenate(
			(x_te_infiltration_2, x_te_infiltration_4), axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1, x_te_portscan_2),
									   axis=0)
		x_te_webattack = np.concatenate(
			(x_te_bruteforce, x_te_sqlinjection, x_te_xss), axis=0)

		y_te_infiltration = np.concatenate(
			(y_te_infiltration_2, y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1, y_te_portscan_2))
		y_te_webattack = np.concatenate(
			(y_te_bruteforce, y_te_sqlinjection, y_te_xss))

		# play label,因为之前合并了某类攻击，所以需要重新定义标签

		y_tr_botnet = np.array([0] * len(y_tr_botnet))
		y_tr_ddos = np.array([1] * len(y_tr_ddos))
		y_tr_glodeneye = np.array([2] * len(y_tr_glodeneye))
		y_tr_hulk = np.array([3] * len(y_tr_hulk))
		y_tr_slowhttp = np.array([4] * len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5] * len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6] * len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7] * len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8] * len(y_tr_infiltration))
		y_tr_portscan = np.array([9] * len(y_tr_portscan))
		y_tr_sshpatator = np.array([10] * len(y_tr_sshpatator))
		y_tr_webattack = np.array([11] * len(y_tr_webattack))
		# y_tr_benign = np.array([12] * len(y_tr_benign))

		y_te_botnet = np.array([0] * len(y_te_botnet))
		y_te_ddos = np.array([1] * len(y_te_ddos))
		y_te_glodeneye = np.array([2] * len(y_te_glodeneye))
		y_te_hulk = np.array([3] * len(y_te_hulk))
		y_te_slowhttp = np.array([4] * len(y_te_slowhttp))
		y_te_slowloris = np.array([5] * len(y_te_slowloris))
		y_te_ftppatator = np.array([6] * len(y_te_ftppatator))
		y_te_heartbleed = np.array([7] * len(y_te_heartbleed))
		y_te_infiltration = np.array([8] * len(y_te_infiltration))
		y_te_portscan = np.array([9] * len(y_te_portscan))
		y_te_sshpatator = np.array([10] * len(y_te_sshpatator))
		y_te_webattack = np.array([11] * len(y_te_webattack))
		# y_te_benign = np.array([12] * len(y_te_benign))

		x_train = np.concatenate((x_tr_botnet, x_tr_ddos, x_tr_glodeneye,
								  x_tr_hulk, x_tr_slowhttp, x_tr_slowloris,
								  x_tr_ftppatator, x_tr_heartbleed,
								  x_tr_infiltration, x_tr_portscan,
								  x_tr_sshpatator, x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet, y_tr_ddos, y_tr_glodeneye,
								  y_tr_hulk, y_tr_slowhttp, y_tr_slowloris,
								  y_tr_ftppatator, y_tr_heartbleed,
								  y_tr_infiltration, y_tr_portscan,
								  y_tr_sshpatator, y_tr_webattack))
		# x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack,x_tr_benign))
		# y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack,y_tr_benign))
		# x_train = np.concatenate((x_tr_botnet,x_tr_glodeneye))
		# y_train = np.concatenate((y_tr_botnet,y_tr_glodeneye))

		x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye,
								 x_te_hulk, x_te_slowhttp, x_te_slowloris,
								 x_te_ftppatator, x_te_heartbleed,
								 x_te_infiltration, x_te_portscan,
								 x_te_sshpatator, x_te_webattack))
		y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye,
								 y_te_hulk, y_te_slowhttp, y_te_slowloris,
								 y_te_ftppatator, y_te_heartbleed,
								 y_te_infiltration, y_te_portscan,
								 y_te_sshpatator, y_te_webattack))
		# x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye, x_te_hulk, x_te_slowhttp, x_te_slowloris,x_te_ftppatator, x_te_heartbleed, x_te_infiltration, x_te_portscan, x_te_sshpatator,x_te_webattack,x_te_benign))
		# y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye, y_te_hulk, y_te_slowhttp, y_te_slowloris,y_te_ftppatator, y_te_heartbleed, y_te_infiltration, y_te_portscan, y_te_sshpatator,y_te_webattack,y_te_benign))
		# x_test = np.concatenate((x_te_botnet,x_te_glodeneye))
		# y_test = np.concatenate((y_te_botnet,y_te_glodeneye))

		return x_train, y_train, x_test, y_test



#2017多模态
class TrainDataSetMul2017():
	def __init__(self, data_type=1):
		self.data_type = data_type
		super(TrainDataSetMul2017, self).__init__()

	def read_csv(self):
		if self.data_type == 1:
			mydata_botnet = pd.read_csv(
				'./flow_labeled/labeld_Botnet.csv')  # 2075
			mydata_DDoS = pd.read_csv(
				'./flow_labeled/labeld_DDoS.csv')  # 261226
			mydata_glodeneye = pd.read_csv(
				'./flow_labeled/labeld_DoS-GlodenEye.csv')  # 20543
			mydata_hulk = pd.read_csv(
				'./flow_labeled/labeld_DoS-Hulk.csv')  # 474656
			mydata_slowhttp = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowhttptest.csv')  # 6786
			mydata_slowloris = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowloris.csv')  # 10537
			mydata_ftppatator = pd.read_csv(
				'./flow_labeled/labeld_FTP-Patator.csv')  # 19941
			mydata_heartbleed = pd.read_csv(
				'./flow_labeled/labeld_Heartbleed-Port.csv')  # 9859
			mydata_infiltration_2 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-2.csv')  # 5126
			mydata_infiltration_4 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-4.csv')  # 168
			mydata_portscan_1 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_1.csv')  # 755
			mydata_portscan_2 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_2.csv')  # 318881
			mydata_sshpatator = pd.read_csv(
				'./flow_labeled/labeld_SSH-Patator.csv')  # 27545
			mydata_bruteforce = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-BruteForce.csv')  # 7716
			mydata_sqlinjection = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-SqlInjection.csv')  # 25
			mydata_xss = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-XSS.csv')  # 2796
		elif self.data_type == 2:
			mydata_botnet = pd.read_csv(
				'./payload_labeled/labeld_Botnet_payload.csv')
			mydata_DDoS = pd.read_csv(
				'./payload_labeled/labeld_DDoS_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./payload_labeled/labeld_DoS-GlodenEye_payload.csv')
			mydata_hulk = pd.read_csv(
				'./payload_labeled/labeld_DoS-Hulk_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowloris_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./payload_labeled/labeld_FTP-Patator_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./payload_labeled/labeld_Heartbleed-Port_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-2_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-4_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_1_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_2_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./payload_labeled/labeld_SSH-Patator_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
			mydata_xss = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-XSS_payload.csv')
		elif self.data_type == 3:
			mydata_botnet = pd.read_csv(
				'./2017_labeled/labeld_Botnet_head_payload.csv')
			mydata_DDoS = pd.read_csv(
				'2017_labeled/labeld_DDoS_head_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./2017_labeled/labeld_DoS-GlodenEye_head_payload.csv')
			mydata_hulk = pd.read_csv(
				'./2017_labeled/labeld_DoS-Hulk_head_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowhttptest_head_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowloris_head_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./2017_labeled/labeld_FTP-Patator_head_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./2017_labeled/labeld_Heartbleed-Port_head_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-2_head_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-4_head_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_1_head_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_2_head_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./2017_labeled/labeld_SSH-Patator_head_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-BruteForce_head_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-SqlInjection_head_payload.csv')
			mydata_xss = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-XSS_head_payload.csv')
		# mydata_benign = pd.read_csv('./head_payload_labeled/labeld_Monday-Benign.csv')

		# benign = mydata_benign.values[:,1:]
		botnet = mydata_botnet.values[:, :]
		ddos = mydata_DDoS.values[:, :]
		glodeneye = mydata_glodeneye.values[:, :]
		hulk = mydata_hulk.values[:, :]
		slowhttp = mydata_slowhttp.values[:, :]
		slowloris = mydata_slowloris.values[:, :]
		ftp_patator = mydata_ftppatator.values[:, :]
		heartbleed = mydata_heartbleed.values[:, :]
		infiltration_2 = mydata_infiltration_2.values[:, :]
		infiltration_4 = mydata_infiltration_4.values[:, :]
		portscan_1 = mydata_portscan_1.values[:, :]
		portscan_2 = mydata_portscan_2.values[:, :]
		ssh_patator = mydata_sshpatator.values[:, :]
		bruteforce = mydata_bruteforce.values[:, :]
		sqlinjection = mydata_sqlinjection.values[:, :]
		xss = mydata_xss.values[:, :]

		# return benign,botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss
		return botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss

	# return botnet,glodeneye

	def get_item(self):
		# botnet,glodeneye = self.read_csv()
		botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss = self.read_csv()

		# print('shape of benign: ',benign.shape)
		print('shape of botnet: ', botnet.shape)
		print('shape of DDoS: ', ddos.shape)
		print('shape of glodeneye: ', glodeneye.shape)
		print('shape of hulk: ', hulk.shape)
		print('shape of slowhttp: ', slowhttp.shape)
		print('shape of slowloris: ', slowloris.shape)
		print('shape of ftppatator: ', ftp_patator.shape)
		print('shape of heartbleed: ', heartbleed.shape)
		print('shape of infiltration_2: ', infiltration_2.shape)
		print('shape of infiltration_4: ', infiltration_4.shape)
		print('shape of portscan_1: ', portscan_1.shape)
		print('shape of portscan_2: ', portscan_2.shape)
		print('shape of sshpatator: ', ssh_patator.shape)
		print('shape of brutefoece: ', bruteforce.shape)
		print('shape of sqlinjection: ', sqlinjection.shape)
		print('shape of xss: ', xss.shape)

		# x_benign = benign[:,:-1]
		x_botnet = botnet[:, :-1]
		x_ddos = ddos[:, :-1]
		x_glodeneye = glodeneye[:, :-1]
		x_hulk = hulk[:, :-1]
		x_slowhttp = slowhttp[:, :-1]
		x_slowloris = slowloris[:, :-1]
		x_ftppatator = ftp_patator[:, :-1]
		x_heartbleed = heartbleed[:, :-1]
		x_infiltration_2 = infiltration_2[:, :-1]
		x_infiltration_4 = infiltration_4[:, :-1]
		x_portscan_1 = portscan_1[:, :-1]
		x_portscan_2 = portscan_2[:, :-1]
		x_sshpatator = ssh_patator[:, :-1]
		x_bruteforce = bruteforce[:, :-1]
		x_sqlinjection = sqlinjection[:, :-1]
		x_xss = xss[:, :-1]

		# y_benign = benign[:,-1]
		y_botnet = botnet[:, -1]
		y_ddos = ddos[:, -1]
		y_glodeneye = glodeneye[:, -1]
		y_hulk = hulk[:, -1]
		y_slowhttp = slowhttp[:, -1]
		y_slowloris = slowloris[:, -1]
		y_ftppatator = ftp_patator[:, -1]
		y_heartbleed = heartbleed[:, -1]
		y_infiltration_2 = infiltration_2[:, -1]
		y_infiltration_4 = infiltration_4[:, -1]
		y_portscan_1 = portscan_1[:, -1]
		y_portscan_2 = portscan_2[:, -1]
		y_sshpatator = ssh_patator[:, -1]
		y_bruteforce = bruteforce[:, -1]
		y_sqlinjection = sqlinjection[:, -1]
		y_xss = xss[:, -1]

		# x_tr_benign,x_te_benign,y_tr_benign,y_te_benign = train_test_split(x_benign,y_benign,test_size=0.2,random_state=1)
		x_tr_botnet, x_te_botnet, y_tr_botnet, y_te_botnet = train_test_split(
			x_botnet, y_botnet, test_size=0.2, random_state=1)
		x_tr_ddos, x_te_ddos, y_tr_ddos, y_te_ddos = train_test_split(
			x_ddos[:len(x_ddos)//8], y_ddos[:len(x_ddos)//8],
			test_size=0.2, random_state=1)
		x_tr_glodeneye, x_te_glodeneye, y_tr_glodeneye, y_te_glodeneye = train_test_split(
			x_glodeneye, y_glodeneye, test_size=0.2, random_state=1)
		x_tr_hulk, x_te_hulk, y_tr_hulk, y_te_hulk = train_test_split(
			x_hulk[:len(x_ddos)//8], y_hulk[:len(x_ddos)//8],
			test_size=0.2, random_state=1)
		x_tr_slowhttp, x_te_slowhttp, y_tr_slowhttp, y_te_slowhttp = train_test_split(
			x_slowhttp, y_slowhttp, test_size=0.2, random_state=1)
		x_tr_slowloris, x_te_slowloris, y_tr_slowloris, y_te_slowloris = train_test_split(
			x_slowloris, y_slowloris, test_size=0.2, random_state=1)
		x_tr_ftppatator, x_te_ftppatator, y_tr_ftppatator, y_te_ftppatator = train_test_split(
			x_ftppatator, y_ftppatator, test_size=0.2, random_state=1)
		x_tr_heartbleed, x_te_heartbleed, y_tr_heartbleed, y_te_heartbleed = train_test_split(
			x_heartbleed, y_heartbleed, test_size=0.2, random_state=1)
		x_tr_infiltration_2, x_te_infiltration_2, y_tr_infiltration_2, y_te_infiltration_2 = train_test_split(
			x_infiltration_2, y_infiltration_2, test_size=0.2, random_state=1)
		x_tr_infiltration_4, x_te_infiltration_4, y_tr_infiltration_4, y_te_infiltration_4 = train_test_split(
			x_infiltration_4, y_infiltration_4, test_size=0.2, random_state=1)
		x_tr_portscan_1, x_te_portscan_1, y_tr_portscan_1, y_te_portscan_1 = train_test_split(
			x_portscan_1[:len(x_portscan_1)//8], y_portscan_1[:len(x_portscan_1)//8],
			test_size=0.2, random_state=1)
		x_tr_portscan_2, x_te_portscan_2, y_tr_portscan_2, y_te_portscan_2 = train_test_split(
			x_portscan_2[:len(x_portscan_2)//8], y_portscan_2[:len(x_portscan_2)//8],
			test_size=0.2, random_state=1)
		x_tr_sshpatator, x_te_sshpatator, y_tr_sshpatator, y_te_sshpatator = train_test_split(
			x_sshpatator, y_sshpatator, test_size=0.2, random_state=1)
		x_tr_bruteforce, x_te_bruteforce, y_tr_bruteforce, y_te_bruteforce = train_test_split(
			x_bruteforce, y_bruteforce, test_size=0.2, random_state=1)
		x_tr_sqlinjection, x_te_sqlinjection, y_tr_sqlinjection, y_te_sqlinjection = train_test_split(
			x_sqlinjection, y_sqlinjection, test_size=0.2, random_state=1)
		x_tr_xss, x_te_xss, y_tr_xss, y_te_xss = train_test_split(x_xss, y_xss,
																  test_size=0.2,
																  random_state=1)

		x_tr_infiltration = np.concatenate(
			(x_tr_infiltration_2, x_tr_infiltration_4), axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1, x_tr_portscan_2),
									   axis=0)
		x_tr_webattack = np.concatenate(
			(x_tr_bruteforce, x_tr_sqlinjection, x_tr_xss), axis=0)

		y_tr_infiltration = np.concatenate(
			(y_tr_infiltration_2, y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1, y_tr_portscan_2))
		y_tr_webattack = np.concatenate(
			(y_tr_bruteforce, y_tr_sqlinjection, y_tr_xss))

		x_te_infiltration = np.concatenate(
			(x_te_infiltration_2, x_te_infiltration_4), axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1, x_te_portscan_2),
									   axis=0)
		x_te_webattack = np.concatenate(
			(x_te_bruteforce, x_te_sqlinjection, x_te_xss), axis=0)

		y_te_infiltration = np.concatenate(
			(y_te_infiltration_2, y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1, y_te_portscan_2))
		y_te_webattack = np.concatenate(
			(y_te_bruteforce, y_te_sqlinjection, y_te_xss))

		# play label,因为之前合并了某类攻击，所以需要重新定义标签

		y_tr_botnet = np.array([0] * len(y_tr_botnet))
		y_tr_ddos = np.array([1] * len(y_tr_ddos))
		y_tr_glodeneye = np.array([2] * len(y_tr_glodeneye))
		y_tr_hulk = np.array([3] * len(y_tr_hulk))
		y_tr_slowhttp = np.array([4] * len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5] * len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6] * len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7] * len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8] * len(y_tr_infiltration))
		y_tr_portscan = np.array([9] * len(y_tr_portscan))
		y_tr_sshpatator = np.array([10] * len(y_tr_sshpatator))
		y_tr_webattack = np.array([11] * len(y_tr_webattack))
		# y_tr_benign = np.array([12] * len(y_tr_benign))

		y_te_botnet = np.array([0] * len(y_te_botnet))
		y_te_ddos = np.array([1] * len(y_te_ddos))
		y_te_glodeneye = np.array([2] * len(y_te_glodeneye))
		y_te_hulk = np.array([3] * len(y_te_hulk))
		y_te_slowhttp = np.array([4] * len(y_te_slowhttp))
		y_te_slowloris = np.array([5] * len(y_te_slowloris))
		y_te_ftppatator = np.array([6] * len(y_te_ftppatator))
		y_te_heartbleed = np.array([7] * len(y_te_heartbleed))
		y_te_infiltration = np.array([8] * len(y_te_infiltration))
		y_te_portscan = np.array([9] * len(y_te_portscan))
		y_te_sshpatator = np.array([10] * len(y_te_sshpatator))
		y_te_webattack = np.array([11] * len(y_te_webattack))
		# y_te_benign = np.array([12] * len(y_te_benign))

		x_train = np.concatenate((x_tr_botnet, x_tr_ddos, x_tr_glodeneye,
								  x_tr_hulk, x_tr_slowhttp, x_tr_slowloris,
								  x_tr_ftppatator, x_tr_heartbleed,
								  x_tr_infiltration, x_tr_portscan,
								  x_tr_sshpatator, x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet, y_tr_ddos, y_tr_glodeneye,
								  y_tr_hulk, y_tr_slowhttp, y_tr_slowloris,
								  y_tr_ftppatator, y_tr_heartbleed,
								  y_tr_infiltration, y_tr_portscan,
								  y_tr_sshpatator, y_tr_webattack))
		# x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack,x_tr_benign))
		# y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack,y_tr_benign))
		# x_train = np.concatenate((x_tr_botnet,x_tr_glodeneye))
		# y_train = np.concatenate((y_tr_botnet,y_tr_glodeneye))

		x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye,
								 x_te_hulk, x_te_slowhttp, x_te_slowloris,
								 x_te_ftppatator, x_te_heartbleed,
								 x_te_infiltration, x_te_portscan,
								 x_te_sshpatator, x_te_webattack))
		y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye,
								 y_te_hulk, y_te_slowhttp, y_te_slowloris,
								 y_te_ftppatator, y_te_heartbleed,
								 y_te_infiltration, y_te_portscan,
								 y_te_sshpatator, y_te_webattack))
		# x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye, x_te_hulk, x_te_slowhttp, x_te_slowloris,x_te_ftppatator, x_te_heartbleed, x_te_infiltration, x_te_portscan, x_te_sshpatator,x_te_webattack,x_te_benign))
		# y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye, y_te_hulk, y_te_slowhttp, y_te_slowloris,y_te_ftppatator, y_te_heartbleed, y_te_infiltration, y_te_portscan, y_te_sshpatator,y_te_webattack,y_te_benign))
		# x_test = np.concatenate((x_te_botnet,x_te_glodeneye))
		# y_test = np.concatenate((y_te_botnet,y_te_glodeneye))

		return x_train, y_train, x_test, y_test



#2017labeld
class TrainDataSetLabeld2017():
	def __init__(self, data_type=1):
		self.data_type = data_type
		super(TrainDataSetLabeld2017, self).__init__()

	def read_csv(self):
		if self.data_type == 1:
			mydata_botnet = pd.read_csv(
				'./flow_labeled/labeld_Botnet.csv')  # 2075
			mydata_DDoS = pd.read_csv(
				'./flow_labeled/labeld_DDoS.csv')  # 261226
			mydata_glodeneye = pd.read_csv(
				'./flow_labeled/labeld_DoS-GlodenEye.csv')  # 20543
			mydata_hulk = pd.read_csv(
				'./flow_labeled/labeld_DoS-Hulk.csv')  # 474656
			mydata_slowhttp = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowhttptest.csv')  # 6786
			mydata_slowloris = pd.read_csv(
				'./flow_labeled/labeld_DoS-Slowloris.csv')  # 10537
			mydata_ftppatator = pd.read_csv(
				'./flow_labeled/labeld_FTP-Patator.csv')  # 19941
			mydata_heartbleed = pd.read_csv(
				'./flow_labeled/labeld_Heartbleed-Port.csv')  # 9859
			mydata_infiltration_2 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-2.csv')  # 5126
			mydata_infiltration_4 = pd.read_csv(
				'./flow_labeled/labeld_Infiltration-4.csv')  # 168
			mydata_portscan_1 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_1.csv')  # 755
			mydata_portscan_2 = pd.read_csv(
				'./flow_labeled/labeld_PortScan_2.csv')  # 318881
			mydata_sshpatator = pd.read_csv(
				'./flow_labeled/labeld_SSH-Patator.csv')  # 27545
			mydata_bruteforce = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-BruteForce.csv')  # 7716
			mydata_sqlinjection = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-SqlInjection.csv')  # 25
			mydata_xss = pd.read_csv(
				'./flow_labeled/labeld_WebAttack-XSS.csv')  # 2796
		elif self.data_type == 2:
			mydata_botnet = pd.read_csv(
				'./payload_labeled/labeld_Botnet_payload.csv')
			mydata_DDoS = pd.read_csv(
				'./payload_labeled/labeld_DDoS_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./payload_labeled/labeld_DoS-GlodenEye_payload.csv')
			mydata_hulk = pd.read_csv(
				'./payload_labeled/labeld_DoS-Hulk_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./payload_labeled/labeld_DoS-Slowloris_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./payload_labeled/labeld_FTP-Patator_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./payload_labeled/labeld_Heartbleed-Port_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-2_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./payload_labeled/labeld_Infiltration-4_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_1_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./payload_labeled/labeld_PortScan_2_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./payload_labeled/labeld_SSH-Patator_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
			mydata_xss = pd.read_csv(
				'./payload_labeled/labeld_WebAttack-XSS_payload.csv')
		elif self.data_type == 3:
			mydata_botnet = pd.read_csv(
				'./2017_labeled/labeld_Botnet_head_payload.csv')
			mydata_DDoS = pd.read_csv(
				'2017_labeled/labeld_DDoS_head_payload.csv')
			mydata_glodeneye = pd.read_csv(
				'./2017_labeled/labeld_DoS-GlodenEye_head_payload.csv')
			mydata_hulk = pd.read_csv(
				'./2017_labeled/labeld_DoS-Hulk_head_payload.csv')
			mydata_slowhttp = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowhttptest_head_payload.csv')
			mydata_slowloris = pd.read_csv(
				'./2017_labeled/labeld_DoS-Slowloris_head_payload.csv')
			mydata_ftppatator = pd.read_csv(
				'./2017_labeled/labeld_FTP-Patator_head_payload.csv')
			mydata_heartbleed = pd.read_csv(
				'./2017_labeled/labeld_Heartbleed-Port_head_payload.csv')
			mydata_infiltration_2 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-2_head_payload.csv')
			mydata_infiltration_4 = pd.read_csv(
				'./2017_labeled/labeld_Infiltration-4_head_payload.csv')
			mydata_portscan_1 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_1_head_payload.csv')
			mydata_portscan_2 = pd.read_csv(
				'./2017_labeled/labeld_PortScan_2_head_payload.csv')
			mydata_sshpatator = pd.read_csv(
				'./2017_labeled/labeld_SSH-Patator_head_payload.csv')
			mydata_bruteforce = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-BruteForce_head_payload.csv')
			mydata_sqlinjection = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-SqlInjection_head_payload.csv')
			mydata_xss = pd.read_csv(
				'./2017_labeled/labeld_WebAttack-XSS_head_payload.csv')
		# mydata_benign = pd.read_csv('./head_payload_labeled/labeld_Monday-Benign.csv')

		# benign = mydata_benign.values[:,1:]
		botnet = mydata_botnet.values[:, 484:]
		ddos = mydata_DDoS.values[:, 484:]
		glodeneye = mydata_glodeneye.values[:, 484:]
		hulk = mydata_hulk.values[:, 484:]
		slowhttp = mydata_slowhttp.values[:, 484:]
		slowloris = mydata_slowloris.values[:, 484:]
		ftp_patator = mydata_ftppatator.values[:, 484:]
		heartbleed = mydata_heartbleed.values[:, 484:]
		infiltration_2 = mydata_infiltration_2.values[:, 484:]
		infiltration_4 = mydata_infiltration_4.values[:, 484:]
		portscan_1 = mydata_portscan_1.values[:, 484:]
		portscan_2 = mydata_portscan_2.values[:, 484:]
		ssh_patator = mydata_sshpatator.values[:, 484:]
		bruteforce = mydata_bruteforce.values[:, 484:]
		sqlinjection = mydata_sqlinjection.values[:, 484:]
		xss = mydata_xss.values[:, 484:]

		# return benign,botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss
		return botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss

	# return botnet,glodeneye

	def get_item(self):
		# botnet,glodeneye = self.read_csv()
		botnet, ddos, glodeneye, hulk, slowhttp, slowloris, ftp_patator, heartbleed, infiltration_2, infiltration_4, portscan_1, portscan_2, ssh_patator, bruteforce, sqlinjection, xss = self.read_csv()

		# print('shape of benign: ',benign.shape)
		print('shape of botnet: ', botnet.shape)
		print('shape of DDoS: ', ddos.shape)
		print('shape of glodeneye: ', glodeneye.shape)
		print('shape of hulk: ', hulk.shape)
		print('shape of slowhttp: ', slowhttp.shape)
		print('shape of slowloris: ', slowloris.shape)
		print('shape of ftppatator: ', ftp_patator.shape)
		print('shape of heartbleed: ', heartbleed.shape)
		print('shape of infiltration_2: ', infiltration_2.shape)
		print('shape of infiltration_4: ', infiltration_4.shape)
		print('shape of portscan_1: ', portscan_1.shape)
		print('shape of portscan_2: ', portscan_2.shape)
		print('shape of sshpatator: ', ssh_patator.shape)
		print('shape of brutefoece: ', bruteforce.shape)
		print('shape of sqlinjection: ', sqlinjection.shape)
		print('shape of xss: ', xss.shape)

		# x_benign = benign[:,:-1]
		x_botnet = botnet[:, :-1]
		x_ddos = ddos[:, :-1]
		x_glodeneye = glodeneye[:, :-1]
		x_hulk = hulk[:, :-1]
		x_slowhttp = slowhttp[:, :-1]
		x_slowloris = slowloris[:, :-1]
		x_ftppatator = ftp_patator[:, :-1]
		x_heartbleed = heartbleed[:, :-1]
		x_infiltration_2 = infiltration_2[:, :-1]
		x_infiltration_4 = infiltration_4[:, :-1]
		x_portscan_1 = portscan_1[:, :-1]
		x_portscan_2 = portscan_2[:, :-1]
		x_sshpatator = ssh_patator[:, :-1]
		x_bruteforce = bruteforce[:, :-1]
		x_sqlinjection = sqlinjection[:, :-1]
		x_xss = xss[:, :-1]

		# y_benign = benign[:,-1]
		y_botnet = botnet[:, -1]
		y_ddos = ddos[:, -1]
		y_glodeneye = glodeneye[:, -1]
		y_hulk = hulk[:, -1]
		y_slowhttp = slowhttp[:, -1]
		y_slowloris = slowloris[:, -1]
		y_ftppatator = ftp_patator[:, -1]
		y_heartbleed = heartbleed[:, -1]
		y_infiltration_2 = infiltration_2[:, -1]
		y_infiltration_4 = infiltration_4[:, -1]
		y_portscan_1 = portscan_1[:, -1]
		y_portscan_2 = portscan_2[:, -1]
		y_sshpatator = ssh_patator[:, -1]
		y_bruteforce = bruteforce[:, -1]
		y_sqlinjection = sqlinjection[:, -1]
		y_xss = xss[:, -1]

		# x_tr_benign,x_te_benign,y_tr_benign,y_te_benign = train_test_split(x_benign,y_benign,test_size=0.2,random_state=1)
		x_tr_botnet, x_te_botnet, y_tr_botnet, y_te_botnet = train_test_split(
			x_botnet, y_botnet, test_size=0.2, random_state=1)
		x_tr_ddos, x_te_ddos, y_tr_ddos, y_te_ddos = train_test_split(
			x_ddos[:len(x_ddos)//8], y_ddos[:len(x_ddos)//8],
			test_size=0.2, random_state=1)
		x_tr_glodeneye, x_te_glodeneye, y_tr_glodeneye, y_te_glodeneye = train_test_split(
			x_glodeneye, y_glodeneye, test_size=0.2, random_state=1)
		x_tr_hulk, x_te_hulk, y_tr_hulk, y_te_hulk = train_test_split(
			x_hulk[:len(x_ddos)//8], y_hulk[:len(x_ddos)//8],
			test_size=0.2, random_state=1)
		x_tr_slowhttp, x_te_slowhttp, y_tr_slowhttp, y_te_slowhttp = train_test_split(
			x_slowhttp, y_slowhttp, test_size=0.2, random_state=1)
		x_tr_slowloris, x_te_slowloris, y_tr_slowloris, y_te_slowloris = train_test_split(
			x_slowloris, y_slowloris, test_size=0.2, random_state=1)
		x_tr_ftppatator, x_te_ftppatator, y_tr_ftppatator, y_te_ftppatator = train_test_split(
			x_ftppatator, y_ftppatator, test_size=0.2, random_state=1)
		x_tr_heartbleed, x_te_heartbleed, y_tr_heartbleed, y_te_heartbleed = train_test_split(
			x_heartbleed, y_heartbleed, test_size=0.2, random_state=1)
		x_tr_infiltration_2, x_te_infiltration_2, y_tr_infiltration_2, y_te_infiltration_2 = train_test_split(
			x_infiltration_2, y_infiltration_2, test_size=0.2, random_state=1)
		x_tr_infiltration_4, x_te_infiltration_4, y_tr_infiltration_4, y_te_infiltration_4 = train_test_split(
			x_infiltration_4, y_infiltration_4, test_size=0.2, random_state=1)
		x_tr_portscan_1, x_te_portscan_1, y_tr_portscan_1, y_te_portscan_1 = train_test_split(
			x_portscan_1[:len(x_portscan_1)//8], y_portscan_1[:len(x_portscan_1)//8],
			test_size=0.2, random_state=1)
		x_tr_portscan_2, x_te_portscan_2, y_tr_portscan_2, y_te_portscan_2 = train_test_split(
			x_portscan_2[:len(x_portscan_2)//8], y_portscan_2[:len(x_portscan_2)//8],
			test_size=0.2, random_state=1)
		x_tr_sshpatator, x_te_sshpatator, y_tr_sshpatator, y_te_sshpatator = train_test_split(
			x_sshpatator, y_sshpatator, test_size=0.2, random_state=1)
		x_tr_bruteforce, x_te_bruteforce, y_tr_bruteforce, y_te_bruteforce = train_test_split(
			x_bruteforce, y_bruteforce, test_size=0.2, random_state=1)
		x_tr_sqlinjection, x_te_sqlinjection, y_tr_sqlinjection, y_te_sqlinjection = train_test_split(
			x_sqlinjection, y_sqlinjection, test_size=0.2, random_state=1)
		x_tr_xss, x_te_xss, y_tr_xss, y_te_xss = train_test_split(x_xss, y_xss,
																  test_size=0.2,
																  random_state=1)

		x_tr_infiltration = np.concatenate(
			(x_tr_infiltration_2, x_tr_infiltration_4), axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1, x_tr_portscan_2),
									   axis=0)
		x_tr_webattack = np.concatenate(
			(x_tr_bruteforce, x_tr_sqlinjection, x_tr_xss), axis=0)

		y_tr_infiltration = np.concatenate(
			(y_tr_infiltration_2, y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1, y_tr_portscan_2))
		y_tr_webattack = np.concatenate(
			(y_tr_bruteforce, y_tr_sqlinjection, y_tr_xss))

		x_te_infiltration = np.concatenate(
			(x_te_infiltration_2, x_te_infiltration_4), axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1, x_te_portscan_2),
									   axis=0)
		x_te_webattack = np.concatenate(
			(x_te_bruteforce, x_te_sqlinjection, x_te_xss), axis=0)

		y_te_infiltration = np.concatenate(
			(y_te_infiltration_2, y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1, y_te_portscan_2))
		y_te_webattack = np.concatenate(
			(y_te_bruteforce, y_te_sqlinjection, y_te_xss))

		# play label,因为之前合并了某类攻击，所以需要重新定义标签

		y_tr_botnet = np.array([0] * len(y_tr_botnet))
		y_tr_ddos = np.array([1] * len(y_tr_ddos))
		y_tr_glodeneye = np.array([2] * len(y_tr_glodeneye))
		y_tr_hulk = np.array([3] * len(y_tr_hulk))
		y_tr_slowhttp = np.array([4] * len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5] * len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6] * len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7] * len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8] * len(y_tr_infiltration))
		y_tr_portscan = np.array([9] * len(y_tr_portscan))
		y_tr_sshpatator = np.array([10] * len(y_tr_sshpatator))
		y_tr_webattack = np.array([11] * len(y_tr_webattack))
		# y_tr_benign = np.array([12] * len(y_tr_benign))

		y_te_botnet = np.array([0] * len(y_te_botnet))
		y_te_ddos = np.array([1] * len(y_te_ddos))
		y_te_glodeneye = np.array([2] * len(y_te_glodeneye))
		y_te_hulk = np.array([3] * len(y_te_hulk))
		y_te_slowhttp = np.array([4] * len(y_te_slowhttp))
		y_te_slowloris = np.array([5] * len(y_te_slowloris))
		y_te_ftppatator = np.array([6] * len(y_te_ftppatator))
		y_te_heartbleed = np.array([7] * len(y_te_heartbleed))
		y_te_infiltration = np.array([8] * len(y_te_infiltration))
		y_te_portscan = np.array([9] * len(y_te_portscan))
		y_te_sshpatator = np.array([10] * len(y_te_sshpatator))
		y_te_webattack = np.array([11] * len(y_te_webattack))
		# y_te_benign = np.array([12] * len(y_te_benign))

		x_train = np.concatenate((x_tr_botnet, x_tr_ddos, x_tr_glodeneye,
								  x_tr_hulk, x_tr_slowhttp, x_tr_slowloris,
								  x_tr_ftppatator, x_tr_heartbleed,
								  x_tr_infiltration, x_tr_portscan,
								  x_tr_sshpatator, x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet, y_tr_ddos, y_tr_glodeneye,
								  y_tr_hulk, y_tr_slowhttp, y_tr_slowloris,
								  y_tr_ftppatator, y_tr_heartbleed,
								  y_tr_infiltration, y_tr_portscan,
								  y_tr_sshpatator, y_tr_webattack))
		# x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack,x_tr_benign))
		# y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack,y_tr_benign))
		# x_train = np.concatenate((x_tr_botnet,x_tr_glodeneye))
		# y_train = np.concatenate((y_tr_botnet,y_tr_glodeneye))

		x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye,
								 x_te_hulk, x_te_slowhttp, x_te_slowloris,
								 x_te_ftppatator, x_te_heartbleed,
								 x_te_infiltration, x_te_portscan,
								 x_te_sshpatator, x_te_webattack))
		y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye,
								 y_te_hulk, y_te_slowhttp, y_te_slowloris,
								 y_te_ftppatator, y_te_heartbleed,
								 y_te_infiltration, y_te_portscan,
								 y_te_sshpatator, y_te_webattack))
		# x_test = np.concatenate((x_te_botnet, x_te_ddos, x_te_glodeneye, x_te_hulk, x_te_slowhttp, x_te_slowloris,x_te_ftppatator, x_te_heartbleed, x_te_infiltration, x_te_portscan, x_te_sshpatator,x_te_webattack,x_te_benign))
		# y_test = np.concatenate((y_te_botnet, y_te_ddos, y_te_glodeneye, y_te_hulk, y_te_slowhttp, y_te_slowloris,y_te_ftppatator, y_te_heartbleed, y_te_infiltration, y_te_portscan, y_te_sshpatator,y_te_webattack,y_te_benign))
		# x_test = np.concatenate((x_te_botnet,x_te_glodeneye))
		# y_test = np.concatenate((y_te_botnet,y_te_glodeneye))

		return x_train, y_train, x_test, y_test

class TrainDataSetPayload_2012():

	def __init__(self, ):
		super(TrainDataSetPayload_2012, self).__init__()

	def read_csv(self):
		# payload_benign = pd.read_csv('./2012_labeled/label_normal.csv')
		# payload_nonclassifed = pd.read_csv('./2012_labeled/label_non-classifed_attacks.csv')
		payload_inflter = pd.read_csv('./2012_labeled/label_Infltering.CSV')
		payload_http = pd.read_csv('./2012_labeled/label_HTTP.CSV')
		payload_distributedenial = pd.read_csv('./2012_labeled/label_Distributed_denial.CSV')
		payload_brutessh = pd.read_csv('./2012_labeled/label_Bruteforce_SSH.CSV')

		# print('finish reading dataset, cost time :',time.time() - start)

		# benign = payload_benign.values[:, 1:]
		# nonclassifed = payload_nonclassifed.values[:, 1:]
		inflter = payload_inflter.values[:, :]
		http = payload_http.values[:, :]
		distributedenial = payload_distributedenial.values[:, :]
		brutessh = payload_brutessh.values[:, :]

		return inflter, http, distributedenial, brutessh

	def get_item(self):
		inflter, http, distributedenial, brutessh = self.read_csv()

		# print('shape of benign: ', benign.shape)
		# print('shape of nonclassifed: ', nonclassifed.shape)
		print('shape of inflter: ', inflter.shape)
		print('shape of http: ', http.shape)
		print('shape of distributedenial: ', distributedenial.shape)
		print('shape of rutessh: ', brutessh.shape)

		# 剔除最后一列标签
		# x_benign = benign[:, :-1]
		# x_nonclassifed = nonclassifed[:, :-1]
		x_inflter = inflter[:, :-1]
		x_http = http[:, :-1]
		x_distributedenial = distributedenial[:, :-1]
		x_brutessh = brutessh[:, :-1]

		# 获取标签
		# y_benign = benign[:, -1]
		# y_nonclassifed = nonclassifed[:, -1]
		y_inflter = inflter[:, -1]
		y_http = http[:, -1]
		y_distributedenial = distributedenial[:, -1]
		y_brutessh = brutessh[:, -1]

		# x_tr_benign, x_te_benign, y_tr_benign, y_te_benign = train_test_split(x_benign, y_benign, test_size=0.2, random_state=1)
		# x_tr_nonclassifed, x_te_nonclassifed, y_tr_nonclassifed, y_te_nonclassifed = train_test_split(x_nonclassifed,y_nonclassifed, test_size=0.2,random_state=1)
		x_tr_inflter, x_te_inflter, y_tr_inflter, y_te_inflter = train_test_split(x_inflter[:len(x_inflter)//8], y_inflter[:len(y_inflter)//8], test_size=0.2,random_state=1)
		x_tr_http, x_te_http, y_tr_http, y_te_http = train_test_split(x_http[:len(x_http)//5], y_http[:len(y_http)//5], test_size=0.2, random_state=1)
		x_tr_distributedenial, x_te_distributedenial, y_tr_distributedenial, y_te_distributedenial = train_test_split(x_distributedenial[:len(x_distributedenial)//5], y_distributedenial[:len(y_distributedenial)//5], test_size=0.2, random_state=1)
		x_tr_brutessh, x_te_brutessh, y_tr_brutessh, y_te_brutessh = train_test_split(x_brutessh, y_brutessh, test_size=0.2, random_state=1)

		x_train = np.concatenate(( x_tr_inflter, x_tr_http, x_tr_distributedenial, x_tr_brutessh))
		y_train = np.concatenate((y_tr_inflter, y_tr_http, y_tr_distributedenial, y_tr_brutessh))

		x_test = np.concatenate(( x_te_inflter, x_te_http, x_te_distributedenial, x_te_brutessh))
		y_test = np.concatenate((y_te_inflter, y_te_http, y_te_distributedenial, y_te_brutessh))

		return x_train, y_train, x_test, y_test

