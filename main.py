#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np 
import time
import os
import random
import pandas as pd

import another
import netmodel
import prepare
import torch
import nets as my_new_nets
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from metric import CosFace,ArcMargin,SphereFace
from loss import FocalLoss
import seaborn as sns



print('\nreading datset, waiting ...... ')
start = time.time()
"""
dataset: 1 ISCX 2012  2 CIC-IDS 2017
data_type in 2017: 1 header only,2 payload only, 3 header and payload
"""
dataset=2

if dataset==1:
	header_payload = True
	train_data_type = prepare.TrainDataSetPayload_2012()
elif dataset==2:
	data_type = 3
	header_payload = False
	if data_type == 3:
		header_payload = True
	train_data_type = prepare.TrainDataSetMul2017(data_type=data_type)

x_train,y_train,x_test,y_test = train_data_type.get_item()
print('\nfinish reading dataset, cost time :',time.time() - start)

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#hyper-parameters

num_workers = 0
#改动2012的类别是4,2017类别是7
NUM_CLASSES = 12
# NUM_CLASSES = 12
batch_size = 256
workers = 0
lr = 1e-4
lr_decay = 5
weight_decay = 1e-4

stage = 0
start_epoch = 0
stage_epochs = [5,3,2]  
total_epochs = sum(stage_epochs)
# total_epochs = 1

best_precision = 0
lowest_loss = 100
print_freq = 1
evaluate = False
resume = False
train_val = False
metric = 'cosface'  # [cosface, arcface] 分类器
# 改动
embedding_size = 12  # 分类器输入
# embedding_size = 4  # 分类器输入
loss = 'cross_entropy' # ['focal_loss', 'cross_entropy']
# 修改读取
train_model = 'plot'
model_type = 'plot'

if not os.path.exists('./model/%s' %model_type):
	os.makedirs('./model/%s' %model_type)

if not os.path.exists('./result/%s' %model_type):
	os.makedirs('./result/%s' %model_type)

if not os.path.exists('./result/%s.txt' %model_type):
	with open('./result/%s.txt' %model_type,'w') as acc_file:
		pass
	
with open('./result/%s.txt' %model_type,'a') as acc_file:
	acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), model_type))

file_path = "./result/{0}/loss.csv".format(model_type)  # 创建CSV文件用于储存LOSS和accuracy
df = pd.DataFrame(columns=["loss", "accuracy"])
df.to_csv(file_path, index=False)



print('\npreparing data, wait wait wait ...')
data_train,data_val,label_train,label_val = train_test_split(x_train,y_train,test_size=0.05,random_state=930802)
my_train_data = np.concatenate((data_train,label_train.reshape(len(label_train),1)),axis=1)
my_val_data = np.concatenate((data_val,label_val.reshape(len(label_val),1)),axis=1)
my_test_data = np.concatenate((x_test,y_test.reshape(len(y_test),1)),axis=1)

train_data = prepare.MDealDataSet2017(my_train_data,header_payload=header_payload)
validate_data = prepare.MDealDataSet2017(my_val_data,header_payload=header_payload)
test_data = prepare.MDealDataSet2017(my_test_data,header_payload=header_payload)
print('trian dataset shape: ',train_data.xshape)
print('validate dataset shape: ',validate_data.xshape)
print('test dataset shape: ',test_data.xshape)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,)
validate_loader = DataLoader(validate_data,batch_size=batch_size*2,shuffle=False,num_workers=num_workers,pin_memory=True)
test_loader = DataLoader(test_data,batch_size=batch_size*2,shuffle=False,num_workers=num_workers,pin_memory=True)


# total models
my_models = {
	'CROSS_CNN':my_new_nets.CROSS_CNN(num_class=NUM_CLASSES,head_payload=header_payload),
	# 'CNN_LSTM':my_new_nets.CNN_LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'p_lstm_cnn':my_new_nets.p_lstm_cnn(num_class=NUM_CLASSES,head_payload=header_payload),
	'CROSS_CNN_LSTM':my_new_nets.CROSS_CNN_LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'TPCNN':my_new_nets.TPCNN(num_class=NUM_CLASSES,head_payload=header_payload),
	'CDLSTM':my_new_nets.CDLSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'TPCNN_C':my_new_nets.TPCNN_C(num_class=NUM_CLASSES,head_payload=header_payload),
	'CNN':my_new_nets.CNN(num_class=NUM_CLASSES,head_payload=header_payload),
	'DILSTM':my_new_nets.DILSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'HPM':my_new_nets.HPM(num_class=NUM_CLASSES,head_payload=header_payload),

	'LSTM':another.LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'CNN_LSTM':another.CNN_LSTM(num_class=NUM_CLASSES,head_payload=header_payload),

	'CNN2017Label':netmodel.CNN_NORMAL(num_class=NUM_CLASSES,head_payload=header_payload),
	'LSTM2017Label':netmodel.LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'CNN_LSTM2017Label':netmodel.CNN_LSTM(num_class=NUM_CLASSES,head_payload=header_payload),

	'Multimoding2017':netmodel.Multimoding2017(num_class=NUM_CLASSES,head_payload=header_payload),
	'Multimoding2017noFCN':netmodel.Multimoding2017noFCN(num_class=NUM_CLASSES,head_payload=header_payload),
}
model = my_models['Multimoding2017noFCN']
model = torch.nn.DataParallel(model).cuda()

print(model)

if metric == 'arcface':
	# metric = ArcFace(embedding_size, NUM_CLASSES).cuda()
	metric = ArcMargin(embedding_size, NUM_CLASSES).cuda()
elif metric == 'cosface':
	metric = CosFace(embedding_size, NUM_CLASSES).cuda()
elif metric == 'sphereface':
	metric = SphereFace(embedding_size, NUM_CLASSES).cuda()

if loss == 'focal_loss':
	loss_function = FocalLoss(gamma=2,num_classes = 12).cuda()
else:
	loss_function = nn.CrossEntropyLoss().cuda()


# loss_function = nn.CrossEntropyLoss().cuda()
#
optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay,amsgrad=True)
# optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric.parameters()}],lr,weight_decay=weight_decay,amsgrad=True)
train_start = time.time()

if evaluate:
	prepare.validate(validate_loader,model,loss_function,best_precision,lowest_loss)
else:
	for epoch in range(start_epoch,total_epochs):
		#train for one epoch
		loss_acc = prepare.train(train_loader,model,metric,loss_function,optimizer,epoch)
		# update the value of loss and accuracy
		save_acc = pd.read_csv("./result/{0}/loss.csv".format(model_type))
		dataframe = pd.DataFrame(loss_acc)
		save_acc = save_acc.append(dataframe, ignore_index=True)
		save_acc.to_csv("./result/{0}/loss.csv".format(model_type), index=False, header=True)

		# evaluate on validate set,返回验证集的平均准确值和平均损失值
		accuracy, avg_loss = prepare.validate(validate_loader, model, metric, loss_function, best_precision,lowest_loss)

		with open('./result/%s.txt' %model_type,'a') as acc_file:
			acc_file.write('Epoch: %2d, Accuracy: %.8f, Loss: %.8f\n' % (epoch, accuracy, avg_loss))

		is_best = accuracy > best_precision
		is_lowest_loss = avg_loss < lowest_loss
		best_precision = max(accuracy,best_precision)
		lowest_loss = min(avg_loss,lowest_loss)
		state = {
		 	'epoch':epoch,
		 	'state_dict':model.state_dict(),
		 	'best_precision':best_precision,
		 	'lowest_loss':lowest_loss,
		 	'stage':stage,
		 	'lr':lr
		}

		prepare.save_checkpoint(state,is_best,is_lowest_loss,model_type)

		if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:  # 第5和第8个epch改变学习率
			stage += 1
			optimizer = prepare.adjust_learning_rate(model,metric,weight_decay,lr,lr_decay)
			model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' %model_type)['state_dict'])
			print('\n \nStep next stage .........\n \n')
			with open('./result/%s.txt' % model_type,'a') as acc_file:
				acc_file.write('\n---------------------Step next stage---------------------\n')
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("		finish training cost time: %ss" %(time.time() - train_start))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

with open('./result/%s.txt' % model_type,'a') as acc_file:
	acc_file.write("*** best accuracy: %.8f %s ***\n" %(best_precision,model_type))

with open('./result/best_acc.txt', 'a') as acc_file:
	acc_file.write('%s  * best acc: %.8f  %s\n' % (
	time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision,model_type))

test_start = time.time()
result = prepare.test(test_loader,model,metric,num_class=NUM_CLASSES,topk=4)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("		finish testing cost time: %ss" %(time.time() - test_start))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

torch.cuda.empty_cache()

top1_pred_label,actual_label = result[0],result[1]
mcm = multilabel_confusion_matrix(actual_label, top1_pred_label)
tp = mcm[:, 1, 1]
tn = mcm[:, 0, 0]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]

conf_mtx= confusion_matrix(actual_label,top1_pred_label)
print('\nConfusion Matrix:')
print(conf_mtx)
# 画出混淆矩阵
xtick=['botnet','DDoS','GlodenEye','Hulk','slowhttp','slowloris','Ftppatator','heartbleed','infiltration','portscan','sshpatator','webattack']
ytick=['botnet','DDoS','GlodenEye','Hulk','slowhttp','slowloris','Ftppatator','heartbleed','infiltration','portscan','sshpatator','webattack']
#xtick=['infiltration','Http_DDoS','DDoS','BruteForce_SSH']
#ytick=['infiltration','Http_DDoS','DDoS','BruteForce_SSH']

plt.figure(figsize = (10,7))
mask = np.zeros_like(conf_mtx)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
	ax = sns.heatmap(conf_mtx,cmap='Blues',annot=True,xticklabels=xtick,yticklabels=ytick,fmt="d")
# ax.set_yticklabels(heatmap_labels)

# plt.colorbar()
plt.xlabel('predict type')
plt.ylabel('actual type')
plt.show()


def multi_index(s,str):   #str是要查询的字符
    length = len(s)     #获取该字符串的长度
    str1 = s            #拷贝字符串
    list = []
    sum = 0             #用来计算每次截取完字符串的总长度
    try:
        while str1.index(str)!=-1:      #当字符串中没有该字符则跳出
            n = str1.index(str)         #查询查找字符的索引
            str2 = str1[0:n + 1]        #截取的前半部分
            str1 = str1[n + 1:length]   #截取的后半部分
            sum = sum + len(str2)       #计算每次截取完字符串的总长度
            list.append(sum - 1)        #把所有索引添加到列表中
            length=length-len(str2)     #截取后半部分的长度
    except ValueError:
        return list
    return list
def num2str(data):
	str_data = []
	for x in data:
		str_data.append(str(round(x,4)))
	my_str = " ".join(str_data)
	return my_str


#改动，2017是12个类别，2012是4个
target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9','class 10','class 11']
#target_names = ['class 0', 'class 1', 'class 2','class 3',]
report = classification_report(actual_label, top1_pred_label, target_names=target_names)
correct_num = []
index = multi_index(report,'\n')
total_num = []
#人为的获取每个类别的数量
for i in index[2:len(index)-4]:
	temp = ''
	dismis = 1
	while(report[i-dismis].isdigit()):
		temp+= report[i-dismis]
		dismis+=1
	total_num.append(int(temp[::-1]))
total_num = np.asarray(total_num)


for r,w in enumerate(conf_mtx):
	correct_num.append(w[r])
correct_num = np.asarray(correct_num)
accuracy = accuracy_score(actual_label, top1_pred_label)
FAR = fp/(fp+tn)
top1_precision = precision_score(actual_label, top1_pred_label, average=None)
top1_recall = recall_score(actual_label, top1_pred_label, average=None)
top1_f1_score = f1_score(actual_label, top1_pred_label, average=None)
pr = top1_precision * top1_recall
apr = (correct_num / total_num) * top1_precision * top1_recall


print(report)
print("accuracy            :",accuracy)
print('e_accuracy		   :',num2str(correct_num / total_num))
print("precision 		   :",num2str(top1_precision))
print("recall 			   :",num2str(top1_recall))
print('f1-socre 		   :',num2str(top1_f1_score))
print('precision_recall    :',num2str(pr))
print('acc_precision_recall:',num2str(apr))
print('False Acceptance Rate',num2str(FAR))



with open(train_model + '.txt','w',encoding='utf-8') as f:
	f.write("\n****************************  " + train_model + "  ****************************\n")
	f.write(report)
	f.write("\naccuracy :\n")
	f.write(str(round(accuracy,4)))
	f.write("\neach accuracy:\n")
	f.write(num2str(correct_num / total_num))
	f.write('\nprecision :\n')
	f.write(num2str(top1_precision))
	f.write("\nrecall :\n")
	f.write(num2str(top1_recall))
	f.write('\nf1-socre :\n')
	f.write(num2str(top1_f1_score))
	f.write('\nFAR :\n')
	f.write(num2str(FAR))
	f.write("\n*****************************************************************")

print("done!")
