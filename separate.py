import pandas as pd
import numpy as np
import os


def split80():
    # path = r'./head_payload_labeled'
    # file_list = os.listdir(path)
    # file_list.remove('labeld_Monday-Benign.csv')
    file_list = ['labeld_Monday-Benign.csv']
    # mydata_botnet = pd.read_csv('labeld_Botnet_head_payload.csv')

    for csv in file_list:
        mydata_botnet = pd.read_csv('./head_payload_labeled/{0}'.format(csv))
        botnet = mydata_botnet.values[:,:-5]
        # blank = botnet[:,0].reshape(-1,1)
        botnet = botnet[:,1:]  #去掉第一列
        y = mydata_botnet.values[:,-1]
        new_botnet = []
        val = botnet[1]
        for val in botnet:
            temp = []
            for i in range(5):
                split = val[i*96:(i+1)*96]
                new_data = split[:80]
                temp = np.concatenate((temp,new_data),axis=0)
            new_botnet.append(temp)

        new_botnet = np.concatenate((new_botnet,y.reshape(-1,1)),axis=1)
        xml_df = pd.DataFrame(new_botnet)
        xml_df.to_csv('./80_head_payload_labeled/{0}'.format(csv), index=True)

# csv内部追加
class MakePandas():
    def __init__(self):
        super(MakePandas, self).__init__()

    def append_excel(self, df, content_list):
        """
        excel文件中追加内容
        :return:
        content_list:待追加的内容列表
        """
        print("进入主任务")
        ds = pd.DataFrame(content_list)
        print(ds)
        df = df.append(ds, ignore_index=True)
        excel_name = '1.csv'
        # excel_path = os.path.dirname(os.path.abspath(__file__)) + excel_name
        excel_path = excel_name
        df.to_csv(excel_path, index=False, header=True)
        return df

    def remove_row(self, df, row_list):
        """
        excel删除指定列
        :param df:
        :param row_list:
        :return:
        """
        df = df.drop(columns=row_list)
        return df

    def create_excel(self):
        """
        创建excel文件
        :return:
        """
        # file_path = os.path.dirname(os.path.abspath(__file__)) + "/1.csv"
        file_path = "1.csv"
        df = pd.DataFrame(columns=["loss", "accuracy"])
        df.to_csv(file_path, index=False)





if __name__ == '__main__':
    split80()
    # loss_acc = {'loss':[],'accuracy':[]}
    m = MakePandas()
    # m.create_excel()

    loss_acc = {'loss':[],'accuracy':[]}
    for i in range(10):
        loss_acc['loss'].append(i)
        loss_acc['accuracy'].append(i+1)
    dp = pd.DataFrame(loss_acc)
    dp.to_csv('1.csv',index=None)
    # df['accuracy'].append(1)
    excel_name = "1.csv"
    df = pd.read_csv(excel_name)

    b = []

    for i in range(1, 10):
        a = []
        a.append(i)
        a.append(i * 2)
        b.append(a)

    df = m.append_excel(df, dp)
    df = m.append_excel(df, dp)
    # print(df)






