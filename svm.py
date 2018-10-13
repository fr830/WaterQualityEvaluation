# -*- coding: utf-8 -*-
# 时间    : 2018/10/13 15:53
# 作者    : xcl

#-*- coding: utf-8 -*-
import pandas as pd

inputfile = 'C:\\Users\\Administrator\\Desktop\\WaterQualityEvaluation\\data/moment.csv' #数据文件
outputfile1 = 'C:\\Users\\Administrator\\Desktop\\cm_train.xls' #训练样本混淆矩阵保存路径
outputfile2 = 'C:\\Users\\Administrator\\Desktop\\cm_test.xls' #测试样本混淆矩阵保存路径
data = pd.read_csv(inputfile, encoding = 'gbk') #读取数据，指定编码为gbk
data = data.values

from numpy.random import shuffle #引入随机函数
shuffle(data) #随机打乱数据
data_train = data[:int(0.8*len(data)), :] #选取前80%为训练数据
data_test = data[int(0.8*len(data)):, :] #选取后20%为测试数据

#构造特征和标签
x_train = data_train[:, 2:]*30
y_train = data_train[:, 0].astype(int)
x_test = data_test[:, 2:]*30
y_test = data_test[:, 0].astype(int)

#导入模型相关的函数，建立并且训练模型
from sklearn import svm
model = svm.SVC()
model.fit(x_train, y_train)
import pickle
pickle.dump(model, open('C:\\Users\\Administrator\\Desktop\\svm.model', 'wb'))
#最后一句保存模型，以后可以通过下面语句重新加载模型：
#model = pickle.load(open('../tmp/svm.model', 'rb'))

#导入输出相关的库，生成混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train)) #训练样本的混淆矩阵
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test)) #测试样本的混淆矩阵

#保存结果
pd.DataFrame(cm_train, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index = range(1, 6), columns = range(1, 6)).to_excel(outputfile2)


'''
'r':默认值，表示从文件读取数据。
'w':表示要向文件写入数据，并截断以前的内容
'a':表示要向文件写入数据，添加到当前内容尾部
'r+':表示对文件进行可读写操作（删除以前的所有数据）
'r+a'：表示对文件可进行读写操作（添加到当前文件尾部）
'b':表示要读写二进制数据
'''