#特例进行整理总结
import numpy as np

#创建实例
a = np.mat('[1,9,10;4,12,16;2,10,12;5,8,13;3,11,14]')

#均值化处理
b = a - a.mean(axis=0)

#方差协方差矩阵
c = np.cov(b.T)

#协方差矩阵的特征值特征向量
d, e = np.linalg.eig(c)#需要验证特征值是否按从大到小的顺序排列。注：d是一维数组，e是多维数组。
f = np.diag(d)#对角阵以d中元素取值，得到的f是数组数据类型。
g = np.asmatrix(f)#把f转换成矩阵。
h = np.asmatrix(e)#把e转化为矩阵。

#找出特征值累计超过90%的特征值，找出循环次数
dsum = np.sum(d)
dsum_extract = 0
i = 0
while((dsum_extract/dsum) < 0.9):
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1

#按循环次数切片，得到变换矩阵。
k = h[:,0:i]

#数据降维
b*k