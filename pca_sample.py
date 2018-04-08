##pca实验：把一个特殊数据矩阵进行降维。
import numpy as np
a = np.mat('[1,9,10;4,12,16;2,10,12;5,8,13;3,11,14]')
a.mean(axis=0)#矩阵a的均值向量，列转置。这里用的是对象方法调用。
b = a - a.mean(axis=0)#标准化后的向量
c = np.cov(b.T)#按列求取的方差协方差矩阵。这里是模块函数调用。
###########################################################
###########################################################
#下面求特征值特征向量。
#关键问题是：不知道svd和eig之间的区别。
#这里只是实用eig就可以。但是何时使用的是svd呢？
d, e = np.linalg.eig(c)
#其中d指的是特征值形成的数组，而e是特征向量形成的矩阵（行矩阵）
#似乎还是不太明白，必须弄清楚这个numpy函数的含义，清晰解读结果
#如何解读见sy.py
#生成一个对角阵
f = np.diag(d)
g = np.asmatrix(f)
#下面选择主成分，依据方差贡献率,90%的贡献率（暂时没用的代码）。
dsum = np.sum(d)
dsum_extract = 0
i = 0
while(dsum_extract/dsum > 0.9):
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1  
################################################################
#测试代码,切片方式
#e[:,i],表示截取第i+1列，返回一个一维数组（前提e表示的是数组）
#如果e转化为matrix会返回什么。
h = np.asmatrix(e)#转化成矩阵
#h[:,i],切片返回列排列的矩阵。本质上是个列向量。
##################################################################
##################################################################
# 关键问题是找到这个i值，然后截取前i项的特征值的特征向量，得到变换矩阵。
#显然上述代码还不能找到所需要的i,动态的数组去存储数据
def counti:
    dsum = np.sum(d)
    dsum_extract = 0
    i = 0
    while(dsum_extract/dsum > 0.9):
       dsum_extract = g[i,i]+dsum_extract
       i = i + 1  
    return i
#这也不是解决方案，不能把循环次数找到。
dsum = np.sum(d)
dsum_extract = 0
i = 0
while((dsum_extract/dsum) < 0.9):
    j = 0
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1
# 这样可以吗？原来问题是在循环表达式的错误
i#测试返回怎样的i值
# 结论：这里返回了正确的循环次数值，但并不是正确的索引值
# 测试g[i,i]的返回值是否正确。
g[0,0]
#结论：正确无误。
#虽然索引值返回不正确，但用于切片又是刚刚好。
h[:,0:i]#这显然就是我们想要的系数矩阵
k = h[:,0:i]
a*k#实现原来的三维变量向二维变量的转化。
#反问一个问题，这里的处理正确吗？k是一个基变换
#再次梳理pca的原理：
"""
pca过程：1、输入矩阵a，求a的调整矩阵为b、求b的协方差矩阵为c
2、依据c的特征值和特征向量找到主成分，特征值从大到小排序，累计特征值的和达到一定的比率
3、调整的矩阵b*k主成分矩阵
"""