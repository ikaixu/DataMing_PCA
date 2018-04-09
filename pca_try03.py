#读图(详细过程见pca_try01.py)
import matplotlib.image
import numpy
allsamples = numpy.arange(10304).reshape(1,10304)
for i in range(1,41):
    for j in range(1,6):
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F')
        b = b.astype(numpy.float64)
        allsamples = numpy.concatenate((allsamples, b), axis=0)
allsamples = numpy.delete(allsamples, 0, 0)
allsamples = numpy.asmatrix(allsamples)
#pca过程(详细过程见pca_try02.py)
samplemean = allsamples.mean(axis=0)
xmean = allsamples - samplemean
sigma = xmean*(xmean.T)
v, d = numpy.linalg.eig(sigma)
f = numpy.diag(v)
g = numpy.asmatrix(f)
h = numpy.asmatrix(d)
dsum = numpy.sum(v)
dsum_extract = 0
i = 0
while((dsum_extract/dsum) < 0.9):
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1
base = (xmean.T) * h[:,0:i] * (numpy.diag(v[0:i] ** (-(1/2))))
"""
测试集测试： 1、测试集投影到上面形成的特征空间中
            2、这里使用了距离的方法，用了三阶近邻的算法
            3、写个文档弄清楚什么是三阶近邻
"""
#先去识别单个图像，在谋求对所有的图片进行识别
lena = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(1)+'\\'+str(6)+'.jpg')
lenb = numpy.reshape(a, (1, -1), order='F')
lenb = lenb.astype(numpy.float64)#读取所有文件


#下面对所有图像进行识别，并计算识别正确率。
accu = 0
for i in range(1:41):
    for j in range(6:11):
        #依然是读取图像
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F')
        b = b.astype(numpy.float64)
        tcoor = b * base    #计算坐标系，1*71阶的矩阵
