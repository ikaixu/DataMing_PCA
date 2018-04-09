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