#读取图像
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
"""
下面就是pca过程: 1、计算均值矩阵
                2、计算协方差矩阵
                3、协方差阵的特征向量、特征值
                4、选取满足一定要求的累积贡献率的特征值
                5、最后求形成的特征空间(一组基变换)
"""
#计算列的均值
samplemean = allsamples.mean(axis=0)
#去中心化的矩阵
xmean = allsamples - samplemean
#大矩阵的特征值用小矩阵来求
sigma = xmean*(xmean.T)
v, d = numpy.linalg.eig(sigma)#调用numpy的线性代数模块求解
#默认结果中特征值是按照从大到小排序的，所以不用处理。
f = numpy.diag(v)
g = numpy.asmatrix(f)
h = numpy.asmatrix(d)
#选择90%的能量
dsum = numpy.sum(v)
dsum_extract = 0
i = 0
while((dsum_extract/dsum) < 0.9):
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1
#这一步输出i的结果为71。说明循环次数是71次。
#下面特征脸形成的坐标系
base = (xmean.T) * h[:,0:i] * (numpy.diag(v[0:i] ** (-(1/2))))
"""
说明：这里和我自己理解的pca有一些区别，需要仔细研究后再解决。
"""