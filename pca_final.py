#读图(所有训练样本)
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
#pca计算特征子空间
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
base = (xmean.T) * h[:,0:(i-1)] * (numpy.diag(v[0:(i-1)] ** (-(1/2))))
allcolor = allsamples * base
#测试训练集(三阶近邻法)，输出样本识别率：
accu = 0
for i in range(1,41):
    for j in range(6,11):
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F') #矢量化
        b = b.astype(numpy.float64)
        tcoor = b * base
        mdist = numpy.arange(200.00).reshape(1,200)
        for k in range(0,200):
            mdist[0,k] = numpy.linalg.norm(tcoor - allcolor[k,:])
        mdist = numpy.argsort(mdist)
        lei1 = int((mdist[0,0] - 1)/5) + 1
        lei2 = int((mdist[0,1] - 1)/5) + 1
        lei3 = int((mdist[0,2] - 1)/5) + 1
        lei = 0
        if (lei1!=lei2 and lei2!=lei3):
            lei = lei1
        elif (lei1==lei2 and lei2!=lei3):
            lei = lei1
        elif (lei1!=lei2 and lei2==lei3):
            lei = lei2
        else:
            lei = lei1
        if (lei == i):
            accu = accu + 1
accurancy = accu / 200