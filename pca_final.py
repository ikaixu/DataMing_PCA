#导入本次所使用的库
import matplotlib.image
import numpy

###读取训练样本
allsamples = numpy.arange(10304).reshape(1,10304) #创建一个初始化数组
for i in range(1,41):
    for j in range(1,6):
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')#读取图像并转化为矩阵
        b = numpy.reshape(a, (1, -1), order='F')#矢量化，把所有数据排成一列，第二列接到第一列后面，以此类推。把得到的(N,1)数组转置为(1,N)
        b = b.astype(numpy.float64)#默认读取数据的数据类型为unit8，现在把它转化成float64
        allsamples = numpy.concatenate((allsamples, b), axis=0)#存入长度相同的数组中
allsamples = numpy.delete(allsamples, 0, 0)#删去数组第一列
allsamples = numpy.asmatrix(allsamples)#训练样本的数据矩阵


####pca计算特征子空间
samplemean = allsamples.mean(axis=0)#训练数据均值
xmean = allsamples - samplemean#去中心化
sigma = xmean*(xmean.T)#大矩阵转化成小矩阵
v, d = numpy.linalg.eig(sigma)#求小矩阵的特征值特征向量,这一步中得到的特征值从大到小排序
f = numpy.diag(v)#依次从大到小的特征值组成的对角数组
g = numpy.asmatrix(f)#上述对角数组化为对角矩阵
h = numpy.asmatrix(d)#特征向量数组化为矩阵
#计算累计贡献率为0.9时的循环次数
dsum = numpy.sum(v)#特征值求和
dsum_extract = 0
i = 0
while((dsum_extract/dsum) < 0.9):
    dsum_extract = g[i,i]+dsum_extract
    i = i + 1
base = (xmean.T) * h[:,0:(i-1)] * (numpy.diag(v[0:(i-1)] ** (-(1/2))))#基变换
allcolor = allsamples * base#训练样本降维


###测试训练集(三阶近邻法)，输出样本识别率：
accu = 0
for i in range(1,41):
    for j in range(6,11):
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F') #矢量化
        b = b.astype(numpy.float64)#数据类型转换
        tcoor = b * base#测试样本数据降维
        #三阶近邻
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
        elif (lei1==lei3 and lei1!=lei2):
            lei = lei1
        else:
            lei = lei1
        if (lei == i):
            accu = accu + 1
accurancy = accu / 200