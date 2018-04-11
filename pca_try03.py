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
base = (xmean.T) * h[:,0:(i-1)] * (numpy.diag(v[0:(i-1)] ** (-(1/2))))
allcolor = allsamples * base    #所有训练样本投影到该空间中,(200, 71)的矩阵。
"""
测试集测试： 1、测试集投影到上面形成的特征空间中
            2、这里使用了距离的方法，用了三阶近邻的算法
            3、写个文档弄清楚什么是三阶近邻
"""
#先去识别单个图像，在谋求对所有的图片进行识别
#第一步：读取一幅图像，并转化为数据矩阵
lena = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(1)+'\\'+str(6)+'.jpg')
lenb = numpy.reshape(lena, (1, -1), order='F')#矢量化
lenb = lenb.astype(numpy.float64)
#第二步：计算坐标
lentcoor = lenb * base #得到的是一个(1,71)的矩阵
#第三步：就是怎么判别的问题。三阶近邻
"""
三阶近邻：三阶近邻法是计算像素的差值的绝对值。
三阶近邻法计算出与测试图像距离最小的三幅图像，
记这三幅图像所属的类分别计为class1，class2，
class3，若class1和class2且class2和class3不
属于同一类，则测试图像属于class1；若class1和
class2相同，则测试图像属于class1，而class2与
测试图像也是相似的；若class2和class3属于同一类，
则测试图像属于class2，而class3与测试图像也是
相似的，但class1虽然与测试图像距离最近却不属于
同一类，可能是由测试图像的姿态和饰物引起的。
"""
#先求每幅训练样本图像与这一幅图像的像素差值。
#mdist = allcolor - lentcoor #命令测试见sy.py
#求mdist的范数。(二阶范数数衡量距离)得到范数数组。
#下面把求像素差值和求距离范数相结合起来处理。
mdist = numpy.arange(200.00).reshape(1,200)
for k in range(0,200):
    mdist[0,k] = numpy.linalg.norm(lentcoor - allcolor[k,:])
#得到的mdist是一个(1,200)的数组。
#对mdist的范数进行排序，从大到小排序。
mdist.sort()#这个内置排序操作会改变原来数组
#再次输入mdist时，这个数组是按照升序进行排序的。
#三阶近邻
#选取距离最近的作为三个，分别为class1、class2、class3(class在python中是关键字，改名字)
#三阶近邻
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
accu = 0
#下面代码是问题？怎么理解，上面类别的构造似乎不正确
if (lei == i):
    accu = accu + 1
#下面对所有图像进行识别，并计算识别正确率。
"""
accu = 0
for i in range(1:41):
    for j in range(6:11):
        #依然是读取图像
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F')
        b = b.astype(numpy.float64)
        tcoor = b * base    #计算坐标系，(1, 71)的矩阵
"""