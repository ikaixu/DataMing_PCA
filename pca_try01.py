"""
读取图像：1、找到一个可以把图像转化为矩阵的模块
         2、得到单一图像的矩阵
         3、矢量化单一图像的矩阵
         4、读取所有的图像，并重复上一个过程。
"""
#模块选用matplotlib,矩阵使用numpy。
import matplotlib.image
import numpy

#下面先读取单幅图像，得到数据矩阵，并且矢量化
lena = matplotlib.image.imread("C:\\Coder\\DataMing_PCA\\ORL\\s1\\1.jpg")
#此时会得到一个数组，可以查看type(lena),lena.dtype与lena.shape,dtype返回unit8什么鬼？lena.shape返回(112, 92)
#需要对unit8这种数据进行转换吗？需要
lena = lena.astype(numpy.float64)
#下面对数据进行行矢量化N=(112*92=10304),1*N的行向量
lenb = numpy.reshape(lena, (1, -1), order='F')
#已基本实现上述要求。len.shape验证数组形式是否正确。

#下面读入所有文件，并且实现对上面第一幅图像所做操作的重复。
allsamples = numpy.arange(10304).reshape(1,10304)
for i in range(1,41):
    for j in range(1,6):
        a = matplotlib.image.imread('C:\\Coder\\DataMing_PCA\\ORL\\s'+str(i)+'\\'+str(j)+'.jpg')
        b = numpy.reshape(a, (1, -1), order='F')
        b = b.astype(numpy.float64)
        allsamples = numpy.concatenate((allsamples, b), axis=0)
allsamples = numpy.delete(allsamples, 0, 0)#去除这个数组的第一行,得到所有样本。
allsamples = numpy.asmatrix(allsamples)#数组转化为矩阵，方便使用numpy内置线性代数模块。