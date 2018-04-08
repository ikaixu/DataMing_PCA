%读取数据的过程，从数据仓中读取训练集（读取建模用的图像），并将其转化为一个矩阵。
%百分号就是matlab里面的注释符号。matlab调试需要设置断点进行调试，一段一段的读取程序，程序中的数据可以在主窗口看到。
allsamples=[];%所有训练图像
for i=1:40
    for j=1:5
        a=imread(strcat('e:\ORL\s',num2str(i),'\',num2str(j),'.jpg'));% 读取图像并转化为矩阵，一个112x92的矩阵
        b=a(1:112*92); % 矢量化，将这个矩阵转化，等价过程：按列排成一列，在对这个只有一列的矩阵转置。
        b=double(b);% double的含义不理解
        allsamples=[allsamples; b]; % allsamples 是一个M * N 矩阵，allsamples 中每一行数据代表一张图片，其中M＝200
    end
end
samplemean=mean(allsamples); % 求这个矩阵的均值向量。和一般均值向量不同，这里是将均值向量转置后的结果。

for i=1:200
    xmean(i,:)=allsamples(i,:)-samplemean; % 矩阵的每一列都减去这个列相对应的均值，多元统计课本有相关描述。
end;

%%pca过程
% 获取特征值及特征向量，奇异值分解见多元统计课本。
sigma=xmean*xmean'; 
[v d]=eig(sigma);
d1=diag(d);

% 按特征值大小以降序排列
dsort = flipud(d1);
vsort = fliplr(v);

%以下选择90%的能量
dsum = sum(dsort);
    dsum_extract = 0;
    p = 0;
    while( dsum_extract/dsum < 0.9)
        p = p + 1;
        dsum_extract = sum(dsort(1:p));
    end
i=1;

base = xmean' * vsort(:,1:p) * diag(dsort(1:p).^(-1/2));
allcoor = allsamples * base;
accu = 0;


%%测试过程，测试集数据
for i=1:40
    for j=6:10 %读入40 x 5 副测试图像
        a=imread(strcat('e:\ORL\s',num2str(i),'\',num2str(j),'.jpg'));
        b=a(1:10304);
        b=double(b);
        tcoor= b * base; %计算坐标，是1×p 阶矩阵
        for k=1:200
            mdist(k)=norm(tcoor-allcoor(k,:));
        end;
%三阶近邻
    [dist,index2]=sort(mdist);
        class1=floor( (index2(1)-1)/5 )+1;
        class2=floor((index2(2)-1)/5)+1;
        class3=floor((index2(3)-1)/5)+1;
        if class1~=class2 && class2~=class3
                class=class1;
            elseif class1==class2
                class=class1;
            elseif class2==class3
                class=class2;
            end;
            if class==i
                accu=accu+1;
            end;
    end;
end;
accuracy=accu/200 %输出识别


