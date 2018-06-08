%pca人脸识别,一阶近邻分类器
%读取数据
allsamples=[];%样本储存
for i=1:40
    for j=1:7
        a=imread(strcat('./ORL/s',num2str(i),'/',num2str(j),'.jpg'));% 读取样本
        b=a(1:112*92); % 矢量化
        b=double(b);% 转化数据类型
        allsamples=[allsamples; b]; % allsamples 为M * N的矩阵,M大小是200
    end
end
samplemean=mean(allsamples); % 均值化

for i=1:280
    xmean(i,:)=allsamples(i,:)-samplemean; % 所有样本
end;

%%pca过程
% 大矩阵转化为小矩阵
sigma=xmean*xmean'; 
[v d]=eig(sigma);
d1=diag(d);

% 按照特征值大小排序
dsort = flipud(d1);
vsort = fliplr(v);

%大小为90%的贡献率
dsum = sum(dsort);
    dsum_extract = 0;
    p = 0;
    while( dsum_extract/dsum < 0.90)
        p = p + 1;
        dsum_extract = sum(dsort(1:p));
    end
i=1;

base = xmean' * vsort(:,1:p) * diag(dsort(1:p).^(-1/2));
allcoor = allsamples * base;
accu = 0;


%%测试过程
for i=1:40
    for j=8:10 %读取后五张照片
        a=imread(strcat('./ORL/s',num2str(i),'/',num2str(j),'.jpg'));
        b=a(1:10304);
        b=double(b);
        tcoor= b * base; %映射到特征空间
        for k=1:280
            mdist(k)=norm(tcoor-allcoor(k,:));
        end;
%���׽���
    [dist,index2]=sort(mdist);
        class1=floor( (index2(1)-1)/7 )+1;
%        class2=floor((index2(2)-1)/5)+1;
%        class3=floor((index2(3)-1)/5)+1;
%        if class1~=class2 && class2~=class3
%                class=class1;
%            elseif class1==class2
%                class=class1;
%            elseif class2==class3
%                class=class2;
%            end;
            if class1==i
                accu=accu+1;
            end;
    end;
end;
accuracy=accu/120 %准确率
fid = fopen("out.txt","a")
fdisp(fid,accuracy)
fclose (fid)
%%outcome
%% 训练样本量为5,测试样本量为5,贡献率96%,准确率0.88
%% 训练样本量为5,测试样本量为5,贡献率93%,准确率0.875
%% 训练样本量为5,测试样本量为5,贡献率90%,准确率0.88
%% 训练样本量为5,测试样本量为5,贡献率87%,准确率0.87
%% 训练样本量为5,测试样本量为5,贡献率84%,准确率0.86
%% 训练样本量为5,测试样本量为5,贡献率81%,准确率0.86
%% 训练样本量为5,测试样本量为5,贡献率78%,准确率0.86
%% 训练样本量为5,测试样本量为5,贡献率75%,准确率0.84
%% 训练样本量为5,测试样本量为5,贡献率72%,准确率0.795
%% 训练样本量为5,测试样本量为5,贡献率69%,准确率0.80
%% 训练样本量为5,测试样本量为5,贡献率66%,准确率0.775
%% 训练样本量为5,测试样本量为5,贡献率63%,准确率0.765
%% 训练样本量为5,测试样本量为5,贡献率60%,准确率0.765
%% 训练样本量为5,测试样本量为5,贡献率57%,准确率0.74