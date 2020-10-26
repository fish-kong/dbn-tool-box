clc;clear;close all;
tic
%% 加载数据
% load('data_process.mat');
load('data_feature.mat');

trainX=double(train_X);
trainYn=double(train_Y);
testX=double(test_X);
testYn=double(test_Y);
clear train_X train_Y test_X test_Y valid_X valid_Y

%% DBN参数设置
rng(0)
% 网络各层节点
input_num=size(trainX,2);%输入层
hidden_num=[50 20];%隐含层,两个数就是两个隐含层 3个数就是3个隐含层
class=size(trainYn,2);%输出层
nodes = [input_num hidden_num class]; %节点数
% 初始化网络权值
dbn = randDBN(nodes);%调用randDBN
nrbm=numel(dbn.rbm);
opts.MaxIter =100;                % 迭代次数
% opts.BatchSize = round(length(trainYn)/4);  % batch规模为四分之一的训练集trainY的长度进行四舍五入取整
opts.BatchSize = 32;  % batch规模
opts.Verbose = 0;               % 是否展示中间过程
opts.StepRatio = 0.1;             % 学习速率
% opts.InitialMomentum = 0.9;%opts.InitialMomentum为0.7
% opts.FinalMomentum = 0.1;%opts.FinalMomentum为0.8
% opts.WeightCost = 0.005;%opts.WeightCost为0
%opts.InitialMomentumIter = 10;

%% RBM逐层预训练
dbn = pretrainDBN(dbn, trainX, opts);%进行dbn的预训练
%% 线性映射-将训练好的各RBM 堆栈初始化DBN网络
dbn= SetLinearMapping(dbn, trainX, trainYn);%调用SetLinearMapping函数

%% 训练DBN-微调整个DBN
opts.MaxIter =100;                % 迭代次数
% opts.BatchSize = round(length(trainYn)/4);  % batch规模为四分之一的训练集trainY的长度进行四舍五入取整
opts.BatchSize =32;
opts.Verbose = 0;               % 是否展示中间过程
opts.StepRatio = 0.1;             % 学习速率
opts.Object = 'CrossEntropy';            % 目标函数: Square CrossEntropy
%opts.Layer = 1;
dbn = trainDBN(dbn, trainX, trainYn, opts);%dbn调用trainDBN函数

%% 测试
% 对训练集进行预测
trainYn_out = v2h( dbn, trainX );%trainYn_out调用v2h函数
[~,trainY] = max(compet(trainYn'));
[~,trainY_out] = max(compet(trainYn_out'));
%compet是神经网络的竞争传递函数，用于指出矩阵中每列的最大值。对应最大值的行的值为1，其他行的值都为0。
%分类
% 计算准确率
accTrain = sum(trainY==trainY_out)/length(trainY);%accTrain为TrainY==trainY_out'的总和除以length(trainY)

% 画训练集预测结果
figure%图形
plot(trainY,'r o')%画一个名为trainY，红色的圆圈
hold on%hold on 是当前轴及图形保持而不被刷新，准备接受此后将绘制
plot(trainY_out,'g +')%画一个名为trainY_out，绿色的加号
legend('真实值','预测值')%legend(图例1，图例2，)
grid on%画网格
xlabel('样本','fontsize',13)%xlabel(x轴说明)
ylabel('类别','fontsize',13)%ylabel(y轴说明)
title(['原始数据迭代100次训练集准确率：' num2str(accTrain*100) '%'],'fontsize',13)%title(图形名称)

% 对测试集进行预测
testYn_out = v2h( dbn, testX );%testYn_out为调用v2h函数
[~,testY] = max(compet(testYn'));
[~,testY_out] = max(compet(testYn_out'));%compet是神经网络的竞争传递函数，用于指出矩阵中每列的最大值。对应最大值的行的值为1，其他行的值都为0。

% 计算准确率
accTest = sum(testY==testY_out)/length(testY);%accTest为testY==testY_out'的总和除以length(testY)

% 画测试集预测结果
figure%图形
plot(testY,'r o')%画一个名为testY，红色的圆圈
hold on%hold on 是当前轴及图形保持而不被刷新，准备接受此后将绘制
plot(testY_out,'g *')%画一个名为testY_out，绿色的加号
legend('真实值','预测值')%legend(图例1，图例2，)
grid on%画网格
xlabel('样本','fontsize',13)%xlabel(x轴说明)
ylabel('类别','fontsize',13)%ylabel(y轴说明)
title(['原始数据迭代100次测试集准确率：' num2str(accTest*100) '%'],'fontsize',13)%title(图形名称)
toc