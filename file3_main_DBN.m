clc;clear;close all;
tic
%% ��������
% load('data_process.mat');
load('data_feature.mat');

trainX=double(train_X);
trainYn=double(train_Y);
testX=double(test_X);
testYn=double(test_Y);
clear train_X train_Y test_X test_Y valid_X valid_Y

%% DBN��������
rng(0)
% �������ڵ�
input_num=size(trainX,2);%�����
hidden_num=[50 20];%������,�������������������� 3��������3��������
class=size(trainYn,2);%�����
nodes = [input_num hidden_num class]; %�ڵ���
% ��ʼ������Ȩֵ
dbn = randDBN(nodes);%����randDBN
nrbm=numel(dbn.rbm);
opts.MaxIter =100;                % ��������
% opts.BatchSize = round(length(trainYn)/4);  % batch��ģΪ�ķ�֮һ��ѵ����trainY�ĳ��Ƚ�����������ȡ��
opts.BatchSize = 32;  % batch��ģ
opts.Verbose = 0;               % �Ƿ�չʾ�м����
opts.StepRatio = 0.1;             % ѧϰ����
% opts.InitialMomentum = 0.9;%opts.InitialMomentumΪ0.7
% opts.FinalMomentum = 0.1;%opts.FinalMomentumΪ0.8
% opts.WeightCost = 0.005;%opts.WeightCostΪ0
%opts.InitialMomentumIter = 10;

%% RBM���Ԥѵ��
dbn = pretrainDBN(dbn, trainX, opts);%����dbn��Ԥѵ��
%% ����ӳ��-��ѵ���õĸ�RBM ��ջ��ʼ��DBN����
dbn= SetLinearMapping(dbn, trainX, trainYn);%����SetLinearMapping����

%% ѵ��DBN-΢������DBN
opts.MaxIter =100;                % ��������
% opts.BatchSize = round(length(trainYn)/4);  % batch��ģΪ�ķ�֮һ��ѵ����trainY�ĳ��Ƚ�����������ȡ��
opts.BatchSize =32;
opts.Verbose = 0;               % �Ƿ�չʾ�м����
opts.StepRatio = 0.1;             % ѧϰ����
opts.Object = 'CrossEntropy';            % Ŀ�꺯��: Square CrossEntropy
%opts.Layer = 1;
dbn = trainDBN(dbn, trainX, trainYn, opts);%dbn����trainDBN����

%% ����
% ��ѵ��������Ԥ��
trainYn_out = v2h( dbn, trainX );%trainYn_out����v2h����
[~,trainY] = max(compet(trainYn'));
[~,trainY_out] = max(compet(trainYn_out'));
%compet��������ľ������ݺ���������ָ��������ÿ�е����ֵ����Ӧ���ֵ���е�ֵΪ1�������е�ֵ��Ϊ0��
%����
% ����׼ȷ��
accTrain = sum(trainY==trainY_out)/length(trainY);%accTrainΪTrainY==trainY_out'���ܺͳ���length(trainY)

% ��ѵ����Ԥ����
figure%ͼ��
plot(trainY,'r o')%��һ����ΪtrainY����ɫ��ԲȦ
hold on%hold on �ǵ�ǰ�ἰͼ�α��ֶ�����ˢ�£�׼�����ܴ˺󽫻���
plot(trainY_out,'g +')%��һ����ΪtrainY_out����ɫ�ļӺ�
legend('��ʵֵ','Ԥ��ֵ')%legend(ͼ��1��ͼ��2��)
grid on%������
xlabel('����','fontsize',13)%xlabel(x��˵��)
ylabel('���','fontsize',13)%ylabel(y��˵��)
title(['ԭʼ���ݵ���100��ѵ����׼ȷ�ʣ�' num2str(accTrain*100) '%'],'fontsize',13)%title(ͼ������)

% �Բ��Լ�����Ԥ��
testYn_out = v2h( dbn, testX );%testYn_outΪ����v2h����
[~,testY] = max(compet(testYn'));
[~,testY_out] = max(compet(testYn_out'));%compet��������ľ������ݺ���������ָ��������ÿ�е����ֵ����Ӧ���ֵ���е�ֵΪ1�������е�ֵ��Ϊ0��

% ����׼ȷ��
accTest = sum(testY==testY_out)/length(testY);%accTestΪtestY==testY_out'���ܺͳ���length(testY)

% �����Լ�Ԥ����
figure%ͼ��
plot(testY,'r o')%��һ����ΪtestY����ɫ��ԲȦ
hold on%hold on �ǵ�ǰ�ἰͼ�α��ֶ�����ˢ�£�׼�����ܴ˺󽫻���
plot(testY_out,'g *')%��һ����ΪtestY_out����ɫ�ļӺ�
legend('��ʵֵ','Ԥ��ֵ')%legend(ͼ��1��ͼ��2��)
grid on%������
xlabel('����','fontsize',13)%xlabel(x��˵��)
ylabel('���','fontsize',13)%ylabel(y��˵��)
title(['ԭʼ���ݵ���100�β��Լ�׼ȷ�ʣ�' num2str(accTest*100) '%'],'fontsize',13)%title(ͼ������)
toc