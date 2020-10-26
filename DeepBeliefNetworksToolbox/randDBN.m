% randDBN: get randomized Deep Belief Nets (DBN) model%得到随机的深度信念网(DBN)模型。
%
% dbn = randDBN( dims, type )%随机得到一个维数为dims,type类型的随机的深度信念网络模型
%
%
%Output parameters:%输出参数
% dbn: the randomized Deep Belief Nets (DBN) model%随机的Deep Belief Nets (dbn)模型
%
%
%Input parameters:%输入参数
% dims: number of nodes%维数为节点数
% type (optional): (default: 'BBDBN' )%类型(可选):(默认:“BBDBN”)
%                 'BBDBN': all RBMs are the Bernoulli-BernoulliRBMs%所有的受限玻尔兹曼机均为伯努利受限波尔兹曼机
%                 'GBDBN': the input RBM is the Gaussian-Bernoulli RBM,other RBMs are the Bernoulli-Bernoulli RBMs%输入首先波尔兹曼机是高斯伯努利受限玻尔兹曼机，其他受限玻尔兹曼机是伯努利受限玻尔兹曼机。
%Version: 20130830%版本为20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    版权(C) 2013年Masayuki Tanaka。保留所有权利。           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = randDBN( dims, type )%建立一个，名为dbn的随机的一个维数为dims,type类型的随机的深度信念网络模型

if( ~exist('type', 'var') || isempty(type) )%如果不存在类型type,变量var或者类型type为空
	type = 'BBDBN';%类型type为BBDBN
end

if( strcmpi( 'GB', type(1:2) ) )%strcmpi比较两个字符串是否完全相等，忽略字母大小写;如果GB与tpye(1:2)字符串相等
 dbn.type = 'GBDBN';%dbn的类型为GBDBN
 rbmtype = 'GBRBM';%RBM的类型为GBRBM
elseif( strcmpi( 'BBP', type(1:3) ) )%如果BBP与type(1:3)字符串相等
 dbn.type = 'BBPDBN';%dbn,type为BBPDBN
 rbmtype = 'BBPRBM';%rbmtype为BBPRBM
else%如果是其他情况
 dbn.type = 'BBDBN';%dbn.type为BBDBN
 rbmtype = 'BBRBM';%rbmtype为BBRBM
end
dbn.rbm = cell( numel(dims)-1, 1 );%dbn.rbm为空的numel(dims)-1行，1列空的单元数组；numel(A)表示A中元素的个数
i = 1;%当i=1时
dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );%单元数组dbn.rbm中第i个元素为randRBM(dims(i),dims(i+1),rbmtype)
for i=2:numel(dbn.rbm) - 1%当i的取值为2到numel(dbn.rbm)时
 dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );%单元数组dbn.rbm中第i个元素为randRBM(dims(i),dims(i+1),rbmtype)
end
i = numel(dbn.rbm);%当i=numel(dbn.rbm)时
if( strcmp( 'P', type(3) ) )%如果P与type(3)字符串长度相等
    dbn.rbm{i} = randRBM( dims(i), dims(i+1), 'BBPRBM' );%单元数组dbn.rbm中第i个元素为randRBM(dims(i),dims(i+1),'BBPRBM')
else%其他情况
    dbn.rbm{i} = randRBM( dims(i), dims(i+1), 'BBRBM' );%单元数组dbn.rbm中第i个元素为randRBM(dims(i),dims(i+1),'BBRBM')
end
