% randRBM: get randomized restricted boltzmann machine (RBM) model%得到随机的深度信念网(DBN)模型。
%
% rbm = randRBM( dimV, dimH, type )%随机得到一个维数为dims,type类型的随机的深度信念网络模型
%
%
%Output parameters:%输出参数
% dbn: the randomized restricted boltzmann machine (RBM) model%随机的限制玻尔兹曼机模型
%
%Input parameters:%输入参数
% dimV: number of visible (input) nodes%dimV为显层(或者输入层)单元数目 
% dimH: number of hidden (output) nodes%dimH为隐层(或者输出层)单元数目
% type (optional): (default: 'BBRBM' )%类型(可选):(默认:“BBDBN”)
%                 'BBRBM': the Bernoulli-Bernoulli RBM%BBRBM:伯努利限制玻尔兹曼机
%                 'GBRBM': the Gaussian-Bernoulli RBM%GBRBM:高斯限制玻尔兹曼机
%
%
%Version: 20130830%版本：20130830


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:  %深度神经网络                       %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%       版权(C) 2013年Masayuki Tanaka。保留所有权利。        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rbm = randRBM( dimV, dimH, type )%建立一个，名为rbm的随机的一个显层数为dimV，隐层数为dimH,type类型的随机的限制玻尔兹曼机模型

if( ~exist('type', 'var') || isempty(type) )%如果不存在类型type,变量var或者类型type为空
	type = 'BBRBM';%type为BBRBM
end

if( strcmpi( 'GB', type(1:2) ) )%如果GB与type(1:2)字符串长度相等
    rbm.type = 'GBRBM';%rbm.type为GBRBM
    rbm.W = randn(dimV, dimH) * 0.1;%rbm.W为一个dimV行，dimH列的随机矩阵*0.1
    rbm.b = zeros(1, dimH);%rbm.b为1行，dimH列的全零矩阵
    rbm.c = zeros(1, dimV);%rbm.c为1行，dimV列的全零矩阵
    rbm.sig = ones(1, dimV);%rbm.sig为1行，dimV列全一矩阵
else%其他情况
    rbm.type = type;%rbm.type为type类型
    rbm.W = randn(dimV, dimH) * 0.1;%rbm.W为一个dimV行，dimH列的随机矩阵*0.1
    rbm.b = zeros(1, dimH);%rbm.b为1行，dimH列的全零矩阵
    rbm.c = zeros(1, dimV);%rbm.c为1行，dimV列的全零矩阵
end

