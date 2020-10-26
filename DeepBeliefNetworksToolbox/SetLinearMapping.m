% SetLinearMapping: set the RBM associated to the linear mapping to the last layer %SetLinearMapping:将与线性映射相关的RBM设置为最后一层。
%
% dbn = SetLinearMapping( dbn, IN, OUT )%将dbn设置为SetLinearMapping(dbn,IN,OUT )
%
%
%Input parameters:%输入参数
% dbn: the Deep Belief Nets (DBN) model：%dbn：深度信念网络模型
% IN: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%IN：可见(输入)变量，其中#的行是数据的数量和#. col是可见(输入)节点的#.
% OUT: teaching data, where # of row is number of data and # of col is # of hidden (output) nodes%OUT:教学数据，其中# of row是数据的数量，而# of col是隐藏(输出)节点的#.
%
%
%Output parameters:%输出数据
% dbn: the set Deep Belief Nets (DBN) model%已建立的深度信念网络模型
%
%
%Example:%举例
% datanum = 1024;%实验数据
% outputnum = 16;%输出数目
% hiddennum = 8;%隐层层数
% inputnum = 4;%输入数目
% 
% inputdata = rand(datanum, inputnum);%输入数据为随机矩阵(datanum,inputnum)
% outputdata = rand(datanum, outputnum);%输出数据为随机矩阵(datanum,outputnum)
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%调用randDBN函数([inputnum,hiddenum,outputnum])
% dbn = pretrainDBN( dbn, inputdata );%调用pretrainDBN函数(dbn，inputdata)
% dbn = SetLinearMapping( dbn, inputdata, outputdata);%调用SetLinearMapping函数(dbn,inputdata,outputdata)
% dbn = trainDBN( dbn, inputdata, outputdata );%调用trainDBN函数(dbn，inputdata，outputdata)
% 
% estimate = v2h( dbn, inputdata );%估计：调用v2h函数(dbn,inputdata)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     版权(C) 2013年Masayuki Tanaka。保留所有权利。          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = SetLinearMapping( dbn, IN, OUT )%建立功能函数SetLinearMapping
nrbm = numel(dbn.rbm);%nrbm为dbn.type的元素个数
if( nrbm > 1 )%如果nrbm大于1
    Hall = v2hall( dbn, IN );%调用v2hall函数
    dbn.rbm{nrbm}.W = linearMapping( Hall{nrbm-1}, OUT );%dbn.rbm{nrbm}.W调用LinearMapping函数
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));%dbn.rbm{nrbm}.b为-0.5乘以一个大小为dbn.rbm{nrbm}.b的全一矩阵
else
    dbn.rbm{nrbm}.W = linearMapping( IN, OUT );%dbn.rbm{nrbm}.W调用linearMapping函数
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));%dbn.rbm{nrbm}.b为-0.5乘以一个大小为dbn.rbm{nrbm}.b的全一矩阵
end
