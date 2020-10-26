% h2v: to transform from hidden (output) variables to visible (input) variables%h2v:从隐藏(输出)变量转换为可见(输入)变量。
%
% V = h2v(dnn, H)%调用h2v(dnn,H)
%
%
%Output parameters:%输出参数
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V：可见(输入)变量，第一行是数据的数量，而#是可见(输入)节点的#
%
%
%Input parameters:输入参数
% dnn: the Deep Neural Network model (dbn, rbm)%dbn：深度神经网络模型(dbn,rbm)
% H: hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%隐藏(输出)变量，第一行是数据的数量，而# of col是隐藏(输出)节点的#。
%
%
%Example:%举例
% datanum = 1024;%数据数目
% outputnum = 16;%输出数目
% inputnum = 4;%输入数目
%
% outputdata = rand(datanum, outputnum);%输出数据为随机矩阵(datanum,outputnum)
%
% dnn = randRBM( inputnum, outputnum );%调用randRBM函数
% inputdata = h2v( dnn, outputdata );%输入数据调用h2v函数
%
%
%Version: 20130830%版本：20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%   版权(C) 2013年Masayuki Tanaka。保留所有权利。            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function V = h2v(dnn, H)%建立功能函数h2v

ind1 = numel(dnn.type);%ind1为dnn.type数目
ind0 = ind1-2;%ind0为ind1-2
type = dnn.type(ind0:ind1);%type为dnn.type(ind0:ind1)

if( isequal(dnn.type, 'BBRBM') )%如果dnn.type和BBRBM的数组容量相同
  V = sigmoid( bsxfun(@plus, H * dnn.W', dnn.c ) );%V为调用sigmoid函数

elseif( isequal(dnn.type, 'GBRBM') )%如果dnn.type和GBRBM的数组容量相同
    h = bsxfun(@times, H * dnn.W', dnn.sig);%bsxfun(@已有定义的函数名， 数组1，数组2)bsxfun(@( 数组1，数组2)函数体表达式，数组1，数组2)例如a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)分别输出为c= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    V = bsxfun(@plus, h, dnn.c );%bsxfun(@已有定义的函数名， 数组1，数组2)bsxfun(@( 数组1，数组2)函数体表达式，数组1，数组2)例如a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)分别输出为c= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    
elseif( isequal(dnn.type, 'BBPRBM') )%如果dnn.type和BBPRBM的数组容量相同
    w2 = dnn.W .* dnn.W;%w2为dnn.W.*dnn.w
    pp = H .* ( 1-H );%pp为H.*（1-H）
    mu = bsxfun(@plus, H * dnn.W', dnn.c );%bsxfun(@已有定义的函数名， 数组1，数组2)bsxfun(@( 数组1，数组2)函数体表达式，数组1，数组2)例如a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)分别输出为c= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    s2 = pp * w2';%s2为pp*w2'
    V = sigmoid( mu ./ sqrt( 1 + s2 * pi / 8 ) );%V调用sigmoid函数
    
elseif( isequal(type, 'DBN') )%如果type与DBN数组容量相同
    nrbm = numel( dnn.rbm );%nrbm为numel(dnn.rbm)
    V0 = H;%V0为H
    for i=nrbm:-1:1%给i赋值，从nrbm到1，步长为-1
        V1 = h2v( dnn.rbm{i}, V0 );%V1调用h2v函数
        V0 = V1;
    end
    V = V1;%将V1值赋值给V
    
end
