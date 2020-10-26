% v2h: to transform from visible (input) variables to hidden (output) variables%v2h:从显层(输入)变量转换为隐层(输出)变量。
%
% H = h2v(dnn, V)%H调用h2v函数(dnn,V)
%
%
%Output parameters:%输出参数
% H: hidden (output) variables, where # of row is number of data and # ofcol is # of hidden (output) nodes%H：隐层的(输出)变量，其中#的行是数据的数量和#. col是隐层(输出)节点的#。
%
%
%Input parameters:%输入参数
% dnn: the Deep Neural Network model (dbn, rbm)%dnn：深度神经网络模型
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V：显层(输入)变量，第一行是数据的数量，而#是显层(输入)节点的#。
%
%
%Example:%举例
% datanum = 1024;%数据数目
% outputnum = 16;%输出数目
% inputnum = 4;%输入数目
%
% inputdata = rand(datanum, outputnum);%输入数据为随机的datanum行，outputnum列矩阵
%
% dnn = randRBM( inputnum, outputnum );%dnn为调用randRBM函数
% outputdata = v2h( dnn, input );%outputdata为调用v2h函数
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
function H = v2h(dnn, V)%建立v2h函数

ind1 = numel(dnn.type);%ind1为dnn.type中元素的个数
ind0 = ind1-2;%ind0为ind1-2
type = dnn.type(ind0:ind1);%type为dnn.type的(ind0:ind1)

if( isequal(dnn.type, 'BBRBM') )%如果dnn.type与BBRBM的数组容量相同
    H = sigmoid( bsxfun(@plus, V * dnn.W, dnn.b ) );%H为调用sigmoid函数

elseif( isequal(dnn.type, 'GBRBM') )%如果dnn.type与GBRBM的数组容量相同
    v = bsxfun(@rdivide, V, dnn.sig );%bsxfun(@已有定义的函数名， 数组1，数组2)bsxfun(@( 数组1，数组2)函数体表达式，数组1，数组2)例如a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)分别输出为c= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    H = sigmoid( bsxfun(@plus, v * dnn.W, dnn.b ) );%H为调用sigmoid函数   

elseif( isequal(dnn.type, 'BBPRBM') )%如果dnn.type与BBPRBM的数组容量相同
    w2 = dnn.W .* dnn.W;%w2=dnn.W.*dnn.W
    pp = V .* ( 1-V );%pp=V.*(1-V)
    mu = bsxfun(@plus, V * dnn.W, dnn.b );%%bsxfun(@已有定义的函数名， 数组1，数组2)bsxfun(@( 数组1，数组2)函数体表达式，数组1，数组2)例如a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)分别输出为c= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    s2 = pp * w2;%s2为pp*w2
    H = sigmoid( mu ./ sqrt( 1 + s2 * pi / 8 ) );%H为调用sigmoid函数

elseif( isequal(type, 'DBN') )%如果type与DBN的数组容量相同
    nrbm = numel( dnn.rbm );%nrbm为dnn.rbm的元素个数
    H0 = V;%H0为V
    for i=1:nrbm%i的取值范围是1到nrbm
        H1 = v2h( dnn.rbm{i}, H0 );%H1为调用v2h函数
        H0 = H1;%H0赋值H1
    end
    H = H1;%H赋值H1
    
end
