% v2hall: to transform from visible (input) variables to all hidden(output)
% variables%v2hall：从显层(输入)变量转换为所有隐层(输出)
%
% Hall = h2vall(dnn, V)%h2vall的调用形式
%
%
%Output parameters:%输出参数
% Hall: all hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%Hall:所有隐层的(输出)变量，其中的一行是数据的数量，而列是隐层(输出)节点。
%
%
%Input parameters:输入参数
% dnn: the Deep Neural Network model (dbn, rbm)%dnn：深度神经网络
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V：显层(输入)变量，其中#的行是数据的数量列是显层(输入)节点数量
%
%
%Version: 20130830%版本：20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %版权(C) 2013年Masayuki Tanaka。保留所有权利。          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hall = v2hall(dnn, V)%v2hall函数的调用格式

ind1 = numel(dnn.type);%ind1为dnn.type中元素的个数
ind0 = ind1-2;%ind0为ind1-2
type = dnn.type(ind0:ind1);%type为dnn.type(ind0:ind1)

if( isequal(type, 'RBM') )%如果type与RBM的数组容量相同
    Hall = cell(1,1);%Hall为空的1*1的元包矩阵
    Hall{1} = v2h( dnn, V );%Hall{1}调用v2h函数

elseif( isequal(type, 'DBN') )%如果type与DBN的数组容量相同
    nrbm = numel( dnn.rbm );%nrbm为dnn.rbm中元素的个数
    Hall = cell(nrbm,1);%Hall为nrbm行，1列的空的元包矩阵
    H0 = V;%H0为V
    for i=1:nrbm%i的取值范围1到nrbm
        H1 = v2h( dnn.rbm{i}, H0 );%H1调用v2h函数
        H0 = H1;%H0赋值H1
        Hall{i} = H1;%Hall{i}为H1
    end
end

