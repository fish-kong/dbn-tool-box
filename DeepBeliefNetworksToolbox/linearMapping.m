% linearMapping: calculate the linear mapping matrix between the input data and the output data%LinearMapping计算输入数据与输出数据之间的线性映射矩阵。
%
% M = linearMapping( IN, OUT )%LinearMapping函数调用格式
%
%
%Output parameters:%输出参数
% M: The linear mapping matrix%M：线性放射矩阵
%
%
%Input parameters:%输入参数
% IN: input data, where # of row is # of data and # of col is # of input
% features%IN:输入数据的行是数据,列是输入特性。
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:输出数据，其中行是数据,列是输出标签。
%
%
%Example:举例
% datanum = 1024;%datanum为1024
% outputnum = 16;%outputnum为16
% inputnum = 4;%inputnum为4
%
% inputdata = rand(datanum, inputnum); %inputdata为随机的datanum行，inputnum列的矩阵
% outputdata = rand(datanum,outputnum);%outputdata为随机的datanum行，outputnum列的矩阵
%
% M = linearMapping(inputdata, outputdata);M为调用LinearMapping函数
%
%
%Version: 20130727%版本：20130727

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %深度神经网络                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     版权(C) 2013年Masayuki Tanaka。保留所有权利。          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M = linearMapping( IN, OUT )%LinearMapping的调用格式
M = pinv(IN) * OUT;%pinv(IN):求IN的伪逆矩阵

%OUT = IN * M;%OUT=IN*M
