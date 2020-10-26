% CalcErrorRate: calculate error rate%CalcErrorRate:计算错误率
%
% ErrorRate = CalcErrorRate( dbn, IN, OUT )%CalcErrorRate函数调用格式
%
%
%Output parameters:%输出参数
% ErrorRate: error rate%ErrorRate:为错误率
%
%
%Input parameters:%输入参数
% dbn: network%dbn:网络
% IN: input data, where # of row is # of data and # of col is # of input features%IN:输入数据中，行是数据，列是是输入特性。
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:输出数据中，行是数据，列是输出标签。
%Version: 20131213%版本：20131213

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:  %深度神经网络                       %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %版权(C) 2013年Masayuki Tanaka。保留所有权利。          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ErrorRate = CalcErrorRate( dbn, IN, OUT )%函数CalcErrorRate的调用格式
 out = v2h( dbn, IN );%out调用v2h函数
 [m ind] = max(out,[],2);%[m ind]为max(out,[],2)
 out = zeros(size(out));%out为大小为out的全零矩阵
 for i=1:size(out,1)%i的取值范围为1到out行数的大小
  out(i,ind(i))=1;%out的第i行，第ind(i)列为1
 end
 
 ErrorRate = abs(OUT-out);%ErrorRate为OUT-out的绝对值
 ErrorRate = mean(sum(ErrorRate,2)/2);%ErrorRate为ErrorRate所有列之和除以2的平均值

end

