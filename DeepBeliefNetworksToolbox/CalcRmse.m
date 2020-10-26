% CalcRmse: calculate the rmse between predictions and OUTs%CalcRmse:计算预测与输出之间的均方根误差
%
% [rmse AveErrNum] = CalcRmse( dbn, IN, OUT )%CalcRmse函数的调用格式
%
%
%Output parameters:%输出参数
% rmse: the rmse between predictions and OUTs%rmse:预测与输出之间的均方根误差
% AveErrNum: average error number after binarization%AveErrNum:二值化后的平均错误数。
%
%
%Input parameters:%输入参数
% dbn: network%dbn:网络
% IN: input data, where # of row is # of data and # of col is # of input features%IN:输入数据中，行是数据，列是是输入特性。
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:输出数据中，行是数据，列是是输出标签。
%
%
%Version: 20130727%版本：201310727

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%      %版权(C) 2013年Masayuki Tanaka。保留所有权利。        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [rmse AveErrNum] = CalcRmse( dbn, IN, OUT )%CalcRmse函数的调用格式
 out = v2h( dbn, IN );%out调用v2h函数

 err = power( OUT - out, 2 );%err为OUT-out的平方
 rmse = sqrt( sum(err(:)) / numel(err) );%rmse为err(:)的总和除以err中元素的个数

 bout = out > 0.5;%bout为out大于0.5
 BOUT = OUT > 0.5;%BOU为TOUT大于0.5

 err = abs( BOUT - bout );%err为BOUT-bout的绝对值
 AveErrNum = mean( sum(err,2) );%AveErrNum为err所有列之和的平均值
end
