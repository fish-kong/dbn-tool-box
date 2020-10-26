% sigmoid: calculate sigmoid function%sigmoid为计算sigmoid函数
%
% y = sigmoid(x)%格式y=sigmoid(x)
%
% y = 1.0 ./ ( 1.0 + exp(-x) );%y的表达式
%
%
%Version: 20130727%版本：20130727


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: 深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%        版权(C) 2013年Masayuki Tanaka。保留所有权利。       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(x)%sigmoid函数的调用格式

y = 1.0 ./ ( 1.0 + exp(-x) );%函数y的表达式
