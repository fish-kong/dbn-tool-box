% sigmoid: calculate sigmoid function%sigmoidΪ����sigmoid����
%
% y = sigmoid(x)%��ʽy=sigmoid(x)
%
% y = 1.0 ./ ( 1.0 + exp(-x) );%y�ı��ʽ
%
%
%Version: 20130727%�汾��20130727


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: ���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%        ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = sigmoid(x)%sigmoid�����ĵ��ø�ʽ

y = 1.0 ./ ( 1.0 + exp(-x) );%����y�ı��ʽ
