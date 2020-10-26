% v2hall: to transform from visible (input) variables to all hidden(output)
% variables%v2hall�����Բ�(����)����ת��Ϊ��������(���)
%
% Hall = h2vall(dnn, V)%h2vall�ĵ�����ʽ
%
%
%Output parameters:%�������
% Hall: all hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%Hall:���������(���)���������е�һ�������ݵ�����������������(���)�ڵ㡣
%
%
%Input parameters:�������
% dnn: the Deep Neural Network model (dbn, rbm)%dnn�����������
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V���Բ�(����)����������#���������ݵ����������Բ�(����)�ڵ�����
%
%
%Version: 20130830%�汾��20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hall = v2hall(dnn, V)%v2hall�����ĵ��ø�ʽ

ind1 = numel(dnn.type);%ind1Ϊdnn.type��Ԫ�صĸ���
ind0 = ind1-2;%ind0Ϊind1-2
type = dnn.type(ind0:ind1);%typeΪdnn.type(ind0:ind1)

if( isequal(type, 'RBM') )%���type��RBM������������ͬ
    Hall = cell(1,1);%HallΪ�յ�1*1��Ԫ������
    Hall{1} = v2h( dnn, V );%Hall{1}����v2h����

elseif( isequal(type, 'DBN') )%���type��DBN������������ͬ
    nrbm = numel( dnn.rbm );%nrbmΪdnn.rbm��Ԫ�صĸ���
    Hall = cell(nrbm,1);%HallΪnrbm�У�1�еĿյ�Ԫ������
    H0 = V;%H0ΪV
    for i=1:nrbm%i��ȡֵ��Χ1��nrbm
        H1 = v2h( dnn.rbm{i}, H0 );%H1����v2h����
        H0 = H1;%H0��ֵH1
        Hall{i} = H1;%Hall{i}ΪH1
    end
end

