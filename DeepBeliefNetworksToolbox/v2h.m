% v2h: to transform from visible (input) variables to hidden (output) variables%v2h:���Բ�(����)����ת��Ϊ����(���)������
%
% H = h2v(dnn, V)%H����h2v����(dnn,V)
%
%
%Output parameters:%�������
% H: hidden (output) variables, where # of row is number of data and # ofcol is # of hidden (output) nodes%H�������(���)����������#���������ݵ�������#. col������(���)�ڵ��#��
%
%
%Input parameters:%�������
% dnn: the Deep Neural Network model (dbn, rbm)%dnn�����������ģ��
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V���Բ�(����)��������һ�������ݵ���������#���Բ�(����)�ڵ��#��
%
%
%Example:%����
% datanum = 1024;%������Ŀ
% outputnum = 16;%�����Ŀ
% inputnum = 4;%������Ŀ
%
% inputdata = rand(datanum, outputnum);%��������Ϊ�����datanum�У�outputnum�о���
%
% dnn = randRBM( inputnum, outputnum );%dnnΪ����randRBM����
% outputdata = v2h( dnn, input );%outputdataΪ����v2h����
%
%
%Version: 20130830%�汾��20130830


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%   ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function H = v2h(dnn, V)%����v2h����

ind1 = numel(dnn.type);%ind1Ϊdnn.type��Ԫ�صĸ���
ind0 = ind1-2;%ind0Ϊind1-2
type = dnn.type(ind0:ind1);%typeΪdnn.type��(ind0:ind1)

if( isequal(dnn.type, 'BBRBM') )%���dnn.type��BBRBM������������ͬ
    H = sigmoid( bsxfun(@plus, V * dnn.W, dnn.b ) );%HΪ����sigmoid����

elseif( isequal(dnn.type, 'GBRBM') )%���dnn.type��GBRBM������������ͬ
    v = bsxfun(@rdivide, V, dnn.sig );%bsxfun(@���ж���ĺ������� ����1������2)bsxfun(@( ����1������2)��������ʽ������1������2)����a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)�ֱ����Ϊc= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    H = sigmoid( bsxfun(@plus, v * dnn.W, dnn.b ) );%HΪ����sigmoid����   

elseif( isequal(dnn.type, 'BBPRBM') )%���dnn.type��BBPRBM������������ͬ
    w2 = dnn.W .* dnn.W;%w2=dnn.W.*dnn.W
    pp = V .* ( 1-V );%pp=V.*(1-V)
    mu = bsxfun(@plus, V * dnn.W, dnn.b );%%bsxfun(@���ж���ĺ������� ����1������2)bsxfun(@( ����1������2)��������ʽ������1������2)����a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)�ֱ����Ϊc= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    s2 = pp * w2;%s2Ϊpp*w2
    H = sigmoid( mu ./ sqrt( 1 + s2 * pi / 8 ) );%HΪ����sigmoid����

elseif( isequal(type, 'DBN') )%���type��DBN������������ͬ
    nrbm = numel( dnn.rbm );%nrbmΪdnn.rbm��Ԫ�ظ���
    H0 = V;%H0ΪV
    for i=1:nrbm%i��ȡֵ��Χ��1��nrbm
        H1 = v2h( dnn.rbm{i}, H0 );%H1Ϊ����v2h����
        H0 = H1;%H0��ֵH1
    end
    H = H1;%H��ֵH1
    
end
