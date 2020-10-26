% h2v: to transform from hidden (output) variables to visible (input) variables%h2v:������(���)����ת��Ϊ�ɼ�(����)������
%
% V = h2v(dnn, H)%����h2v(dnn,H)
%
%
%Output parameters:%�������
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%V���ɼ�(����)��������һ�������ݵ���������#�ǿɼ�(����)�ڵ��#
%
%
%Input parameters:�������
% dnn: the Deep Neural Network model (dbn, rbm)%dbn�����������ģ��(dbn,rbm)
% H: hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%����(���)��������һ�������ݵ���������# of col������(���)�ڵ��#��
%
%
%Example:%����
% datanum = 1024;%������Ŀ
% outputnum = 16;%�����Ŀ
% inputnum = 4;%������Ŀ
%
% outputdata = rand(datanum, outputnum);%�������Ϊ�������(datanum,outputnum)
%
% dnn = randRBM( inputnum, outputnum );%����randRBM����
% inputdata = h2v( dnn, outputdata );%�������ݵ���h2v����
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
function V = h2v(dnn, H)%�������ܺ���h2v

ind1 = numel(dnn.type);%ind1Ϊdnn.type��Ŀ
ind0 = ind1-2;%ind0Ϊind1-2
type = dnn.type(ind0:ind1);%typeΪdnn.type(ind0:ind1)

if( isequal(dnn.type, 'BBRBM') )%���dnn.type��BBRBM������������ͬ
  V = sigmoid( bsxfun(@plus, H * dnn.W', dnn.c ) );%VΪ����sigmoid����

elseif( isequal(dnn.type, 'GBRBM') )%���dnn.type��GBRBM������������ͬ
    h = bsxfun(@times, H * dnn.W', dnn.sig);%bsxfun(@���ж���ĺ������� ����1������2)bsxfun(@( ����1������2)��������ʽ������1������2)����a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)�ֱ����Ϊc= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    V = bsxfun(@plus, h, dnn.c );%bsxfun(@���ж���ĺ������� ����1������2)bsxfun(@( ����1������2)��������ʽ������1������2)����a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)�ֱ����Ϊc= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    
elseif( isequal(dnn.type, 'BBPRBM') )%���dnn.type��BBPRBM������������ͬ
    w2 = dnn.W .* dnn.W;%w2Ϊdnn.W.*dnn.w
    pp = H .* ( 1-H );%ppΪH.*��1-H��
    mu = bsxfun(@plus, H * dnn.W', dnn.c );%bsxfun(@���ж���ĺ������� ����1������2)bsxfun(@( ����1������2)��������ʽ������1������2)����a=[1 2 3]; b=[10 ;20 ;30];c=bsxfun(@plus,a,b);d=bsxfun(@(a,b)a.^2+b.^2,a,b)�ֱ����Ϊc= [11 12 13;21 22 23;31 32 33]d= [101 104 109;401 404 409;901 904 909]
    s2 = pp * w2';%s2Ϊpp*w2'
    V = sigmoid( mu ./ sqrt( 1 + s2 * pi / 8 ) );%V����sigmoid����
    
elseif( isequal(type, 'DBN') )%���type��DBN����������ͬ
    nrbm = numel( dnn.rbm );%nrbmΪnumel(dnn.rbm)
    V0 = H;%V0ΪH
    for i=nrbm:-1:1%��i��ֵ����nrbm��1������Ϊ-1
        V1 = h2v( dnn.rbm{i}, V0 );%V1����h2v����
        V0 = V1;
    end
    V = V1;%��V1ֵ��ֵ��V
    
end
