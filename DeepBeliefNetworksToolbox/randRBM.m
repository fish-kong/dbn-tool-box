% randRBM: get randomized restricted boltzmann machine (RBM) model%�õ���������������(DBN)ģ�͡�
%
% rbm = randRBM( dimV, dimH, type )%����õ�һ��ά��Ϊdims,type���͵�����������������ģ��
%
%
%Output parameters:%�������
% dbn: the randomized restricted boltzmann machine (RBM) model%��������Ʋ���������ģ��
%
%Input parameters:%�������
% dimV: number of visible (input) nodes%dimVΪ�Բ�(���������)��Ԫ��Ŀ 
% dimH: number of hidden (output) nodes%dimHΪ����(���������)��Ԫ��Ŀ
% type (optional): (default: 'BBRBM' )%����(��ѡ):(Ĭ��:��BBDBN��)
%                 'BBRBM': the Bernoulli-Bernoulli RBM%BBRBM:��Ŭ�����Ʋ���������
%                 'GBRBM': the Gaussian-Bernoulli RBM%GBRBM:��˹���Ʋ���������
%
%
%Version: 20130830%�汾��20130830


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:  %���������                       %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%       ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rbm = randRBM( dimV, dimH, type )%����һ������Ϊrbm�������һ���Բ���ΪdimV��������ΪdimH,type���͵���������Ʋ���������ģ��

if( ~exist('type', 'var') || isempty(type) )%�������������type,����var��������typeΪ��
	type = 'BBRBM';%typeΪBBRBM
end

if( strcmpi( 'GB', type(1:2) ) )%���GB��type(1:2)�ַ����������
    rbm.type = 'GBRBM';%rbm.typeΪGBRBM
    rbm.W = randn(dimV, dimH) * 0.1;%rbm.WΪһ��dimV�У�dimH�е��������*0.1
    rbm.b = zeros(1, dimH);%rbm.bΪ1�У�dimH�е�ȫ�����
    rbm.c = zeros(1, dimV);%rbm.cΪ1�У�dimV�е�ȫ�����
    rbm.sig = ones(1, dimV);%rbm.sigΪ1�У�dimV��ȫһ����
else%�������
    rbm.type = type;%rbm.typeΪtype����
    rbm.W = randn(dimV, dimH) * 0.1;%rbm.WΪһ��dimV�У�dimH�е��������*0.1
    rbm.b = zeros(1, dimH);%rbm.bΪ1�У�dimH�е�ȫ�����
    rbm.c = zeros(1, dimV);%rbm.cΪ1�У�dimV�е�ȫ�����
end

