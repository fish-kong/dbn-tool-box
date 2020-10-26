% randDBN: get randomized Deep Belief Nets (DBN) model%�õ���������������(DBN)ģ�͡�
%
% dbn = randDBN( dims, type )%����õ�һ��ά��Ϊdims,type���͵�����������������ģ��
%
%
%Output parameters:%�������
% dbn: the randomized Deep Belief Nets (DBN) model%�����Deep Belief Nets (dbn)ģ��
%
%
%Input parameters:%�������
% dims: number of nodes%ά��Ϊ�ڵ���
% type (optional): (default: 'BBDBN' )%����(��ѡ):(Ĭ��:��BBDBN��)
%                 'BBDBN': all RBMs are the Bernoulli-BernoulliRBMs%���е����޲�����������Ϊ��Ŭ�����޲���������
%                 'GBDBN': the input RBM is the Gaussian-Bernoulli RBM,other RBMs are the Bernoulli-Bernoulli RBMs%�������Ȳ����������Ǹ�˹��Ŭ�����޲������������������޲����������ǲ�Ŭ�����޲�����������
%Version: 20130830%�汾Ϊ20130830

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = randDBN( dims, type )%����һ������Ϊdbn�������һ��ά��Ϊdims,type���͵�����������������ģ��

if( ~exist('type', 'var') || isempty(type) )%�������������type,����var��������typeΪ��
	type = 'BBDBN';%����typeΪBBDBN
end

if( strcmpi( 'GB', type(1:2) ) )%strcmpi�Ƚ������ַ����Ƿ���ȫ��ȣ�������ĸ��Сд;���GB��tpye(1:2)�ַ������
 dbn.type = 'GBDBN';%dbn������ΪGBDBN
 rbmtype = 'GBRBM';%RBM������ΪGBRBM
elseif( strcmpi( 'BBP', type(1:3) ) )%���BBP��type(1:3)�ַ������
 dbn.type = 'BBPDBN';%dbn,typeΪBBPDBN
 rbmtype = 'BBPRBM';%rbmtypeΪBBPRBM
else%������������
 dbn.type = 'BBDBN';%dbn.typeΪBBDBN
 rbmtype = 'BBRBM';%rbmtypeΪBBRBM
end
dbn.rbm = cell( numel(dims)-1, 1 );%dbn.rbmΪ�յ�numel(dims)-1�У�1�пյĵ�Ԫ���飻numel(A)��ʾA��Ԫ�صĸ���
i = 1;%��i=1ʱ
dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );%��Ԫ����dbn.rbm�е�i��Ԫ��ΪrandRBM(dims(i),dims(i+1),rbmtype)
for i=2:numel(dbn.rbm) - 1%��i��ȡֵΪ2��numel(dbn.rbm)ʱ
 dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );%��Ԫ����dbn.rbm�е�i��Ԫ��ΪrandRBM(dims(i),dims(i+1),rbmtype)
end
i = numel(dbn.rbm);%��i=numel(dbn.rbm)ʱ
if( strcmp( 'P', type(3) ) )%���P��type(3)�ַ����������
    dbn.rbm{i} = randRBM( dims(i), dims(i+1), 'BBPRBM' );%��Ԫ����dbn.rbm�е�i��Ԫ��ΪrandRBM(dims(i),dims(i+1),'BBPRBM')
else%�������
    dbn.rbm{i} = randRBM( dims(i), dims(i+1), 'BBRBM' );%��Ԫ����dbn.rbm�е�i��Ԫ��ΪrandRBM(dims(i),dims(i+1),'BBRBM')
end
