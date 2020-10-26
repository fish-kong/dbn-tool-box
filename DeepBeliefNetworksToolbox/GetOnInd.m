% GetOnInd: get indexes which are used (not dropped) nodes%GetOnInd��ȡʹ��(δɾ��)�ڵ�ı�׼��
%
% OnInd = GetOnInd( dbn, DropOutRate, strbm )%OnInd�ĵ��ø�ʽ
%
%
%Output parameters:%�������
% OnInd: indexes which are used (not dropped) nodes%OnInd:ʹ��(δɾ��)�ڵ�ı�׼��
%
%
%Input parameters:%�������
% dbn: the Original Deep Belief Nets (DBN) model%����������������(DBN)ģ�͡�
% DropOutRate: 0 < DropOutRate < 1%DropOutRate��ȡֵ��ΧΪ0��1
% strbm (optional): started rbm layer to dropout (Default: 1)%strbm(optional):��ʼrbm����ѧ(Ĭ��ֵ:1)
%
%
%Reference:%�ο�
%for details of the dropout%�����ѧ��ϸ��
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton���ˣ�ͨ����ֹ����̽������Эͬ��Ӧ�����������磬2012�ꡣ
%
%
%Version: 20130821%�汾��20130821

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%      %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OnInd = GetOnInd( dbn, DropOutRate, strbm )%����OnInd���ܺ���

if( ~exist('strbm', 'var') || isempty(strbm) )%�������������strbm,����var��������strbmΪ��
	strbm = 1;%strbmΪ1
end

OnInd = cell(numel(dbn.rbm),1);%OnIndΪ��Ϊdbn.rbm��Ԫ�صĸ�������Ϊ1�еĿյĵ�Ԫ����

for n=1:numel(dbn.rbm)%n��ȡֵ��Χ��1��dbn.rbm��Ԫ�صĸ���
    dimV = size(dbn.rbm{n}.W,1);%dimVΪdbn.rbm{n}��Ԫ�صĴ�С
    if( n >= strbm )%���n�����strbm
        OnNum = round(dimV*DropOutRate(n));%OnNumΪ�����dimV*DropOutRateά����
        OnInd{n} = sort(randperm(dimV, OnNum));%sort�������ܰ�����Ԫ�ذ������������ ���A�Ǿ���sort(A) ��A��ÿһ��Ԫ�ذ����������С�P=randperm(N)����һ������N����0��N֮����������Ԫ�ص�����P=randperm(N,K)����һ������K����0��N֮������Ԫ���������磺randperm��6,3������Ϊ[4 2 5]

    else
        OnInd{n} = 1:dimV;%OnInd{n}�ķ�Χ��1��dimV
    end
end