% GetDroppedDBN: get dropped dbn%GetDroppedDBN���õ�һ���½���dbn
%
% [DropedDBN OnInd] = GetDroppedDBN(dbn, DropOutRate,strbm)%GetDroppedDBN�ĵ��ø�ʽ
%
%
%Output parameters:%�������
% DropedDBN: the generated dropped Deep Belief Nets (DBN) model%DropedDBN:���������������(DBN)ģ�͡�
% OnInd: indexes which are used (not dropped) nodes%OnInd:ʹ��(δɾ��)�ڵ�ı�׼��
%
%
%Input parameters:%�������
% dbn: the Original Deep Belief Nets (DBN) model%dbn:��������������(DBN)ģ�͡�
% DropOutRate: 0 < DropOutRate < 1%DropOut��ȡֵ��ΧΪ0��1
% strbm (optional): started rbm layer to dropout (Default: 1)%strbm (optional):�������Ʋ�����������(Ĭ��ֵ:1)
%
%
%Reference:%�ο�
%for details of the dropout%�����ѧ��ϸ��
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton���ˣ�ͨ����ֹ����̽������Эͬ��Ӧ�����������磬2012�ꡣ
%
%
%Version: 20130920%�汾��20131024

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %���������                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%        %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DropedDBN OnInd] = GetDroppedDBN(dbn, DropOutRate, strbm)%GetDroppedDBN�����ĵ��ø�ʽ

if( ~exist('strbm', 'var') || isempty(strbm) )%�������������strbm,����var��������strbmΪ��
	strbm = 1;%strbmΪ1
end

nrbm = numel(dbn.rbm);%nrbmΪdbn.rbm��Ԫ�صĸ���

OnInd = GetOnInd(dbn, DropOutRate, strbm);%OnInd����GetOnInd����

DropedDBN.type = dbn.type;%DropedDBNΪdbn.type
DropedDBN.rbm = cell(nrbm,1);%DropedDBN.rbmΪnrbm��Ԫ�صĸ�������Ϊ1�еĿյĵ�Ԫ����

for n=1:nrbm-1%n��ȡֵ��ΧΪ1��nrbm-1
    DropedDBN.rbm{n}.type = dbn.rbm{n}.type;%DropedDBN.rbm{n}Ϊdbn.rbm{n}.type
    DropedDBN.rbm{n}.W = dbn.rbm{n}.W(OnInd{n},OnInd{n+1});%DropedDBN.rbm{n}.WΪdbn.rbm{n}.W(OnInd{n},OnInd{n+1})
    DropedDBN.rbm{n}.b = dbn.rbm{n}.b(1,OnInd{n+1});%DropedDBN.rbm{n}.bΪdbn.rbm{n}.b(1,OnInd{n+1})
    DropedDBN.rbm{n}.c = dbn.rbm{n}.c(1,OnInd{n});%DropedDBN.rbm{n}.cΪdbn.rbm{n}.c(1,OnInd{n})
    if( isequal(dbn.rbm{n}.type(1:2), 'GB') )%���dbn.rbm{n}.type(1:2)��GB������������ͬ
    	DropedDBN.rbm{n}.sig = dbn.rbm{n}.sig(1,OnInd{n});%DropedDBN.rbm{n}.sig(1,OnInd{n})
    end
end

n = nrbm;%nΪnrbm
DropedDBN.rbm{n}.type = dbn.rbm{n}.type;%DropedDBN.rbm{n}.typeΪdbn.rbm{n}.type
DropedDBN.rbm{n}.W = dbn.rbm{n}.W(OnInd{n},:);%DropedDBN.rbm{n}.WΪdbn.rbm{n}.W(OnInd{n},:)
DropedDBN.rbm{n}.b = dbn.rbm{n}.b;%DropedDBN.rbm{n}.bΪdbn.rbm{n}.b
DropedDBN.rbm{n}.c = dbn.rbm{n}.c(1,OnInd{n});%DropedDBN.rbm{n}.cΪdbn.rbm{n}.c(1,OnInd{n})
if( isequal(dbn.rbm{n}.type(1:2), 'GB') )%���dbn.rbm{n}.type(1:2)��GB������������ͬ
	DropedDBN.rbm{n}.sig = dbn.rbm{n}.sig(1,OnInd{n});%DropedDBN.rbm{n}.sigΪdbn.rbm{n}.sig(1,OnInd{n})
end
