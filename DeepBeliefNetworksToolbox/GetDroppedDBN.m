% GetDroppedDBN: get dropped dbn%GetDroppedDBN：得到一个下降的dbn
%
% [DropedDBN OnInd] = GetDroppedDBN(dbn, DropOutRate,strbm)%GetDroppedDBN的调用格式
%
%
%Output parameters:%输出参数
% DropedDBN: the generated dropped Deep Belief Nets (DBN) model%DropedDBN:产生的深度信念网(DBN)模型。
% OnInd: indexes which are used (not dropped) nodes%OnInd:使用(未删除)节点的标准。
%
%
%Input parameters:%输入参数
% dbn: the Original Deep Belief Nets (DBN) model%dbn:最初的深层信仰网(DBN)模型。
% DropOutRate: 0 < DropOutRate < 1%DropOut的取值范围为0到1
% strbm (optional): started rbm layer to dropout (Default: 1)%strbm (optional):启动限制玻尔兹曼机层(默认值:1)
%
%
%Reference:%参考
%for details of the dropout%关于辍学的细节
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton等人，通过阻止功能探测器的协同适应，改善神经网络，2012年。
%
%
%Version: 20130920%版本：20131024

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %深度神经网络                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%        %版权(C) 2013年Masayuki Tanaka。保留所有权利。      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DropedDBN OnInd] = GetDroppedDBN(dbn, DropOutRate, strbm)%GetDroppedDBN函数的调用格式

if( ~exist('strbm', 'var') || isempty(strbm) )%如果不存在类型strbm,变量var或者类型strbm为空
	strbm = 1;%strbm为1
end

nrbm = numel(dbn.rbm);%nrbm为dbn.rbm中元素的个数

OnInd = GetOnInd(dbn, DropOutRate, strbm);%OnInd调用GetOnInd函数

DropedDBN.type = dbn.type;%DropedDBN为dbn.type
DropedDBN.rbm = cell(nrbm,1);%DropedDBN.rbm为nrbm中元素的个数，列为1列的空的单元数组

for n=1:nrbm-1%n的取值范围为1到nrbm-1
    DropedDBN.rbm{n}.type = dbn.rbm{n}.type;%DropedDBN.rbm{n}为dbn.rbm{n}.type
    DropedDBN.rbm{n}.W = dbn.rbm{n}.W(OnInd{n},OnInd{n+1});%DropedDBN.rbm{n}.W为dbn.rbm{n}.W(OnInd{n},OnInd{n+1})
    DropedDBN.rbm{n}.b = dbn.rbm{n}.b(1,OnInd{n+1});%DropedDBN.rbm{n}.b为dbn.rbm{n}.b(1,OnInd{n+1})
    DropedDBN.rbm{n}.c = dbn.rbm{n}.c(1,OnInd{n});%DropedDBN.rbm{n}.c为dbn.rbm{n}.c(1,OnInd{n})
    if( isequal(dbn.rbm{n}.type(1:2), 'GB') )%如果dbn.rbm{n}.type(1:2)与GB的数组容量相同
    	DropedDBN.rbm{n}.sig = dbn.rbm{n}.sig(1,OnInd{n});%DropedDBN.rbm{n}.sig(1,OnInd{n})
    end
end

n = nrbm;%n为nrbm
DropedDBN.rbm{n}.type = dbn.rbm{n}.type;%DropedDBN.rbm{n}.type为dbn.rbm{n}.type
DropedDBN.rbm{n}.W = dbn.rbm{n}.W(OnInd{n},:);%DropedDBN.rbm{n}.W为dbn.rbm{n}.W(OnInd{n},:)
DropedDBN.rbm{n}.b = dbn.rbm{n}.b;%DropedDBN.rbm{n}.b为dbn.rbm{n}.b
DropedDBN.rbm{n}.c = dbn.rbm{n}.c(1,OnInd{n});%DropedDBN.rbm{n}.c为dbn.rbm{n}.c(1,OnInd{n})
if( isequal(dbn.rbm{n}.type(1:2), 'GB') )%如果dbn.rbm{n}.type(1:2)与GB的数组容量相同
	DropedDBN.rbm{n}.sig = dbn.rbm{n}.sig(1,OnInd{n});%DropedDBN.rbm{n}.sig为dbn.rbm{n}.sig(1,OnInd{n})
end
