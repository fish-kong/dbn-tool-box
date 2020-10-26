% GetOnInd: get indexes which are used (not dropped) nodes%GetOnInd获取使用(未删除)节点的标准。
%
% OnInd = GetOnInd( dbn, DropOutRate, strbm )%OnInd的调用格式
%
%
%Output parameters:%输出参数
% OnInd: indexes which are used (not dropped) nodes%OnInd:使用(未删除)节点的标准。
%
%
%Input parameters:%输入参数
% dbn: the Original Deep Belief Nets (DBN) model%最初的深度信念网络(DBN)模型。
% DropOutRate: 0 < DropOutRate < 1%DropOutRate的取值范围为0到1
% strbm (optional): started rbm layer to dropout (Default: 1)%strbm(optional):起始rbm层的辍学(默认值:1)
%
%
%Reference:%参考
%for details of the dropout%关于辍学的细节
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton等人，通过阻止功能探测器的协同适应，改善神经网络，2012年。
%
%
%Version: 20130821%版本：20130821

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%深度神经网络                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%      %版权(C) 2013年Masayuki Tanaka。保留所有权利。        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OnInd = GetOnInd( dbn, DropOutRate, strbm )%建立OnInd功能函数

if( ~exist('strbm', 'var') || isempty(strbm) )%如果不存在类型strbm,变量var或者类型strbm为空
	strbm = 1;%strbm为1
end

OnInd = cell(numel(dbn.rbm),1);%OnInd为行为dbn.rbm中元素的个数，列为1列的空的单元数组

for n=1:numel(dbn.rbm)%n的取值范围是1到dbn.rbm中元素的个数
    dimV = size(dbn.rbm{n}.W,1);%dimV为dbn.rbm{n}行元素的大小
    if( n >= strbm )%如果n大等于strbm
        OnNum = round(dimV*DropOutRate(n));%OnNum为随机的dimV*DropOutRate维矩阵
        OnInd{n} = sort(randperm(dimV, OnNum));%sort函数功能把数组元素按升序或降序排列 如果A是矩阵，sort(A) 对A按每一列元素按照升序排列。P=randperm(N)返回一个包含N个在0到N之间产生的随机元素的向量P=randperm(N,K)返回一个包含K个在0到N之间的随机元素向量例如：randperm（6,3）可能为[4 2 5]

    else
        OnInd{n} = 1:dimV;%OnInd{n}的范围是1到dimV
    end
end