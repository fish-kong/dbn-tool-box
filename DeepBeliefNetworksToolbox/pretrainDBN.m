% pretrainDBN: pre-training the Deep Belief Nets (DBN) model by ContrastiveDivergence Learning %DBN��Ԥѵ����ͨ���Աȷ�ɢѧϰԤѵ�����������(DBN)ģ�͡�
%
%
% dbn = pretrainDBN(dbn, V, opts)%��dbn����Ԥѵ��
%
%
%Output parameters:%�������
% dbn: the trained Deep Belief Nets (DBN) model%dbn����ѵ���������������
%
%
%Input parameters:%�������
% dbn: the initial Deep Belief Nets (DBN) model%dbn��Ϊ����������������ģ�͡�
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%�ɼ�(����)��������һ�������ݵ���������#�ǿɼ�(����)�ڵ��#��
% opts (optional): options%ѡ��(��ѡ):ѡ��
%
% options (defualt value):%ѡ��(Ĭ��ֵ)
%  opts.LayerNum: # of tarining RBMs counted from input layer (all layer)%opts.LayerNum��ָѵ���е����޲����������������(���в�)�Ĳ���
%  opts.MaxIter: Maxium iteration number (100)%opts.MaxIter�����ķ������(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter(0.5)%opts.InitialMomentum����ʼ������ֱ����ʼ������(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum(5)%opts.InitialMomentumIter����ʼ�����ĵ�����(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter(0.9)%opts.FinalMomentum����ʼ����֮������ն���(0.9)
%  opts.WeightCost: Weight cost (0.0002)%opts.WeightCost��Ȩ��(0.0002)
%  opts.DropOutRate: List of Dropout rates for each layer(0)%opts.DropOutRate��ÿһ�����ѧ�ʣ�0��
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio��ѧϰ������0.01��
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize��С��������#(�������ݵ�#)
%  opts.Verbose: verbose or not (false)%opts.Verbose����ϸ�Ĺ������(����)
%  opts.SparseQ: q parameter of sparse learning (0)%opts.SparseQ��ϡ��ѧϰ��q����(0)
%  opts.SparseLambda: lambda parameter (weight) of sparse learning(0)%opts.SparseLambda:�˲���(����)��ϡ��ѧϰ
%
%
%Example:%����
% datanum = 1024;%������Ŀ
% outputnum = 16;%�����Ŀ
% hiddennum = 8;%������Ŀ
% inputnum = 4;%������Ŀ
% inputdata = rand(datanum, inputnum);%��������Ϊһ��datanum�У�inputnum�е��������
% outputdata = rand(datanum, outputnum);%�������Ϊһ��datanum�У�outputnum�е��������
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%dbnΪ��������������[������Ŀ��������Ŀ�������Ŀ]
% dbn = pretrainDBN( dbn, inputdata );dbnΪԤѵ�����ѵ������(dbn,��������)
% dbn = SetLinearMapping( dbn, inputdata, outputdata );%����SetLinearMapping����
% dbn = trainDBN( dbn, inputdata, outputdata );%����trainDBN����
% 
% estimate = v2h( dbn, inputdata );%����v2h����
%
%
%Reference:%�ο�
%for details of the dropout%�����ѧ��ϸ��
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton���ˣ�ͨ����ֹ����̽������Эͬ��Ӧ�����������磬2012�ꡣ
%for details of the sparse learning%����ϡ��ѧϰ��ϸ�ڡ�
% Lee et al, Sparse deep belief net model for visual area V2, NIPS 2008.%Lee���ˣ�ϡ��������������ģ�͵��Ӿ�����V2, NIPS 2008��
%
%
%Version: 20130821%�汾��20130821


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:   %���������                      %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = pretrainDBN(dbn, V, opts)%pretrainDBN�ĵ��ø�ʽ

LayerNum = numel( dbn.rbm );%LayerNumΪdbn.rbm��Ԫ�صĸ���
DropOutRate = zeros(LayerNum,1);%DropOutRateΪ

X = V;%XΪV

if( exist('opts' ) )%�������opts
 if( isfield(opts,'LayerNum') )%���ṹ��opts�Ƿ������LayerNumָ������ ��������������߼�1; ���opts������LayerNum�����opts���ǽṹ�����͵ģ� �����߼�0��
  LayerNum = opts.LayerNum;%LayerNumΪopts.LayerNum
 end
 if( isfield(opts,'DropOutRate') )%���ṹ��opts�Ƿ������DropOutRateָ������ ��������������߼�1; ���opts������DropOutRate�����opts���ǽṹ�����͵ģ� �����߼�0��
  DropOutRate = opts.DropOutRate;%DropOutRateΪopts.DropOutRate
  if( numel( DropOutRate ) == 1 )%���DropOutRate�е�Ԫ�ظ�����Ϊ1
   DropOutRate = ones(LayerNum,1) * DropOutRate;%DropOutRateΪLayerNum�У�1�е�ȫһ�������DropOutRate
  end
 end
 
else
 opts = [];%optsΪ�վ���
end

for i=1:LayerNum%i��ȡֵ��ΧΪ��1��LayerNum
	opts.DropOutRate = DropOutRate(i);%opts.DropOutRateΪDropOutRate(i)
    dbn.rbm{i} = pretrainRBM(dbn.rbm{i}, X, opts);%dbn.rbm{1}����pretrainRBM����
    X0 = X;%X0ΪX
    X = v2h( dbn.rbm{i}, X0 );%XΪ����v2h����
end
