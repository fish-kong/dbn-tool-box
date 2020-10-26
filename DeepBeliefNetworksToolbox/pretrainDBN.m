% pretrainDBN: pre-training the Deep Belief Nets (DBN) model by ContrastiveDivergence Learning %DBN的预训练：通过对比发散学习预训练深层信念网(DBN)模型。
%
%
% dbn = pretrainDBN(dbn, V, opts)%对dbn进行预训练
%
%
%Output parameters:%输出参数
% dbn: the trained Deep Belief Nets (DBN) model%dbn：被训练的深度信念网络
%
%
%Input parameters:%输入参数
% dbn: the initial Deep Belief Nets (DBN) model%dbn：为最初的深度信念网络模型。
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%可见(输入)变量，第一行是数据的数量，而#是可见(输入)节点的#。
% opts (optional): options%选择(可选):选项
%
% options (defualt value):%选项(默认值)
%  opts.LayerNum: # of tarining RBMs counted from input layer (all layer)%opts.LayerNum：指训练中的受限玻尔兹曼机从输入层(所有层)的层数
%  opts.MaxIter: Maxium iteration number (100)%opts.MaxIter马克西姆迭代数(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter(0.5)%opts.InitialMomentum：初始动量，直到初始动量仪(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum(5)%opts.InitialMomentumIter：初始动量的迭代数(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter(0.9)%opts.FinalMomentum：初始动量之后的最终动量(0.9)
%  opts.WeightCost: Weight cost (0.0002)%opts.WeightCost：权重(0.0002)
%  opts.DropOutRate: List of Dropout rates for each layer(0)%opts.DropOutRate：每一层的退学率（0）
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio：学习步长（0.01）
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize：小批量数据#(所有数据的#)
%  opts.Verbose: verbose or not (false)%opts.Verbose：详细的过程与否(错误)
%  opts.SparseQ: q parameter of sparse learning (0)%opts.SparseQ：稀疏学习的q参数(0)
%  opts.SparseLambda: lambda parameter (weight) of sparse learning(0)%opts.SparseLambda:λ参数(重量)的稀疏学习
%
%
%Example:%举例
% datanum = 1024;%数据数目
% outputnum = 16;%输出数目
% hiddennum = 8;%隐层数目
% inputnum = 4;%输入数目
% inputdata = rand(datanum, inputnum);%输入数据为一个datanum行，inputnum列的随机矩阵
% outputdata = rand(datanum, outputnum);%输出矩阵为一个datanum行，outputnum列的随机矩阵
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%dbn为随机深度信念网络[输入数目，隐层数目，输出数目]
% dbn = pretrainDBN( dbn, inputdata );dbn为预训练深度训练网络(dbn,输入数据)
% dbn = SetLinearMapping( dbn, inputdata, outputdata );%调用SetLinearMapping函数
% dbn = trainDBN( dbn, inputdata, outputdata );%调用trainDBN函数
% 
% estimate = v2h( dbn, inputdata );%调用v2h函数
%
%
%Reference:%参考
%for details of the dropout%关于辍学的细节
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton等人，通过阻止功能探测器的协同适应，改善神经网络，2012年。
%for details of the sparse learning%关于稀疏学习的细节。
% Lee et al, Sparse deep belief net model for visual area V2, NIPS 2008.%Lee等人，稀疏的深层信仰网络模型的视觉区域V2, NIPS 2008。
%
%
%Version: 20130821%版本：20130821


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:   %深度神经网络                      %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %版权(C) 2013年Masayuki Tanaka。保留所有权利。          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = pretrainDBN(dbn, V, opts)%pretrainDBN的调用格式

LayerNum = numel( dbn.rbm );%LayerNum为dbn.rbm中元素的个数
DropOutRate = zeros(LayerNum,1);%DropOutRate为

X = V;%X为V

if( exist('opts' ) )%如果存在opts
 if( isfield(opts,'LayerNum') )%检查结构体opts是否包含由LayerNum指定的域， 如果包含，返回逻辑1; 如果opts不包含LayerNum域或者opts不是结构体类型的， 返回逻辑0。
  LayerNum = opts.LayerNum;%LayerNum为opts.LayerNum
 end
 if( isfield(opts,'DropOutRate') )%检查结构体opts是否包含由DropOutRate指定的域， 如果包含，返回逻辑1; 如果opts不包含DropOutRate域或者opts不是结构体类型的， 返回逻辑0。
  DropOutRate = opts.DropOutRate;%DropOutRate为opts.DropOutRate
  if( numel( DropOutRate ) == 1 )%如果DropOutRate中的元素个数恒为1
   DropOutRate = ones(LayerNum,1) * DropOutRate;%DropOutRate为LayerNum行，1列的全一矩阵乘以DropOutRate
  end
 end
 
else
 opts = [];%opts为空矩阵
end

for i=1:LayerNum%i的取值范围为从1到LayerNum
	opts.DropOutRate = DropOutRate(i);%opts.DropOutRate为DropOutRate(i)
    dbn.rbm{i} = pretrainRBM(dbn.rbm{i}, X, opts);%dbn.rbm{1}调用pretrainRBM函数
    X0 = X;%X0为X
    X = v2h( dbn.rbm{i}, X0 );%X为调用v2h函数
end
