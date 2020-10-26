% pretrainRBM: pre-training the restricted boltzmann machine (RBM) model by Contrastive Divergence Learning %pretrainRBM：预先训练受限的玻尔兹曼机(RBM)模型通过对比差异学习
%
% rbm = pretrainRBM(rbm, V, opts)%pretrainRBM的调用格式
%
%
%Output parameters:%输出参数
% rbm: the restricted boltzmann machine (RBM) model%rbm:限制玻尔兹曼机
%
%
%Input parameters:输入参数
% rbm: the initial boltzmann machine (RBM) model%rbm：最初的玻尔兹曼机(RBM)模型。
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%显层(输入)变量，第一行是数据的数量，显层(输入)节点的数量。
% opts (optional): options%选择(可选):选项
%
% options (defualt value):%选项（默认值）
%  opts.MaxIter: Maxium iteration number (100)% opts.MaxIter:马克西姆迭代数(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)%opts.InitialMomentum:初始动量，直到初始动量仪(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)%opts.InitialMomentumIter:初始动量的迭代数(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)%opts.FinalMomentum:初始动量之后的最终动量(0.9)
%  opts.WeightCost: Weight cost (0.0002) %opts.WeightCost：权重(0.0002)
%  opts.DropOutRate: Dropout rate (0)%opts.DropOutRate：每一层的退学率（0）
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio：学习步长（0.01）
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize：小批量数据#(所有数据的#)
%  opts.Verbose: verbose or not (false)%opts.Verbose：详细的过程与否(错误)
%  opts.SparseQ: q parameter of sparse learning (0)%opts.SparseQ：稀疏学习的q参数(0)
%  opts.SparseLambda: lambda parameter (weight) of sparse learning (0)%opts.SparseLambda:λ参数(重量)的稀疏学习
%
%
%Example:%举例
% datanum = 1024;%数据数目
% outputnum = 16;%输出数目
% inputnum = 4;%输入数目
% 
% inputdata = rand(datanum, inputnum);%输入数据为一个datanum行，inputnum列的随机矩阵
% outputdata = rand(datanum, outputnum);%输出数据为一个datanum行，outputnum列的随机矩阵
% 
% rbm = randRBM(inputnum, outputnum);%调用randDBN函数
% rbm = pretrainRBM( rbm, inputdata );%调用PretrainRBM函数
%
%
%Reference:%参考
%for details of the dropout%关于辍学的细节
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton等人，通过阻止功能探测器的协同适应，改善神经网络，2012年。
%for details of the sparse learning%关于稀疏学习的细节。
% Lee et al, Sparse deep belief net model for visual area V2, NIPS 2008.%Lee等人，稀疏的深层信仰网络模型的视觉区域V2, NIPS 2008。
%for implimentation of contrastive divergence learning%对对比发散学习的影响。
% http://read.pudn.com/downloads103/sourcecode/math/421402/drtoolbox/techniques/train_rbm.m__.htm
%
%
%Version: 20131022%版本：20131022


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %深度神经网路                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%       %版权(C) 2013年Masayuki Tanaka。保留所有权利。       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rbm = pretrainRBM(rbm, V, opts )%pretrianRBM函数的调用格式

% Important parameters%重要的参数
InitialMomentum = 0.5;     % momentum for first five iterations(前五个迭代的动量)初始动量为0.5
FinalMomentum = 0.9;       % momentum for remaining iterations(动量对剩余的迭代)最后动量为0.9
WeightCost = 0.0002;       % costs of weight update(权重更新)权重为0.0002
InitialMomentumIter = 5;%初始动量迭代次数为5

MaxIter = 100;%最大迭代数为100
DropOutRate = 0;%辍学率为0
StepRatio = 0.01;%步长为0.01
BatchSize = 0;%批量大小为0
Verbose = false;%Verbose为错误的

SparseQ = 0;%稀疏参数Q为0
SparseLambda = 0;%稀疏参数Lambda为0


if( exist('opts' ) )%如果存在opts
 if( isfield(opts,'MaxIter') )%检查结构体opts是否包含由MaxIter指定的域， 如果包含，返回逻辑1; 如果opts不包含MaxIter域或者opts不是结构体类型的， 返回逻辑0。
  MaxIter = opts.MaxIter;%MaxIter为opts.MaIter
 end
 if( isfield(opts,'InitialMomentum') )%检查结构体opts是否包含由InitialMomentum指定的域， 如果包含，返回逻辑1; 如果opts不包含InitialMomentum域或者opts不是结构体类型的， 返回逻辑0。
  InitialMomentum = opts.InitialMomentum;%InitialMomentum为opts.InitialMomentum
 end
 if( isfield(opts,'InitialMomentumIter') )%检查结构体opts是否包含由InitialMomentumIter指定的域， 如果包含，返回逻辑1; 如果opts不包含InitialMomentumIter域或者opts不是结构体类型的， 返回逻辑0。
  InitialMomentumIter = opts.InitialMomentumIter;%InitialMomentumIter为opts.InitialMomentumIter
 end
 if( isfield(opts,'FinalMomentum') )%检查结构体opts是否包含由FinalMomentum指定的域， 如果包含，返回逻辑1; 如果opts不包含FinalMomentum域或者opts不是结构体类型的， 返回逻辑0。
  FinalMomentum = opts.FinalMomentum;%FinalMomentum为opts.FinalMomentum
 end
 if( isfield(opts,'WeightCost') )%检查结构体opts是否包含由WeightCost指定的域， 如果包含，返回逻辑1; 如果opts不包含WeightCost域或者opts不是结构体类型的， 返回逻辑0。
  WeightCost = opts.WeightCost;%WeightCost为opts.WeightCost
 end
 if( isfield(opts,'DropOutRate') )%检查结构体opts是否包含由DropOutRate指定的域， 如果包含，返回逻辑1; 如果opts不包含DropOutRate域或者opts不是结构体类型的， 返回逻辑0。
  DropOutRate = opts.DropOutRate;%DropOutRate为opts.DropOutRate
 end
 if( isfield(opts,'StepRatio') )%检查结构体opts是否包含由StepRatio指定的域， 如果包含，返回逻辑1; 如果opts不包含StepRatio域或者opts不是结构体类型的， 返回逻辑0。
  StepRatio = opts.StepRatio;%StepRatio为opts.StepRatio
 end
 if( isfield(opts,'BatchSize') )%检查结构体opts是否包含由BatchSize指定的域， 如果包含，返回逻辑1; 如果opts不包含BatchSize域或者opts不是结构体类型的， 返回逻辑0。
  BatchSize = opts.BatchSize;%BatchSize为otps.BatchSize
 end
 if( isfield(opts,'Verbose') )%检查结构体opts是否包含由Verbose指定的域， 如果包含，返回逻辑1; 如果opts不包含Verbose域或者opts不是结构体类型的， 返回逻辑0。
  Verbose = opts.Verbose;%Verbose为opts.Verbose
 end
 if( isfield(opts,'SparseQ') )%检查结构体opts是否包含由SparseQ指定的域， 如果包含，返回逻辑1; 如果opts不包含SparseQ域或者opts不是结构体类型的， 返回逻辑0。
  SparseQ = opts.SparseQ;%SparseQ为opts.SparseQ
 end
 if( isfield(opts,'SparseLambda') )%检查结构体opts是否包含由SparseLambda指定的域， 如果包含，返回逻辑1; 如果opts不包含SparseLambda域或者opts不是结构体类型的， 返回逻辑0。
  SparseLambda = opts.SparseLambda;%SparseLambda为opts.SparseLambda
 end

else
 opts = [];%opts为空的矩阵
end

num = size(V,1);%num为V的行数
dimH = size(rbm.b, 2);%dimH为rbm.b的列数
dimV = size(rbm.c, 2);%dimV为rbm.c的列数

if( BatchSize <= 0 )%如果BatchSize小等于0
  BatchSize = num;%BatchSize为num
end

if( DropOutRate > 0 )%如果DropOutRate大于0
    DropOutNum = round(dimV * DropOutRate);%DropOutNum为随机的dimV*DropOutRate为矩阵 
    DropOutRate = DropOutNum / num;%DropOutRate为DropOutNum / num
end


deltaW = zeros(dimV, dimH);%deltaW为dimV行，dimH列的全零矩阵
deltaB = zeros(1, dimH);%deltaB为1行，dimH列的全零矩阵
deltaC = zeros(1, dimV);%deltaB为1行，dimV列的全零矩阵

if( Verbose )% verbose表示详细信息
    timer = tic;%计时器为tic
end

for iter=1:MaxIter%迭代次数为从1到最大迭代次数

    
    % Set momentum%建立动量
	if( iter <= InitialMomentumIter )%如果迭代次数小等于初始动量迭代
		momentum = InitialMomentum;%动量为初始动量
	else
		momentum = FinalMomentum;%动量为最后动量
    end

     if( SparseLambda > 0 )%如果稀疏参数Lambda大于0
        dsW = zeros(dimV, dimH);%dsW为dimV行，dimH列的全零矩阵
        dsB = zeros(1, dimH);%dsW为1行，dimH列的全零矩阵

        vis0 = V;%vis0为V
        hid0 = v2h( rbm, vis0 );%hid0调用v2h函数

        dH = hid0 .* ( 1.0 - hid0 );%dH为hid0.*(1.0-hid0)
        sH = sum( hid0, 1 );%sH为hid0中行元素指和
    end

    if( SparseLambda > 0 )%如果稀疏系数Lambda大于0
        mH = sH / num;%mH为sH/num
        sdH = sum( dH, 1 );%sdH为dH的所有行元素之和
        svdH = dH' * vis0;%svdH为dH的转置乘以vis0

        dsW = dsW + SparseLambda * 2.0 * bsxfun(@times, (SparseQ-mH)', svdH)';%dsW为dsW加稀疏系数Lambda乘以2.0乘以bskfun函数；两个非"单一维度"相互匹配的数组a和b做函数fun运算时，bsxfun会隐含扩充a或b使得a和b结构相同，以便实现逐元素运算。使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
        dsB = dsB + SparseLambda * 2.0 * (SparseQ-mH) .* sdH;%dsB为dsB加SpareLambda*2.0*(SparseQ-mH).*sdH
    end


	ind = randperm(num);%ind为随机打乱num范围内序列；randperm是matlab函数，功能是随机打乱一个数字序列。其内的参数决定了随机数的范围。
	for batch=1:BatchSize:num%batch的取值为1到num，步长为BatchSize

		bind = ind(batch:min([batch + BatchSize - 1, num]));%如果ind是一个矩阵，那么，ind=ind([4,3,2,1])的意思就是要取出四个元素。顺序是从第一列开始往下数，数完了往右第二列往下数，一致后最后。

        if( DropOutRate > 0 )%如果DropOutRate大于0
            cMat = zeros(dimV,1);%cMat为dimV行，1列的全零矩阵
            p = randperm(dimV, DropOutNum);%p为随机打乱dimV行，DropOutNum列范围内序列；randperm是matlab函数，功能是随机打乱一个数字序列。其内的参数决定了随机数的范围。
            cMat(p) = 1;%cMat第p个元素为1
            cMat = diag(cMat);%cMat为对角线为cMat元素的矩阵
        end
        
        % Gibbs sampling step 0%吉布斯抽样步长0
        vis0 = double(V(bind,:)); % Set values of visible nodes设置显层节点的值。
        if( DropOutRate > 0 )%如果DropOutRate大于0
            vis0 = vis0 * cMat;%vis0为vis0乘以cMat
        end
        hid0 = v2h( rbm, vis0 );  % Compute hidden nodes%计算隐藏节点%调用v2h函数

        % Gibbs sampling step 1%吉布斯抽样步长1
        if( isequal(rbm.type(3), 'P') )%如果rbm.type(3)与P的数组容量相同
            bhid0 = hid0;%bhid0为hid0
        else
            bhid0 = double( rand(size(hid0)) < hid0 );%bhid0为double类型
        end
        vis1 = h2v( rbm, bhid0 );  % Compute visible nodes%计算隐藏节点%调用h2v函数
        if( DropOutRate > 0 )%如果DropOutRate大于0
            vis1 = vis1 * cMat;%vis1为vis1*cMat
        end
        hid1 = v2h( rbm, vis1 );  % Compute hidden nodes%计算隐藏节点%调用v2h函数

		posprods = hid0' * vis0;%posprod为hid0的转置乘以vis0
		negprods = hid1' * vis1;%negprods为hid1的转置乘以vis1
		% Compute the weights update by contrastive divergence%通过对比散度来计算权重的更新。

        dW = (posprods - negprods)';%dW为（posprods-negprods）的转置
        dB = (sum(hid0, 1) - sum(hid1, 1));%dB为hid0所有行元素之和减去hid1所有行元素之和
        dC = (sum(vis0, 1) - sum(vis1, 1));%dC为vis0所有行元素之和减去vis1所有行元素之和
        
        if( strcmpi( 'GBRBM', rbm.type ) )%strcmpi比较两个字符串是否完全相等，忽略字母大小写;如果GBRBM与rbm.type字符串相等
        	dW = bsxfun(@rdivide, dW, rbm.sig');%使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
        	dC = bsxfun(@rdivide, dC, rbm.sig .* rbm.sig);%使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
        end

		deltaW = momentum * deltaW + (StepRatio / num) * dW;%deltaW为动量乘以deltaW加步长率除以num乘以dW
		deltaB = momentum * deltaB + (StepRatio / num) * dB;%deltaB为动量乘以deltaB加步长率除以num乘以dB
		deltaC = momentum * deltaC + (StepRatio / num) * dC;%deltaC为动量乘以deltaC加步长率除以num乘以dC

         if( SparseLambda > 0 )%如果稀疏系数Lambda大于零
            deltaW = deltaW + numel(bind) / num * dsW;%deltaW为deltaW加bind中的元素的个数除以num乘以dsW
            deltaB = deltaB + numel(bind) / num * dsB;%deltaB为deltaB加bind中的元素的个数除以num乘以dsB
        end

		% Update the network weights%更新网络权重
		rbm.W = rbm.W + deltaW - WeightCost * rbm.W;%rbm.W为rbm.W加deltaw减权重乘以rbm.W
		rbm.b = rbm.b + deltaB;%rbm.b为rbm.b加deltaB
		rbm.c = rbm.c + deltaC;%rbm.c为rbm.c加deltaC

    end

    if( SparseLambda > 0 && strcmpi( 'GBRBM', rbm.type ) )%如果SparseLambda大于0或者GBRBM与rbm.type字符串相等
        dsW = bsxfun(@rdivide, dsW, rbm.sig');%使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
    end

    
	if( Verbose )%如果展示过程
        H = v2h( rbm, V );%H为调用v2h函数
        Vr = h2v( rbm, H );%Vr为h2v函数
		err = power( V - Vr, 2 );%err为V-Vr的平方
		rmse = sqrt( sum(err(:)) / numel(err) );%rmse为开sum(err(:))除以numel(err)的平方
        
        totalti = toc(timer);%tic和toc用来记录matlab命令执行的时间。tic用来保存当前时间，而后使用toc来记录程序完成时间。
        aveti = totalti / iter;%aveti为totalti除以iter
        estti = (MaxIter-iter) * aveti;%estti为最大迭代次数减去迭代次数乘以aveti
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');%datestr是将日期和时间转换为字符串格式函数
        
		fprintf( '%3d : %9.4f %9.4f %9.4f %s\n', iter, rmse, mean(H(:)), aveti, eststr );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（fid,format,variables）,按指定的格式将变量的值输出到屏幕或指定文件
        %d 整数
        %e 实数：科学计算法形式
        %f 实数：小数形式
        %g 由系统自动选取上述两种格式之一
        %s 输出字符串
    end
end

