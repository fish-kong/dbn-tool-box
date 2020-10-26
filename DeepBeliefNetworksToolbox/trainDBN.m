% trainDBN: training the Deep Belief Nets (DBN) model by back projection algorithm%trainDBN:通过反投影算法训练深度信念网(DBN)模型。
%
% [dbn rmse] = trainDBN( dbn, IN, OUT, opts)%trainDBN的调用格式
%
%
%Output parameters:%输出参数
% dbn: the trained Deep Belief Nets (DBN) model%dbn：被训练的深度置信网络
% rmse: the rmse between the teaching data and the estimates%教学数据与估算之间的标准误差
%
%
%Input parameters:%输入参数
% dbn: the initial Deep Belief Nets (DBN) model%dbn：最初的深度置信网络模型。
% IN: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%IN：可见(输入)变量，行是数据的数量列是可见(输入)节点
% OUT: teaching hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%OUT：教学隐藏(输出)变量，其中行是数据的数量列是隐藏(输出)节点数。
% opts (optional): options%选择(可选):选项
% options (defualt value):%选项（默认值）
%  opts.Layer: # of tarining RBMs counted from output layer (all layer)%opts.Layer:计算训练中的受限玻尔兹曼机从输出层(所有层)
%  opts.MaxIter: Maxium iteration number (100)%opts.MaxIter:马克西姆迭代数(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)%opts.InitialMomentum：初始动量，直到初始动量仪(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)%opts.InitialMomentumIter：初始动量的迭代数(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)%opts.FinalMomentum：初始动量之后的最终动量(0.9)
%  opts.WeightCost: Weight cost (0.0002)%opts.WeightCost：权重(0.0002)
%  opts.DropOutRate: List of Dropout rates for each layer (0)%opts.DropOutRate：每一层的退学率（0）
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio：学习步长（0.01）
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize：小批量数据#(所有数据的#)
%  opts.Object: specify the object function ('Square')%opts.Object：指定对象函数('平方')
%              'Square' %平方
%              'CrossEntorpy'%交叉熵
%  opts.Verbose: verbose or not (false)%opts.Verbose：详细的过程与否(错误)
%
%
%Example:%举例
% datanum = 1024;%数据数目
% outputnum = 16;%输出数目
% hiddennum = 8;%隐层数目
% inputnum = 4;%输入数目
% 
% inputdata = rand(datanum, inputnum);%输入数据为一个datanum行，inputnum列的随机矩阵
% outputdata = rand(datanum, outputnum);%输出数据为一个datanum行，outputnum列的随机矩阵
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%调用randDBN函数
% dbn = pretrainDBN( dbn, inputdata );%dbn为预训练深度训练网络(dbn,输入数据)
% dbn = SetLinearMapping( dbn, inputdata, outputdata );%调用SetLinearMapping函数
% dbn = trainDBN( dbn, inputdata, outputdata );%调用trainDBN函数
% 
% estimate = v2h( dbn, inputdata );%调用v2h函数
%
%
%Reference:%参考
%for details of the dropout%关于辍学的细节
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton等人，通过阻止功能探测器的协同适应，改善神经网络，2012年。
%
%
%Version: 20131024%版本：20131024


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:   %深度神经网络                      %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     %版权(C) 2013年Masayuki Tanaka。保留所有权利。         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dbn rmse] = trainDBN( dbn, IN, OUT, opts)%trainDBN的调用格式

% Important parameters%重要的参数
InitialMomentum = 0.5;     % momentum for first five iterations%前五个迭代的动量；初始动量0.5
FinalMomentum = 0.9;       % momentum for remaining iterations%动量对剩余的迭代；最后动量0.9
WeightCost = 0.0002;       % costs of weight update%权重更新；权重为0.0002
InitialMomentumIter = 5;%初始动量迭代次数为5

MaxIter = 1000;%最大迭代次数
StepRatio = 0.01;%步长为0.01
BatchSize = 0;%批量大小为0
Verbose = false;%Verbose为错误的

Layer = 0;%层数为零
strbm = 1;%strbm为1

nrbm = numel( dbn.rbm );%nrbm为dbn.rbm中元素的个数
DropOutRate = zeros(nrbm,1);%DropOutRate为nrbm行，1列的全零矩阵

OBJECTSQUARE = 1;%OBJECTSQUARE为1
OBJECTCROSSENTROPY = 2;%OBJECTCROSSENTROPY为2
Object = OBJECTSQUARE;%Object为OBJECTSQUARE

TestIN = [];%TestIN为空矩阵
TestOUT = [];%TestOUT为空矩阵
fp = [];%fp为空矩阵

debug = 0;%debug为0

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
  if( numel(DropOutRate) == 1 )%如果DropOutRate中的元素个数恒为1
      DropOutRate = ones(nrbm,1) * DropOutRate;%DropOutRate为LayerNum行，1列的全一矩阵乘以DropOutRate
  end
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
 if( isfield(opts,'Layer') )%检查结构体opts是否包含由Layer指定的域， 如果包含，返回逻辑1; 如果opts不包含Layer域或者opts不是结构体类型的， 返回逻辑0。
  Layer = opts.Layer;%Layer为opts.Layer
 end
 if( isfield(opts,'Object') )%检查结构体opts是否包含由Object指定的域， 如果包含，返回逻辑1; 如果opts不包含Object域或者opts不是结构体类型的， 返回逻辑0。
  if( strcmpi( opts.Object, 'Square' ) )%如果opts.Object与Square字符串长度相等
   Object = OBJECTSQUARE;%Object为OBJECTSQUARE
  elseif( strcmpi( opts.Object, 'CrossEntropy' ) )%如果opts.Object与CrossEntropy字符串长度相等
   Object = OBJECTCROSSENTROPY;%Object为OBJECTCROSSENTROPY
  end
 end
 if( isfield(opts,'TestIN') )%检查结构体opts是否包含由TestIN指定的域， 如果包含，返回逻辑1; 如果opts不包含TestIN域或者opts不是结构体类型的， 返回逻辑0。
     TestIN = opts.TestIN;%TestIN为opts.TestIN
 end
 if( isfield(opts,'TestOUT') )%检查结构体opts是否包含由TestOUT指定的域， 如果包含，返回逻辑1; 如果opts不包含TestOUT域或者opts不是结构体类型的， 返回逻辑0。
     TestOUT = opts.TestOUT;%TestOUT为opts.TestOUT
 end
 if( isfield(opts,'LogFilename') )%检查结构体opts是否包含由LogFilename指定的域， 如果包含，返回逻辑1; 如果opts不包含LogFilename域或者opts不是结构体类型的， 返回逻辑0。
     fp = fopen( opts.LogFilename, 'w' );% fopen()是个将数据按指定格式读入到matlab中的函数。w 写入（文件若不存在，自动创建）
 end
 if( isfield(opts,'Debug') )%检查结构体opts是否包含由MaxIter指定的域， 如果包含，返回逻辑1; 如果opts不包含MaxIter域或者opts不是结构体类型的， 返回逻辑0。
     debug = opts.Debug;%debug为opts.Debug
 end
end

num = size(IN,1);%num为IN的行元素的大小
if( BatchSize <= 0 )%如果BatchSize小等于0
  BatchSize = num;%BatchSize为num
end

if( Layer > 0 )%如果Layer大于零
    strbm = nrbm - Layer + 1;%strbm为nrbm-Layer+1
end

deltaDbn = dbn;%deltaDbn为dbn
for n=strbm:nrbm%n的取值范围为strbmd到nrbm
    deltaDbn.rbm{n}.W = zeros(size(dbn.rbm{n}.W));%deltaDbn.rbm{n}.W为大小为dbn.rbm{n}.W的全零矩阵
    deltaDbn.rbm{n}.b = zeros(size(dbn.rbm{n}.b));%deltaDbn.rbm{n}.b为大小为dbn.rbm{n}.b的全零矩阵
end

if( Layer > 0 )%如果Layer大于零
    strbm = nrbm - Layer + 1;%strbm为nrbm-Layer+1
end

if( sum(DropOutRate > 0) )%DropOutRate的总和大于0
    OnInd = GetOnInd( dbn, DropOutRate, strbm );%OnInd调用GetOnInd函数
    for n=max([2,strbm]):nrbm%n的范围为[2,strbm]中的最大值到strbm
        dbn.rbm{n}.W = dbn.rbm{n}.W / numel(OnInd{n-1}) * size(dbn.rbm{n-1}.W,2);%dbn.rbm{n}.W为dbn.rbm{n}.W除以OnInd{n-1}中元素的个数在乘以dbn.rbm{n-1}.W的列数的大小
    end
end

if( Verbose )%如果verbose表示详细信息
    timer = tic;%timer为tic
end

for iter=1:MaxIter%迭代次数为1到最大迭代次数
    
    % Set momentum%建立动量
	if( iter <= InitialMomentumIter )%如果迭代次数小等于初始迭代次数
		momentum = InitialMomentum;%动量为初始动量
	else
		momentum = FinalMomentum;%动量为最后动量
    end
    
	ind = randperm(num);%P=randperm(N)返回一个包含N个在0到N之间产生的随机元素的向量P=randperm(N,K)返回一个包含K个在0到N之间的随机元素向量例如：randperm（6,3）可能为[4 2 5]
	for batch=1:BatchSize:num%batch的取值范围为1到num，步长为BatchSize		
		bind = ind(batch:min([batch + BatchSize - 1, num]));%如果ind是一个矩阵，那么，ind=ind([4,3,2,1])的意思就是要取出四个元素。顺序是从第一列开始往下数，数完了往右第二列往下数，一致后最后。
        
        if( isequal(dbn.type(3), 'P') )%如果rbm.type(3)与P的数组容量相同
            
            Hall = v2hall( dbn, IN(bind,:) );%Hall调用v2hall函数
            for n=nrbm:-1:strbm%n的取值范围是nrbm到strbm,步长为-1
                if( n-1 > 0 )%如果n-1大于0
                    in = Hall{n-1};%in为Hall{n-1}
                else
                    in = IN(bind,:);%in为IN(bind,:)
                end
                
                [intDerDel intDerTau] = internal( dbn.rbm{n}, in );%建立一个internal功能函数
                derSgm = Hall{n} .* ( 1 - Hall{n} );%derSgm为Hall{n}.*(1-Hall{n});
                if( n+1 > nrbm )%如果n+1大于nrbm
                    derDel = intDerDel .* ( Hall{nrbm} - OUT(bind,:) );%derDel为intDerDel.*(Hall{nrbm}-OUT(bind,:))
                    derTau = intDerTau .* ( Hall{nrbm} - OUT(bind,:) );%derTau为intDerDel.*(Hall{nrbm}-OUT(bind,:))
                    if( Object == OBJECTSQUARE )%如果Object恒等于OBJECTSQUARE
                        derDel = derDel .* derSgm;%derDel为derDel.*derSgm
                        derTau = derTau .* derSgm;%deTau为derTau.*derSgm
                    end
                else
                    al = derDel * dbn.rbm{n+1}.W' + derTau * ( dbn.rbm{n+1}.W .* dbn.rbm{n+1}.W )' .* ( 1 - 2 * Hall{n} );%al为derDel*dbn.rbm{n+1}.W的转置加derTau乘以dbn.rbm{n+1}.W点乘dbn.rbm{n+1}.W的积的转置点乘1减2倍的Hall{n}
                    
                    derDel = al .* derSgm .* intDerDel;%derDel为al.*derSgm.*intDerDel
                    derTau = al .* derSgm .* intDerTau;%derTau为al.*derSgm.*intDerTau
                end
                
                deltaW = ( in' * derDel + 2 * (in .* (1-in))' * derTau .* dbn.rbm{n}.W ) / numel(bind);%deltaW为in的转置乘以derDel加2*（in.*(1-in)）的转置乘以derTau.*dbn.rbm{n}.W除以bind中元素的个数
                deltab = mean(derDel,1);%deltab为矩阵derDel列的平均值；matlab中的mean函数函数功能是求数组的平均数或者均值。
                
                deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W - StepRatio * deltaW;%deltaDbn.rbm{n}.W为动量momentum*deltaDbn.rbm{n}.W减步长率StepRatio乘以deltaW
                deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.b为动量momentum*deltaDbn.rbm{n}.b减步长率StepRatio乘以deltaW
                
                if( debug )%程序调试（Debug）的基本任务就是要找到并去除程序中的错误。
                    EP = 1E-8;%EP为1E-8
                    dif = zeros(size(dbn.rbm{n}.W));%dif为大小的dbn.rbm{n}.W的全零矩阵
                    for i=1:size(dif,1)%i的取值范围是1到dif的行的大小
                        for j=1:size(dif,2)%j的取值范围是1到dif的列的大小
                            tDBN = dbn;%tDBN为dbn
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) - EP;%tDBN.rbm{n}.W(i,j)为tDBN.rbm{n}.W(i,j)-EP
                            er0 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er0调用ObjectFunc函数
                            tDBN = dbn;%tDBN为dbn
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) + EP;%tDBN.rbm{n}.W(i,j)为tDBN.rbm{n}.W(i,j)+EP
                            er1 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er1调用ObjectFunc函数
                            d = (er1-er0)/(2*EP);%d为(er1-er0)/(2*EP)
                            dif(i,j) = abs(d - deltaW(i,j) ) / size(OUT,2);%dif(i,j)为d-deltaW(i,j)的绝对值除以OUT列数的大小
                        end
                    end
                    fprintf( 'max err %d : %g\n', n, max(dif(:)) );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（fid,format,variables）,按指定的格式将变量的值输出到屏幕或指定文件
                end
                
            end
        else
            trainDBN = dbn;%trainDBN为dbn
            if( DropOutRate > 0 )%如果DropOutRate大于零
                [trainDBN OnInd] = GetDroppedDBN( trainDBN, DropOutRate, strbm );%调用GetDroppedDBN函数
                Hall = v2hall( trainDBN, IN(bind,OnInd{1}) );%调用v2hall函数
            else
                Hall = v2hall( trainDBN, IN(bind,:) ); %调用v2hall函数
            end
            
                   
            for n=nrbm:-1:strbm%n的取值范围为nrbm到strbm,步长为-1
                derSgm = Hall{n} .* ( 1 - Hall{n} );%derSgm为Hall{n}.*(1-Hall{n})
                if( n+1 > nrbm )%如果n+1大于nrbm
                    der = ( Hall{nrbm} - OUT(bind,:) );%der为Hall{nrbm}-OUT(bind,:)
                    if( Object == OBJECTSQUARE )%如果Object恒等于OBJECTSQUARE
                        der = derSgm .* der;%der为derSgm.*der
                    end
                else
                    der = derSgm .* ( der * trainDBN.rbm{n+1}.W' );%der为derSgm.*(der*trainDBN.rbm{n+1}.W的转置)
                end

                if( n-1 > 0 )%如果n-1大于0
                    in = Hall{n-1};%in为Hall{n-1}
                else
                    if( DropOutRate > 0 )%如果DropOutRate大于0
                        in = IN(bind,OnInd{1});%in为IN(bind,OnInd{1})
                    else
                        in = IN(bind, :);%in为IN(bind,:)
                    end
                end

                in = cat(2, ones(numel(bind),1), in);%C = cat(dim, A, B) 按dim来联结A和B两个数组。

                deltaWb = in' * der / numel(bind);%deltaWb为in的转置*除以bind中元素的个数
                deltab = deltaWb(1,:);%deltab为deltaWb(1,:)
                deltaW = deltaWb(2:end,:);%deltaW为deltaWb(2:end,:)

                if( strcmpi( dbn.rbm{n}.type, 'GBRBM' ) )%strcmpi比较两个字符串是否完全相等，忽略字母大小写;如果GBRBM与dbn.rbm{n}.type字符串相等
                    deltaW = bsxfun( @rdivide, deltaW, trainDBN.rbm{n}.sig' );%使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
                end

                deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W;%deltaDbn.rbm{n}.W为momentum*deltaDbn.rbm{n}.W
                deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b;%deltaDbn.rbm{n}.b为momentum*deltaDbn.rbm{n}.b

                if( DropOutRate > 0 )%如果DropOutRate大于零
                    if( n == nrbm )%如果n恒等于nrbm
                        deltaDbn.rbm{n}.W(OnInd{n},:) = deltaDbn.rbm{n}.W(OnInd{n},:) - StepRatio * deltaW;%deltaDbn.rbm{n}.W(OnInd{n},:)为deltaDbn.rbm{n}.W(OnInd{n},:)-StepRatio*deltaW
                        deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.b为deltaDbn.rbm{n}.b-StepRatio*deltab
                    else
                        deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) = deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) - StepRatio * deltaW;%deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1})为deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) - StepRatio * deltaW;
                        deltaDbn.rbm{n}.b(1,OnInd{n+1}) = deltaDbn.rbm{n}.b(1,OnInd{n+1}) - StepRatio * deltab;
                    end
                else
                    deltaDbn.rbm{n}.W = deltaDbn.rbm{n}.W - StepRatio * deltaW;%deltaDbn.rbm{n}.W为deltaDbn.rbm{n}.W-StepRatio*deltaW
                    deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.b为deltaDbn.rbm{n}.b-StepRatio*deltab
                end
                
                if( debug )%程序调试（Debug）的基本任务就是要找到并去除程序中的错误。
                    EP = 1E-8;%EP为1E-8
                    dif = zeros(size(trainDBN.rbm{n}.W));%dif为大小为trainDBN.rbm{n}.W的全零矩阵
                    for i=1:size(dif,1)%i的范围是1到dif行数的大小
                        for j=1:size(dif,2)%j的范围是1到dif列数的大小
                            tDBN = trainDBN;%tDBN为trainDBN
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) - EP;%tDBN.rbm{n}.W(i,j)为tDBN.rbm{n}.W(i,j)-EP
                            er0 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er0调用ObjectFunc函数
                            tDBN = trainDBN;%tDBN为trainDBN
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) + EP;%tDBN.rbm{n}.W(i,j)为tDBN.rbm{n}.W(i,j)+EP
                            er1 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er1调用ObjectFun函数
                            d = (er1-er0)/(2*EP);%d为er1-er0的差除以2*EP
                            dif(i,j) = abs(d - deltaW(i,j) ) / size(OUT,2);%dif(i,f)为d-deltaW（i，j）的绝对值除以OUT的列数的大小
                        end
                    end
                    fprintf( 'max err: %g\n', max(dif(:)) );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（fid,format,variables）,按指定的格式将变量的值输出到屏幕或指定文件
                end
                
            end
        end
        
        for n=strbm:nrbm  %n的取值范围是strbm到nrbm          
            dbn.rbm{n}.W = dbn.rbm{n}.W + deltaDbn.rbm{n}.W;%dbn.rbm{n}.W为dbn.rbm{n}.W+deltaDbn.rbm{n}.W
            dbn.rbm{n}.b = dbn.rbm{n}.b + deltaDbn.rbm{n}.b;%dbn.rbm{n}.b为dbn.rbm{n}.b+deltaDbn.rbm{n}.b 
        end

    end
    
    if( Verbose )%如果展示过程
        tdbn = dbn;%tdbn为dbn
        if( sum(DropOutRate > 0) )%如果DropOutRate的综合大于0
            OnInd = GetOnInd( tdbn, DropOutRate, strbm );%OnInd为调用GetOnInd函数
            for n=max([2,strbm]):nrbm%n的取值范围为max([2,strbm])到nrbm
                tdbn.rbm{n}.W = tdbn.rbm{n}.W * numel(OnInd{n-1}) / size(tdbn.rbm{n-1}.W,2);%tdbn.rbm{n}.W为tdbn.rbm{n}.W*OnInd{n-1}中元素的个数除以tdbn.rbm{n-1}列数的大小
            end
        end
        out = v2h( tdbn, IN );%out调用v2h函数
        err = power( OUT - out, 2 );%err为OUT-out的平方
        rmse = sqrt( sum(err(:)) / numel(err) );%rmse为err(:)的总和除以err中元素的个数的商开方
        msg = sprintf('%3d : %9.4f', iter, rmse );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（fid,format,variables）,按指定的格式将变量的值输出到屏幕或指定文件
        
        if( ~isempty( TestIN ) && ~isempty( TestOUT ) )%如果TestIN不是空而且TestOUT不是空
            out = v2h( tdbn, TestIN );%out调用v2h函数
            err = power( TestOUT - out, 2 );%err为TestOUT-out的平方
            rmse = sqrt( sum(err(:)) / numel(err) );%rmse为err(:)的总和除以err中元素的个数的商开方
            msg = [msg,' ',sprintf('%9.4f', rmse )];%msg为[msg,' ',sprintf('%9.4f',rmse)]            
        end
        
        totalti = toc(timer);%这两个函数一般配合使用，tic表示计时的开始，toc表示计时的结束
        aveti = totalti / iter;%aveti为totalti除迭代次数
        estti = (MaxIter-iter) * aveti;%estti为MaxIter与iter的差乘以aveti
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');%eststr为datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS')datestr是将日期和时间转换为字符串格式函数。datenum用来将给定的日期字符串转换为日期数字。
        
        fprintf( '%s %s\n', msg, eststr );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（ '%s %s\n', msg, eststr）,按指定的格式将变量的值输出到屏幕或指定文件
        if( ~isempty( fp ) )%如果TestIN不是空
            fprintf( fp, '%s %s\n', msg, eststr );%fprintf函数可以将数据按指定格式写入到文本文件中。fprintf（fp, '%s %s\n', msg, eststr）,按指定的格式将变量的值输出到屏幕或指定文件
        end
    end
end

if( sum(DropOutRate > 0) )%如果DropOutRate的总和大于0
    OnInd = GetOnInd( dbn, DropOutRate, strbm );%OnInd调用GetOnInd函数
    for n=max([2,strbm]):nrbm%n的取值范围为max([2,strbm])到nrbm
        dbn.rbm{n}.W = dbn.rbm{n}.W * numel(OnInd{n-1}) / size(dbn.rbm{n-1}.W,2);%dbn.rbm{n}.W为dbn.rbm{n}.W乘以OnInd{n-1}中元素的个数除以dbn.rbm{n-1}中列数的大小
    end
end

if( ~isempty( fp ) )%如果TestIN不是空
    fclose(fp);%fclose一般与fopen成对使用。在matlab中打开文件要使用fopen函数。当不需要对文件进行操作之后，就可以使用fclose函数对这个文件进行关闭
end

end

function [del tau] = internal(rbm,IN)%建立internal函数
 w2 = rbm.W .* rbm.W;%w2为rbm,W.*rbm.W
 pp = IN .* ( 1-IN );%pp为IN.*(1-IN)
 mu = bsxfun(@plus, IN * rbm.W, rbm.b );%使用函数bsxfun可以避免用循环结构编程。bsxfun调用格式:bsxfun(@已有定义的函数名， 数组1，数组2)
 s2 = pp * w2;%s2为pp*w2
 
 tmp = 1 + s2 * (pi / 8);%tmp为1+s2*(pi/8)
 del = power( tmp, -1/2);%del为tep的-1/2次方
 tau = -(pi/16) * mu .* del ./ tmp;%tau为-(pi/16)*mu.*del./tmp
end