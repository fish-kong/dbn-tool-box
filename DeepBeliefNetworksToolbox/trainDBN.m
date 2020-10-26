% trainDBN: training the Deep Belief Nets (DBN) model by back projection algorithm%trainDBN:ͨ����ͶӰ�㷨ѵ�����������(DBN)ģ�͡�
%
% [dbn rmse] = trainDBN( dbn, IN, OUT, opts)%trainDBN�ĵ��ø�ʽ
%
%
%Output parameters:%�������
% dbn: the trained Deep Belief Nets (DBN) model%dbn����ѵ���������������
% rmse: the rmse between the teaching data and the estimates%��ѧ���������֮��ı�׼���
%
%
%Input parameters:%�������
% dbn: the initial Deep Belief Nets (DBN) model%dbn������������������ģ�͡�
% IN: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%IN���ɼ�(����)�������������ݵ��������ǿɼ�(����)�ڵ�
% OUT: teaching hidden (output) variables, where # of row is number of data and # of col is # of hidden (output) nodes%OUT����ѧ����(���)�����������������ݵ�������������(���)�ڵ�����
% opts (optional): options%ѡ��(��ѡ):ѡ��
% options (defualt value):%ѡ�Ĭ��ֵ��
%  opts.Layer: # of tarining RBMs counted from output layer (all layer)%opts.Layer:����ѵ���е����޲����������������(���в�)
%  opts.MaxIter: Maxium iteration number (100)%opts.MaxIter:�����ķ������(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)%opts.InitialMomentum����ʼ������ֱ����ʼ������(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)%opts.InitialMomentumIter����ʼ�����ĵ�����(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)%opts.FinalMomentum����ʼ����֮������ն���(0.9)
%  opts.WeightCost: Weight cost (0.0002)%opts.WeightCost��Ȩ��(0.0002)
%  opts.DropOutRate: List of Dropout rates for each layer (0)%opts.DropOutRate��ÿһ�����ѧ�ʣ�0��
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio��ѧϰ������0.01��
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize��С��������#(�������ݵ�#)
%  opts.Object: specify the object function ('Square')%opts.Object��ָ��������('ƽ��')
%              'Square' %ƽ��
%              'CrossEntorpy'%������
%  opts.Verbose: verbose or not (false)%opts.Verbose����ϸ�Ĺ������(����)
%
%
%Example:%����
% datanum = 1024;%������Ŀ
% outputnum = 16;%�����Ŀ
% hiddennum = 8;%������Ŀ
% inputnum = 4;%������Ŀ
% 
% inputdata = rand(datanum, inputnum);%��������Ϊһ��datanum�У�inputnum�е��������
% outputdata = rand(datanum, outputnum);%�������Ϊһ��datanum�У�outputnum�е��������
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%����randDBN����
% dbn = pretrainDBN( dbn, inputdata );%dbnΪԤѵ�����ѵ������(dbn,��������)
% dbn = SetLinearMapping( dbn, inputdata, outputdata );%����SetLinearMapping����
% dbn = trainDBN( dbn, inputdata, outputdata );%����trainDBN����
% 
% estimate = v2h( dbn, inputdata );%����v2h����
%
%
%Reference:%�ο�
%for details of the dropout%�����ѧ��ϸ��
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton���ˣ�ͨ����ֹ����̽������Эͬ��Ӧ�����������磬2012�ꡣ
%
%
%Version: 20131024%�汾��20131024


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:   %���������                      %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dbn rmse] = trainDBN( dbn, IN, OUT, opts)%trainDBN�ĵ��ø�ʽ

% Important parameters%��Ҫ�Ĳ���
InitialMomentum = 0.5;     % momentum for first five iterations%ǰ��������Ķ�������ʼ����0.5
FinalMomentum = 0.9;       % momentum for remaining iterations%������ʣ��ĵ����������0.9
WeightCost = 0.0002;       % costs of weight update%Ȩ�ظ��£�Ȩ��Ϊ0.0002
InitialMomentumIter = 5;%��ʼ������������Ϊ5

MaxIter = 1000;%����������
StepRatio = 0.01;%����Ϊ0.01
BatchSize = 0;%������СΪ0
Verbose = false;%VerboseΪ�����

Layer = 0;%����Ϊ��
strbm = 1;%strbmΪ1

nrbm = numel( dbn.rbm );%nrbmΪdbn.rbm��Ԫ�صĸ���
DropOutRate = zeros(nrbm,1);%DropOutRateΪnrbm�У�1�е�ȫ�����

OBJECTSQUARE = 1;%OBJECTSQUAREΪ1
OBJECTCROSSENTROPY = 2;%OBJECTCROSSENTROPYΪ2
Object = OBJECTSQUARE;%ObjectΪOBJECTSQUARE

TestIN = [];%TestINΪ�վ���
TestOUT = [];%TestOUTΪ�վ���
fp = [];%fpΪ�վ���

debug = 0;%debugΪ0

if( exist('opts' ) )%�������opts
 if( isfield(opts,'MaxIter') )%���ṹ��opts�Ƿ������MaxIterָ������ ��������������߼�1; ���opts������MaxIter�����opts���ǽṹ�����͵ģ� �����߼�0��
  MaxIter = opts.MaxIter;%MaxIterΪopts.MaIter
 end
 if( isfield(opts,'InitialMomentum') )%���ṹ��opts�Ƿ������InitialMomentumָ������ ��������������߼�1; ���opts������InitialMomentum�����opts���ǽṹ�����͵ģ� �����߼�0��
  InitialMomentum = opts.InitialMomentum;%InitialMomentumΪopts.InitialMomentum
 end
 if( isfield(opts,'InitialMomentumIter') )%���ṹ��opts�Ƿ������InitialMomentumIterָ������ ��������������߼�1; ���opts������InitialMomentumIter�����opts���ǽṹ�����͵ģ� �����߼�0��
  InitialMomentumIter = opts.InitialMomentumIter;%InitialMomentumIterΪopts.InitialMomentumIter
 end
 if( isfield(opts,'FinalMomentum') )%���ṹ��opts�Ƿ������FinalMomentumָ������ ��������������߼�1; ���opts������FinalMomentum�����opts���ǽṹ�����͵ģ� �����߼�0��
  FinalMomentum = opts.FinalMomentum;%FinalMomentumΪopts.FinalMomentum
 end
 if( isfield(opts,'WeightCost') )%���ṹ��opts�Ƿ������WeightCostָ������ ��������������߼�1; ���opts������WeightCost�����opts���ǽṹ�����͵ģ� �����߼�0��
  WeightCost = opts.WeightCost;%WeightCostΪopts.WeightCost
 end
 if( isfield(opts,'DropOutRate') )%���ṹ��opts�Ƿ������DropOutRateָ������ ��������������߼�1; ���opts������DropOutRate�����opts���ǽṹ�����͵ģ� �����߼�0��
  DropOutRate = opts.DropOutRate;%DropOutRateΪopts.DropOutRate
  if( numel(DropOutRate) == 1 )%���DropOutRate�е�Ԫ�ظ�����Ϊ1
      DropOutRate = ones(nrbm,1) * DropOutRate;%DropOutRateΪLayerNum�У�1�е�ȫһ�������DropOutRate
  end
 end
 if( isfield(opts,'StepRatio') )%���ṹ��opts�Ƿ������StepRatioָ������ ��������������߼�1; ���opts������StepRatio�����opts���ǽṹ�����͵ģ� �����߼�0��
  StepRatio = opts.StepRatio;%StepRatioΪopts.StepRatio
 end
 if( isfield(opts,'BatchSize') )%���ṹ��opts�Ƿ������BatchSizeָ������ ��������������߼�1; ���opts������BatchSize�����opts���ǽṹ�����͵ģ� �����߼�0��
  BatchSize = opts.BatchSize;%BatchSizeΪotps.BatchSize
 end
 if( isfield(opts,'Verbose') )%���ṹ��opts�Ƿ������Verboseָ������ ��������������߼�1; ���opts������Verbose�����opts���ǽṹ�����͵ģ� �����߼�0��
  Verbose = opts.Verbose;%VerboseΪopts.Verbose
 end
 if( isfield(opts,'Layer') )%���ṹ��opts�Ƿ������Layerָ������ ��������������߼�1; ���opts������Layer�����opts���ǽṹ�����͵ģ� �����߼�0��
  Layer = opts.Layer;%LayerΪopts.Layer
 end
 if( isfield(opts,'Object') )%���ṹ��opts�Ƿ������Objectָ������ ��������������߼�1; ���opts������Object�����opts���ǽṹ�����͵ģ� �����߼�0��
  if( strcmpi( opts.Object, 'Square' ) )%���opts.Object��Square�ַ����������
   Object = OBJECTSQUARE;%ObjectΪOBJECTSQUARE
  elseif( strcmpi( opts.Object, 'CrossEntropy' ) )%���opts.Object��CrossEntropy�ַ����������
   Object = OBJECTCROSSENTROPY;%ObjectΪOBJECTCROSSENTROPY
  end
 end
 if( isfield(opts,'TestIN') )%���ṹ��opts�Ƿ������TestINָ������ ��������������߼�1; ���opts������TestIN�����opts���ǽṹ�����͵ģ� �����߼�0��
     TestIN = opts.TestIN;%TestINΪopts.TestIN
 end
 if( isfield(opts,'TestOUT') )%���ṹ��opts�Ƿ������TestOUTָ������ ��������������߼�1; ���opts������TestOUT�����opts���ǽṹ�����͵ģ� �����߼�0��
     TestOUT = opts.TestOUT;%TestOUTΪopts.TestOUT
 end
 if( isfield(opts,'LogFilename') )%���ṹ��opts�Ƿ������LogFilenameָ������ ��������������߼�1; ���opts������LogFilename�����opts���ǽṹ�����͵ģ� �����߼�0��
     fp = fopen( opts.LogFilename, 'w' );% fopen()�Ǹ������ݰ�ָ����ʽ���뵽matlab�еĺ�����w д�루�ļ��������ڣ��Զ�������
 end
 if( isfield(opts,'Debug') )%���ṹ��opts�Ƿ������MaxIterָ������ ��������������߼�1; ���opts������MaxIter�����opts���ǽṹ�����͵ģ� �����߼�0��
     debug = opts.Debug;%debugΪopts.Debug
 end
end

num = size(IN,1);%numΪIN����Ԫ�صĴ�С
if( BatchSize <= 0 )%���BatchSizeС����0
  BatchSize = num;%BatchSizeΪnum
end

if( Layer > 0 )%���Layer������
    strbm = nrbm - Layer + 1;%strbmΪnrbm-Layer+1
end

deltaDbn = dbn;%deltaDbnΪdbn
for n=strbm:nrbm%n��ȡֵ��ΧΪstrbmd��nrbm
    deltaDbn.rbm{n}.W = zeros(size(dbn.rbm{n}.W));%deltaDbn.rbm{n}.WΪ��СΪdbn.rbm{n}.W��ȫ�����
    deltaDbn.rbm{n}.b = zeros(size(dbn.rbm{n}.b));%deltaDbn.rbm{n}.bΪ��СΪdbn.rbm{n}.b��ȫ�����
end

if( Layer > 0 )%���Layer������
    strbm = nrbm - Layer + 1;%strbmΪnrbm-Layer+1
end

if( sum(DropOutRate > 0) )%DropOutRate���ܺʹ���0
    OnInd = GetOnInd( dbn, DropOutRate, strbm );%OnInd����GetOnInd����
    for n=max([2,strbm]):nrbm%n�ķ�ΧΪ[2,strbm]�е����ֵ��strbm
        dbn.rbm{n}.W = dbn.rbm{n}.W / numel(OnInd{n-1}) * size(dbn.rbm{n-1}.W,2);%dbn.rbm{n}.WΪdbn.rbm{n}.W����OnInd{n-1}��Ԫ�صĸ����ڳ���dbn.rbm{n-1}.W�������Ĵ�С
    end
end

if( Verbose )%���verbose��ʾ��ϸ��Ϣ
    timer = tic;%timerΪtic
end

for iter=1:MaxIter%��������Ϊ1������������
    
    % Set momentum%��������
	if( iter <= InitialMomentumIter )%�����������С���ڳ�ʼ��������
		momentum = InitialMomentum;%����Ϊ��ʼ����
	else
		momentum = FinalMomentum;%����Ϊ�����
    end
    
	ind = randperm(num);%P=randperm(N)����һ������N����0��N֮����������Ԫ�ص�����P=randperm(N,K)����һ������K����0��N֮������Ԫ���������磺randperm��6,3������Ϊ[4 2 5]
	for batch=1:BatchSize:num%batch��ȡֵ��ΧΪ1��num������ΪBatchSize		
		bind = ind(batch:min([batch + BatchSize - 1, num]));%���ind��һ��������ô��ind=ind([4,3,2,1])����˼����Ҫȡ���ĸ�Ԫ�ء�˳���Ǵӵ�һ�п�ʼ�����������������ҵڶ�����������һ�º����
        
        if( isequal(dbn.type(3), 'P') )%���rbm.type(3)��P������������ͬ
            
            Hall = v2hall( dbn, IN(bind,:) );%Hall����v2hall����
            for n=nrbm:-1:strbm%n��ȡֵ��Χ��nrbm��strbm,����Ϊ-1
                if( n-1 > 0 )%���n-1����0
                    in = Hall{n-1};%inΪHall{n-1}
                else
                    in = IN(bind,:);%inΪIN(bind,:)
                end
                
                [intDerDel intDerTau] = internal( dbn.rbm{n}, in );%����һ��internal���ܺ���
                derSgm = Hall{n} .* ( 1 - Hall{n} );%derSgmΪHall{n}.*(1-Hall{n});
                if( n+1 > nrbm )%���n+1����nrbm
                    derDel = intDerDel .* ( Hall{nrbm} - OUT(bind,:) );%derDelΪintDerDel.*(Hall{nrbm}-OUT(bind,:))
                    derTau = intDerTau .* ( Hall{nrbm} - OUT(bind,:) );%derTauΪintDerDel.*(Hall{nrbm}-OUT(bind,:))
                    if( Object == OBJECTSQUARE )%���Object�����OBJECTSQUARE
                        derDel = derDel .* derSgm;%derDelΪderDel.*derSgm
                        derTau = derTau .* derSgm;%deTauΪderTau.*derSgm
                    end
                else
                    al = derDel * dbn.rbm{n+1}.W' + derTau * ( dbn.rbm{n+1}.W .* dbn.rbm{n+1}.W )' .* ( 1 - 2 * Hall{n} );%alΪderDel*dbn.rbm{n+1}.W��ת�ü�derTau����dbn.rbm{n+1}.W���dbn.rbm{n+1}.W�Ļ���ת�õ��1��2����Hall{n}
                    
                    derDel = al .* derSgm .* intDerDel;%derDelΪal.*derSgm.*intDerDel
                    derTau = al .* derSgm .* intDerTau;%derTauΪal.*derSgm.*intDerTau
                end
                
                deltaW = ( in' * derDel + 2 * (in .* (1-in))' * derTau .* dbn.rbm{n}.W ) / numel(bind);%deltaWΪin��ת�ó���derDel��2*��in.*(1-in)����ת�ó���derTau.*dbn.rbm{n}.W����bind��Ԫ�صĸ���
                deltab = mean(derDel,1);%deltabΪ����derDel�е�ƽ��ֵ��matlab�е�mean���������������������ƽ�������߾�ֵ��
                
                deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W - StepRatio * deltaW;%deltaDbn.rbm{n}.WΪ����momentum*deltaDbn.rbm{n}.W��������StepRatio����deltaW
                deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.bΪ����momentum*deltaDbn.rbm{n}.b��������StepRatio����deltaW
                
                if( debug )%������ԣ�Debug���Ļ����������Ҫ�ҵ���ȥ�������еĴ���
                    EP = 1E-8;%EPΪ1E-8
                    dif = zeros(size(dbn.rbm{n}.W));%difΪ��С��dbn.rbm{n}.W��ȫ�����
                    for i=1:size(dif,1)%i��ȡֵ��Χ��1��dif���еĴ�С
                        for j=1:size(dif,2)%j��ȡֵ��Χ��1��dif���еĴ�С
                            tDBN = dbn;%tDBNΪdbn
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) - EP;%tDBN.rbm{n}.W(i,j)ΪtDBN.rbm{n}.W(i,j)-EP
                            er0 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er0����ObjectFunc����
                            tDBN = dbn;%tDBNΪdbn
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) + EP;%tDBN.rbm{n}.W(i,j)ΪtDBN.rbm{n}.W(i,j)+EP
                            er1 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er1����ObjectFunc����
                            d = (er1-er0)/(2*EP);%dΪ(er1-er0)/(2*EP)
                            dif(i,j) = abs(d - deltaW(i,j) ) / size(OUT,2);%dif(i,j)Ϊd-deltaW(i,j)�ľ���ֵ����OUT�����Ĵ�С
                        end
                    end
                    fprintf( 'max err %d : %g\n', n, max(dif(:)) );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf��fid,format,variables��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
                end
                
            end
        else
            trainDBN = dbn;%trainDBNΪdbn
            if( DropOutRate > 0 )%���DropOutRate������
                [trainDBN OnInd] = GetDroppedDBN( trainDBN, DropOutRate, strbm );%����GetDroppedDBN����
                Hall = v2hall( trainDBN, IN(bind,OnInd{1}) );%����v2hall����
            else
                Hall = v2hall( trainDBN, IN(bind,:) ); %����v2hall����
            end
            
                   
            for n=nrbm:-1:strbm%n��ȡֵ��ΧΪnrbm��strbm,����Ϊ-1
                derSgm = Hall{n} .* ( 1 - Hall{n} );%derSgmΪHall{n}.*(1-Hall{n})
                if( n+1 > nrbm )%���n+1����nrbm
                    der = ( Hall{nrbm} - OUT(bind,:) );%derΪHall{nrbm}-OUT(bind,:)
                    if( Object == OBJECTSQUARE )%���Object�����OBJECTSQUARE
                        der = derSgm .* der;%derΪderSgm.*der
                    end
                else
                    der = derSgm .* ( der * trainDBN.rbm{n+1}.W' );%derΪderSgm.*(der*trainDBN.rbm{n+1}.W��ת��)
                end

                if( n-1 > 0 )%���n-1����0
                    in = Hall{n-1};%inΪHall{n-1}
                else
                    if( DropOutRate > 0 )%���DropOutRate����0
                        in = IN(bind,OnInd{1});%inΪIN(bind,OnInd{1})
                    else
                        in = IN(bind, :);%inΪIN(bind,:)
                    end
                end

                in = cat(2, ones(numel(bind),1), in);%C = cat(dim, A, B) ��dim������A��B�������顣

                deltaWb = in' * der / numel(bind);%deltaWbΪin��ת��*����bind��Ԫ�صĸ���
                deltab = deltaWb(1,:);%deltabΪdeltaWb(1,:)
                deltaW = deltaWb(2:end,:);%deltaWΪdeltaWb(2:end,:)

                if( strcmpi( dbn.rbm{n}.type, 'GBRBM' ) )%strcmpi�Ƚ������ַ����Ƿ���ȫ��ȣ�������ĸ��Сд;���GBRBM��dbn.rbm{n}.type�ַ������
                    deltaW = bsxfun( @rdivide, deltaW, trainDBN.rbm{n}.sig' );%ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
                end

                deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W;%deltaDbn.rbm{n}.WΪmomentum*deltaDbn.rbm{n}.W
                deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b;%deltaDbn.rbm{n}.bΪmomentum*deltaDbn.rbm{n}.b

                if( DropOutRate > 0 )%���DropOutRate������
                    if( n == nrbm )%���n�����nrbm
                        deltaDbn.rbm{n}.W(OnInd{n},:) = deltaDbn.rbm{n}.W(OnInd{n},:) - StepRatio * deltaW;%deltaDbn.rbm{n}.W(OnInd{n},:)ΪdeltaDbn.rbm{n}.W(OnInd{n},:)-StepRatio*deltaW
                        deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.bΪdeltaDbn.rbm{n}.b-StepRatio*deltab
                    else
                        deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) = deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) - StepRatio * deltaW;%deltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1})ΪdeltaDbn.rbm{n}.W(OnInd{n},OnInd{n+1}) - StepRatio * deltaW;
                        deltaDbn.rbm{n}.b(1,OnInd{n+1}) = deltaDbn.rbm{n}.b(1,OnInd{n+1}) - StepRatio * deltab;
                    end
                else
                    deltaDbn.rbm{n}.W = deltaDbn.rbm{n}.W - StepRatio * deltaW;%deltaDbn.rbm{n}.WΪdeltaDbn.rbm{n}.W-StepRatio*deltaW
                    deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - StepRatio * deltab;%deltaDbn.rbm{n}.bΪdeltaDbn.rbm{n}.b-StepRatio*deltab
                end
                
                if( debug )%������ԣ�Debug���Ļ����������Ҫ�ҵ���ȥ�������еĴ���
                    EP = 1E-8;%EPΪ1E-8
                    dif = zeros(size(trainDBN.rbm{n}.W));%difΪ��СΪtrainDBN.rbm{n}.W��ȫ�����
                    for i=1:size(dif,1)%i�ķ�Χ��1��dif�����Ĵ�С
                        for j=1:size(dif,2)%j�ķ�Χ��1��dif�����Ĵ�С
                            tDBN = trainDBN;%tDBNΪtrainDBN
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) - EP;%tDBN.rbm{n}.W(i,j)ΪtDBN.rbm{n}.W(i,j)-EP
                            er0 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er0����ObjectFunc����
                            tDBN = trainDBN;%tDBNΪtrainDBN
                            tDBN.rbm{n}.W(i,j) = tDBN.rbm{n}.W(i,j) + EP;%tDBN.rbm{n}.W(i,j)ΪtDBN.rbm{n}.W(i,j)+EP
                            er1 = ObjectFunc( tDBN, IN(bind,:), OUT(bind,:), opts );%er1����ObjectFun����
                            d = (er1-er0)/(2*EP);%dΪer1-er0�Ĳ����2*EP
                            dif(i,j) = abs(d - deltaW(i,j) ) / size(OUT,2);%dif(i,f)Ϊd-deltaW��i��j���ľ���ֵ����OUT�������Ĵ�С
                        end
                    end
                    fprintf( 'max err: %g\n', max(dif(:)) );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf��fid,format,variables��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
                end
                
            end
        end
        
        for n=strbm:nrbm  %n��ȡֵ��Χ��strbm��nrbm          
            dbn.rbm{n}.W = dbn.rbm{n}.W + deltaDbn.rbm{n}.W;%dbn.rbm{n}.WΪdbn.rbm{n}.W+deltaDbn.rbm{n}.W
            dbn.rbm{n}.b = dbn.rbm{n}.b + deltaDbn.rbm{n}.b;%dbn.rbm{n}.bΪdbn.rbm{n}.b+deltaDbn.rbm{n}.b 
        end

    end
    
    if( Verbose )%���չʾ����
        tdbn = dbn;%tdbnΪdbn
        if( sum(DropOutRate > 0) )%���DropOutRate���ۺϴ���0
            OnInd = GetOnInd( tdbn, DropOutRate, strbm );%OnIndΪ����GetOnInd����
            for n=max([2,strbm]):nrbm%n��ȡֵ��ΧΪmax([2,strbm])��nrbm
                tdbn.rbm{n}.W = tdbn.rbm{n}.W * numel(OnInd{n-1}) / size(tdbn.rbm{n-1}.W,2);%tdbn.rbm{n}.WΪtdbn.rbm{n}.W*OnInd{n-1}��Ԫ�صĸ�������tdbn.rbm{n-1}�����Ĵ�С
            end
        end
        out = v2h( tdbn, IN );%out����v2h����
        err = power( OUT - out, 2 );%errΪOUT-out��ƽ��
        rmse = sqrt( sum(err(:)) / numel(err) );%rmseΪerr(:)���ܺͳ���err��Ԫ�صĸ������̿���
        msg = sprintf('%3d : %9.4f', iter, rmse );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf��fid,format,variables��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
        
        if( ~isempty( TestIN ) && ~isempty( TestOUT ) )%���TestIN���ǿն���TestOUT���ǿ�
            out = v2h( tdbn, TestIN );%out����v2h����
            err = power( TestOUT - out, 2 );%errΪTestOUT-out��ƽ��
            rmse = sqrt( sum(err(:)) / numel(err) );%rmseΪerr(:)���ܺͳ���err��Ԫ�صĸ������̿���
            msg = [msg,' ',sprintf('%9.4f', rmse )];%msgΪ[msg,' ',sprintf('%9.4f',rmse)]            
        end
        
        totalti = toc(timer);%����������һ�����ʹ�ã�tic��ʾ��ʱ�Ŀ�ʼ��toc��ʾ��ʱ�Ľ���
        aveti = totalti / iter;%avetiΪtotalti����������
        estti = (MaxIter-iter) * aveti;%esttiΪMaxIter��iter�Ĳ����aveti
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');%eststrΪdatestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS')datestr�ǽ����ں�ʱ��ת��Ϊ�ַ�����ʽ������datenum�����������������ַ���ת��Ϊ�������֡�
        
        fprintf( '%s %s\n', msg, eststr );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf�� '%s %s\n', msg, eststr��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
        if( ~isempty( fp ) )%���TestIN���ǿ�
            fprintf( fp, '%s %s\n', msg, eststr );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf��fp, '%s %s\n', msg, eststr��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
        end
    end
end

if( sum(DropOutRate > 0) )%���DropOutRate���ܺʹ���0
    OnInd = GetOnInd( dbn, DropOutRate, strbm );%OnInd����GetOnInd����
    for n=max([2,strbm]):nrbm%n��ȡֵ��ΧΪmax([2,strbm])��nrbm
        dbn.rbm{n}.W = dbn.rbm{n}.W * numel(OnInd{n-1}) / size(dbn.rbm{n-1}.W,2);%dbn.rbm{n}.WΪdbn.rbm{n}.W����OnInd{n-1}��Ԫ�صĸ�������dbn.rbm{n-1}�������Ĵ�С
    end
end

if( ~isempty( fp ) )%���TestIN���ǿ�
    fclose(fp);%fcloseһ����fopen�ɶ�ʹ�á���matlab�д��ļ�Ҫʹ��fopen������������Ҫ���ļ����в���֮�󣬾Ϳ���ʹ��fclose����������ļ����йر�
end

end

function [del tau] = internal(rbm,IN)%����internal����
 w2 = rbm.W .* rbm.W;%w2Ϊrbm,W.*rbm.W
 pp = IN .* ( 1-IN );%ppΪIN.*(1-IN)
 mu = bsxfun(@plus, IN * rbm.W, rbm.b );%ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
 s2 = pp * w2;%s2Ϊpp*w2
 
 tmp = 1 + s2 * (pi / 8);%tmpΪ1+s2*(pi/8)
 del = power( tmp, -1/2);%delΪtep��-1/2�η�
 tau = -(pi/16) * mu .* del ./ tmp;%tauΪ-(pi/16)*mu.*del./tmp
end