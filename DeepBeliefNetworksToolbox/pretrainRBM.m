% pretrainRBM: pre-training the restricted boltzmann machine (RBM) model by Contrastive Divergence Learning %pretrainRBM��Ԥ��ѵ�����޵Ĳ���������(RBM)ģ��ͨ���ԱȲ���ѧϰ
%
% rbm = pretrainRBM(rbm, V, opts)%pretrainRBM�ĵ��ø�ʽ
%
%
%Output parameters:%�������
% rbm: the restricted boltzmann machine (RBM) model%rbm:���Ʋ���������
%
%
%Input parameters:�������
% rbm: the initial boltzmann machine (RBM) model%rbm������Ĳ���������(RBM)ģ�͡�
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%�Բ�(����)��������һ�������ݵ��������Բ�(����)�ڵ��������
% opts (optional): options%ѡ��(��ѡ):ѡ��
%
% options (defualt value):%ѡ�Ĭ��ֵ��
%  opts.MaxIter: Maxium iteration number (100)% opts.MaxIter:�����ķ������(100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)%opts.InitialMomentum:��ʼ������ֱ����ʼ������(0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)%opts.InitialMomentumIter:��ʼ�����ĵ�����(5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)%opts.FinalMomentum:��ʼ����֮������ն���(0.9)
%  opts.WeightCost: Weight cost (0.0002) %opts.WeightCost��Ȩ��(0.0002)
%  opts.DropOutRate: Dropout rate (0)%opts.DropOutRate��ÿһ�����ѧ�ʣ�0��
%  opts.StepRatio: Learning step size (0.01)%opts.StepRatio��ѧϰ������0.01��
%  opts.BatchSize: # of mini-batch data (# of all data)%opts.BatchSize��С��������#(�������ݵ�#)
%  opts.Verbose: verbose or not (false)%opts.Verbose����ϸ�Ĺ������(����)
%  opts.SparseQ: q parameter of sparse learning (0)%opts.SparseQ��ϡ��ѧϰ��q����(0)
%  opts.SparseLambda: lambda parameter (weight) of sparse learning (0)%opts.SparseLambda:�˲���(����)��ϡ��ѧϰ
%
%
%Example:%����
% datanum = 1024;%������Ŀ
% outputnum = 16;%�����Ŀ
% inputnum = 4;%������Ŀ
% 
% inputdata = rand(datanum, inputnum);%��������Ϊһ��datanum�У�inputnum�е��������
% outputdata = rand(datanum, outputnum);%�������Ϊһ��datanum�У�outputnum�е��������
% 
% rbm = randRBM(inputnum, outputnum);%����randDBN����
% rbm = pretrainRBM( rbm, inputdata );%����PretrainRBM����
%
%
%Reference:%�ο�
%for details of the dropout%�����ѧ��ϸ��
% Hinton et al, Improving neural networks by preventing co-adaptation of feature detectors, 2012.%Hinton���ˣ�ͨ����ֹ����̽������Эͬ��Ӧ�����������磬2012�ꡣ
%for details of the sparse learning%����ϡ��ѧϰ��ϸ�ڡ�
% Lee et al, Sparse deep belief net model for visual area V2, NIPS 2008.%Lee���ˣ�ϡ��������������ģ�͵��Ӿ�����V2, NIPS 2008��
%for implimentation of contrastive divergence learning%�ԶԱȷ�ɢѧϰ��Ӱ�졣
% http://read.pudn.com/downloads103/sourcecode/math/421402/drtoolbox/techniques/train_rbm.m__.htm
%
%
%Version: 20131022%�汾��20131022


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %�������·                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%       %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rbm = pretrainRBM(rbm, V, opts )%pretrianRBM�����ĵ��ø�ʽ

% Important parameters%��Ҫ�Ĳ���
InitialMomentum = 0.5;     % momentum for first five iterations(ǰ��������Ķ���)��ʼ����Ϊ0.5
FinalMomentum = 0.9;       % momentum for remaining iterations(������ʣ��ĵ���)�����Ϊ0.9
WeightCost = 0.0002;       % costs of weight update(Ȩ�ظ���)Ȩ��Ϊ0.0002
InitialMomentumIter = 5;%��ʼ������������Ϊ5

MaxIter = 100;%��������Ϊ100
DropOutRate = 0;%�ѧ��Ϊ0
StepRatio = 0.01;%����Ϊ0.01
BatchSize = 0;%������СΪ0
Verbose = false;%VerboseΪ�����

SparseQ = 0;%ϡ�����QΪ0
SparseLambda = 0;%ϡ�����LambdaΪ0


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
 if( isfield(opts,'SparseQ') )%���ṹ��opts�Ƿ������SparseQָ������ ��������������߼�1; ���opts������SparseQ�����opts���ǽṹ�����͵ģ� �����߼�0��
  SparseQ = opts.SparseQ;%SparseQΪopts.SparseQ
 end
 if( isfield(opts,'SparseLambda') )%���ṹ��opts�Ƿ������SparseLambdaָ������ ��������������߼�1; ���opts������SparseLambda�����opts���ǽṹ�����͵ģ� �����߼�0��
  SparseLambda = opts.SparseLambda;%SparseLambdaΪopts.SparseLambda
 end

else
 opts = [];%optsΪ�յľ���
end

num = size(V,1);%numΪV������
dimH = size(rbm.b, 2);%dimHΪrbm.b������
dimV = size(rbm.c, 2);%dimVΪrbm.c������

if( BatchSize <= 0 )%���BatchSizeС����0
  BatchSize = num;%BatchSizeΪnum
end

if( DropOutRate > 0 )%���DropOutRate����0
    DropOutNum = round(dimV * DropOutRate);%DropOutNumΪ�����dimV*DropOutRateΪ���� 
    DropOutRate = DropOutNum / num;%DropOutRateΪDropOutNum / num
end


deltaW = zeros(dimV, dimH);%deltaWΪdimV�У�dimH�е�ȫ�����
deltaB = zeros(1, dimH);%deltaBΪ1�У�dimH�е�ȫ�����
deltaC = zeros(1, dimV);%deltaBΪ1�У�dimV�е�ȫ�����

if( Verbose )% verbose��ʾ��ϸ��Ϣ
    timer = tic;%��ʱ��Ϊtic
end

for iter=1:MaxIter%��������Ϊ��1������������

    
    % Set momentum%��������
	if( iter <= InitialMomentumIter )%�����������С���ڳ�ʼ��������
		momentum = InitialMomentum;%����Ϊ��ʼ����
	else
		momentum = FinalMomentum;%����Ϊ�����
    end

     if( SparseLambda > 0 )%���ϡ�����Lambda����0
        dsW = zeros(dimV, dimH);%dsWΪdimV�У�dimH�е�ȫ�����
        dsB = zeros(1, dimH);%dsWΪ1�У�dimH�е�ȫ�����

        vis0 = V;%vis0ΪV
        hid0 = v2h( rbm, vis0 );%hid0����v2h����

        dH = hid0 .* ( 1.0 - hid0 );%dHΪhid0.*(1.0-hid0)
        sH = sum( hid0, 1 );%sHΪhid0����Ԫ��ָ��
    end

    if( SparseLambda > 0 )%���ϡ��ϵ��Lambda����0
        mH = sH / num;%mHΪsH/num
        sdH = sum( dH, 1 );%sdHΪdH��������Ԫ��֮��
        svdH = dH' * vis0;%svdHΪdH��ת�ó���vis0

        dsW = dsW + SparseLambda * 2.0 * bsxfun(@times, (SparseQ-mH)', svdH)';%dsWΪdsW��ϡ��ϵ��Lambda����2.0����bskfun������������"��һά��"�໥ƥ�������a��b������fun����ʱ��bsxfun����������a��bʹ��a��b�ṹ��ͬ���Ա�ʵ����Ԫ�����㡣ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
        dsB = dsB + SparseLambda * 2.0 * (SparseQ-mH) .* sdH;%dsBΪdsB��SpareLambda*2.0*(SparseQ-mH).*sdH
    end


	ind = randperm(num);%indΪ�������num��Χ�����У�randperm��matlab�������������������һ���������С����ڵĲ���������������ķ�Χ��
	for batch=1:BatchSize:num%batch��ȡֵΪ1��num������ΪBatchSize

		bind = ind(batch:min([batch + BatchSize - 1, num]));%���ind��һ��������ô��ind=ind([4,3,2,1])����˼����Ҫȡ���ĸ�Ԫ�ء�˳���Ǵӵ�һ�п�ʼ�����������������ҵڶ�����������һ�º����

        if( DropOutRate > 0 )%���DropOutRate����0
            cMat = zeros(dimV,1);%cMatΪdimV�У�1�е�ȫ�����
            p = randperm(dimV, DropOutNum);%pΪ�������dimV�У�DropOutNum�з�Χ�����У�randperm��matlab�������������������һ���������С����ڵĲ���������������ķ�Χ��
            cMat(p) = 1;%cMat��p��Ԫ��Ϊ1
            cMat = diag(cMat);%cMatΪ�Խ���ΪcMatԪ�صľ���
        end
        
        % Gibbs sampling step 0%����˹��������0
        vis0 = double(V(bind,:)); % Set values of visible nodes�����Բ�ڵ��ֵ��
        if( DropOutRate > 0 )%���DropOutRate����0
            vis0 = vis0 * cMat;%vis0Ϊvis0����cMat
        end
        hid0 = v2h( rbm, vis0 );  % Compute hidden nodes%�������ؽڵ�%����v2h����

        % Gibbs sampling step 1%����˹��������1
        if( isequal(rbm.type(3), 'P') )%���rbm.type(3)��P������������ͬ
            bhid0 = hid0;%bhid0Ϊhid0
        else
            bhid0 = double( rand(size(hid0)) < hid0 );%bhid0Ϊdouble����
        end
        vis1 = h2v( rbm, bhid0 );  % Compute visible nodes%�������ؽڵ�%����h2v����
        if( DropOutRate > 0 )%���DropOutRate����0
            vis1 = vis1 * cMat;%vis1Ϊvis1*cMat
        end
        hid1 = v2h( rbm, vis1 );  % Compute hidden nodes%�������ؽڵ�%����v2h����

		posprods = hid0' * vis0;%posprodΪhid0��ת�ó���vis0
		negprods = hid1' * vis1;%negprodsΪhid1��ת�ó���vis1
		% Compute the weights update by contrastive divergence%ͨ���Ա�ɢ��������Ȩ�صĸ��¡�

        dW = (posprods - negprods)';%dWΪ��posprods-negprods����ת��
        dB = (sum(hid0, 1) - sum(hid1, 1));%dBΪhid0������Ԫ��֮�ͼ�ȥhid1������Ԫ��֮��
        dC = (sum(vis0, 1) - sum(vis1, 1));%dCΪvis0������Ԫ��֮�ͼ�ȥvis1������Ԫ��֮��
        
        if( strcmpi( 'GBRBM', rbm.type ) )%strcmpi�Ƚ������ַ����Ƿ���ȫ��ȣ�������ĸ��Сд;���GBRBM��rbm.type�ַ������
        	dW = bsxfun(@rdivide, dW, rbm.sig');%ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
        	dC = bsxfun(@rdivide, dC, rbm.sig .* rbm.sig);%ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
        end

		deltaW = momentum * deltaW + (StepRatio / num) * dW;%deltaWΪ��������deltaW�Ӳ����ʳ���num����dW
		deltaB = momentum * deltaB + (StepRatio / num) * dB;%deltaBΪ��������deltaB�Ӳ����ʳ���num����dB
		deltaC = momentum * deltaC + (StepRatio / num) * dC;%deltaCΪ��������deltaC�Ӳ����ʳ���num����dC

         if( SparseLambda > 0 )%���ϡ��ϵ��Lambda������
            deltaW = deltaW + numel(bind) / num * dsW;%deltaWΪdeltaW��bind�е�Ԫ�صĸ�������num����dsW
            deltaB = deltaB + numel(bind) / num * dsB;%deltaBΪdeltaB��bind�е�Ԫ�صĸ�������num����dsB
        end

		% Update the network weights%��������Ȩ��
		rbm.W = rbm.W + deltaW - WeightCost * rbm.W;%rbm.WΪrbm.W��deltaw��Ȩ�س���rbm.W
		rbm.b = rbm.b + deltaB;%rbm.bΪrbm.b��deltaB
		rbm.c = rbm.c + deltaC;%rbm.cΪrbm.c��deltaC

    end

    if( SparseLambda > 0 && strcmpi( 'GBRBM', rbm.type ) )%���SparseLambda����0����GBRBM��rbm.type�ַ������
        dsW = bsxfun(@rdivide, dsW, rbm.sig');%ʹ�ú���bsxfun���Ա�����ѭ���ṹ��̡�bsxfun���ø�ʽ:bsxfun(@���ж���ĺ������� ����1������2)
    end

    
	if( Verbose )%���չʾ����
        H = v2h( rbm, V );%HΪ����v2h����
        Vr = h2v( rbm, H );%VrΪh2v����
		err = power( V - Vr, 2 );%errΪV-Vr��ƽ��
		rmse = sqrt( sum(err(:)) / numel(err) );%rmseΪ��sum(err(:))����numel(err)��ƽ��
        
        totalti = toc(timer);%tic��toc������¼matlab����ִ�е�ʱ�䡣tic�������浱ǰʱ�䣬����ʹ��toc����¼�������ʱ�䡣
        aveti = totalti / iter;%avetiΪtotalti����iter
        estti = (MaxIter-iter) * aveti;%esttiΪ������������ȥ������������aveti
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');%datestr�ǽ����ں�ʱ��ת��Ϊ�ַ�����ʽ����
        
		fprintf( '%3d : %9.4f %9.4f %9.4f %s\n', iter, rmse, mean(H(:)), aveti, eststr );%fprintf�������Խ����ݰ�ָ����ʽд�뵽�ı��ļ��С�fprintf��fid,format,variables��,��ָ���ĸ�ʽ��������ֵ�������Ļ��ָ���ļ�
        %d ����
        %e ʵ������ѧ���㷨��ʽ
        %f ʵ����С����ʽ
        %g ��ϵͳ�Զ�ѡȡ�������ָ�ʽ֮һ
        %s ����ַ���
    end
end

