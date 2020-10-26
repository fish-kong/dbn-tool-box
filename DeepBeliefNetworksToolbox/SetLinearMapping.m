% SetLinearMapping: set the RBM associated to the linear mapping to the last layer %SetLinearMapping:��������ӳ����ص�RBM����Ϊ���һ�㡣
%
% dbn = SetLinearMapping( dbn, IN, OUT )%��dbn����ΪSetLinearMapping(dbn,IN,OUT )
%
%
%Input parameters:%�������
% dbn: the Deep Belief Nets (DBN) model��%dbn�������������ģ��
% IN: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes%IN���ɼ�(����)����������#���������ݵ�������#. col�ǿɼ�(����)�ڵ��#.
% OUT: teaching data, where # of row is number of data and # of col is # of hidden (output) nodes%OUT:��ѧ���ݣ�����# of row�����ݵ���������# of col������(���)�ڵ��#.
%
%
%Output parameters:%�������
% dbn: the set Deep Belief Nets (DBN) model%�ѽ����������������ģ��
%
%
%Example:%����
% datanum = 1024;%ʵ������
% outputnum = 16;%�����Ŀ
% hiddennum = 8;%�������
% inputnum = 4;%������Ŀ
% 
% inputdata = rand(datanum, inputnum);%��������Ϊ�������(datanum,inputnum)
% outputdata = rand(datanum, outputnum);%�������Ϊ�������(datanum,outputnum)
% 
% dbn = randDBN([inputnum, hiddennum, outputnum]);%����randDBN����([inputnum,hiddenum,outputnum])
% dbn = pretrainDBN( dbn, inputdata );%����pretrainDBN����(dbn��inputdata)
% dbn = SetLinearMapping( dbn, inputdata, outputdata);%����SetLinearMapping����(dbn,inputdata,outputdata)
% dbn = trainDBN( dbn, inputdata, outputdata );%����trainDBN����(dbn��inputdata��outputdata)
% 
% estimate = v2h( dbn, inputdata );%���ƣ�����v2h����(dbn,inputdata)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dbn = SetLinearMapping( dbn, IN, OUT )%�������ܺ���SetLinearMapping
nrbm = numel(dbn.rbm);%nrbmΪdbn.type��Ԫ�ظ���
if( nrbm > 1 )%���nrbm����1
    Hall = v2hall( dbn, IN );%����v2hall����
    dbn.rbm{nrbm}.W = linearMapping( Hall{nrbm-1}, OUT );%dbn.rbm{nrbm}.W����LinearMapping����
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));%dbn.rbm{nrbm}.bΪ-0.5����һ����СΪdbn.rbm{nrbm}.b��ȫһ����
else
    dbn.rbm{nrbm}.W = linearMapping( IN, OUT );%dbn.rbm{nrbm}.W����linearMapping����
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));%dbn.rbm{nrbm}.bΪ-0.5����һ����СΪdbn.rbm{nrbm}.b��ȫһ����
end
