% linearMapping: calculate the linear mapping matrix between the input data and the output data%LinearMapping���������������������֮�������ӳ�����
%
% M = linearMapping( IN, OUT )%LinearMapping�������ø�ʽ
%
%
%Output parameters:%�������
% M: The linear mapping matrix%M�����Է������
%
%
%Input parameters:%�������
% IN: input data, where # of row is # of data and # of col is # of input
% features%IN:�������ݵ���������,�����������ԡ�
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:������ݣ�������������,���������ǩ��
%
%
%Example:����
% datanum = 1024;%datanumΪ1024
% outputnum = 16;%outputnumΪ16
% inputnum = 4;%inputnumΪ4
%
% inputdata = rand(datanum, inputnum); %inputdataΪ�����datanum�У�inputnum�еľ���
% outputdata = rand(datanum,outputnum);%outputdataΪ�����datanum�У�outputnum�еľ���
%
% M = linearMapping(inputdata, outputdata);MΪ����LinearMapping����
%
%
%Version: 20130727%�汾��20130727

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network: %���������                        %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%     ��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M = linearMapping( IN, OUT )%LinearMapping�ĵ��ø�ʽ
M = pinv(IN) * OUT;%pinv(IN):��IN��α�����

%OUT = IN * M;%OUT=IN*M
