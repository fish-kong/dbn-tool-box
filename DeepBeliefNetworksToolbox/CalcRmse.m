% CalcRmse: calculate the rmse between predictions and OUTs%CalcRmse:����Ԥ�������֮��ľ��������
%
% [rmse AveErrNum] = CalcRmse( dbn, IN, OUT )%CalcRmse�����ĵ��ø�ʽ
%
%
%Output parameters:%�������
% rmse: the rmse between predictions and OUTs%rmse:Ԥ�������֮��ľ��������
% AveErrNum: average error number after binarization%AveErrNum:��ֵ�����ƽ����������
%
%
%Input parameters:%�������
% dbn: network%dbn:����
% IN: input data, where # of row is # of data and # of col is # of input features%IN:���������У��������ݣ��������������ԡ�
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:��������У��������ݣ������������ǩ��
%
%
%Version: 20130727%�汾��201310727

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:%���������                         %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%      %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [rmse AveErrNum] = CalcRmse( dbn, IN, OUT )%CalcRmse�����ĵ��ø�ʽ
 out = v2h( dbn, IN );%out����v2h����

 err = power( OUT - out, 2 );%errΪOUT-out��ƽ��
 rmse = sqrt( sum(err(:)) / numel(err) );%rmseΪerr(:)���ܺͳ���err��Ԫ�صĸ���

 bout = out > 0.5;%boutΪout����0.5
 BOUT = OUT > 0.5;%BOUΪTOUT����0.5

 err = abs( BOUT - bout );%errΪBOUT-bout�ľ���ֵ
 AveErrNum = mean( sum(err,2) );%AveErrNumΪerr������֮�͵�ƽ��ֵ
end
