% CalcErrorRate: calculate error rate%CalcErrorRate:���������
%
% ErrorRate = CalcErrorRate( dbn, IN, OUT )%CalcErrorRate�������ø�ʽ
%
%
%Output parameters:%�������
% ErrorRate: error rate%ErrorRate:Ϊ������
%
%
%Input parameters:%�������
% dbn: network%dbn:����
% IN: input data, where # of row is # of data and # of col is # of input features%IN:���������У��������ݣ��������������ԡ�
% OUT: output data, where # of row is # of data and # of col is # of output labels%OUT:��������У��������ݣ����������ǩ��
%Version: 20131213%�汾��20131213

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:  %���������                       %
%                                                          %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%    %��Ȩ(C) 2013��Masayuki Tanaka����������Ȩ����          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ErrorRate = CalcErrorRate( dbn, IN, OUT )%����CalcErrorRate�ĵ��ø�ʽ
 out = v2h( dbn, IN );%out����v2h����
 [m ind] = max(out,[],2);%[m ind]Ϊmax(out,[],2)
 out = zeros(size(out));%outΪ��СΪout��ȫ�����
 for i=1:size(out,1)%i��ȡֵ��ΧΪ1��out�����Ĵ�С
  out(i,ind(i))=1;%out�ĵ�i�У���ind(i)��Ϊ1
 end
 
 ErrorRate = abs(OUT-out);%ErrorRateΪOUT-out�ľ���ֵ
 ErrorRate = mean(sum(ErrorRate,2)/2);%ErrorRateΪErrorRate������֮�ͳ���2��ƽ��ֵ

end

