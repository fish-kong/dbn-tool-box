function err = ObjectFunc( dbn, IN, OUT, opts )%ObjectFunc�����ĵ��ø�ʽ

OBJECTSQUARE = 1;%OBJECTSQUAREΪ1
OBJECTCROSSENTROPY = 2;%OBJECTCROSSENTROPYΪ2
Object = OBJECTSQUARE;%ObjectΪOBJECTSQUARE

if( exist('opts' ) )%�����������opts
 if( isfield(opts,'Object') )%���ṹ��opts�Ƿ������Objectָ������ ��������������߼�1; ���opts������Object�����opts���ǽṹ�����͵ģ� �����߼�0��
  if( strcmpi( opts.object, 'Square' ) )%���opts.object��Square�ַ����������
   Object = OBJECTSQUARE;%ObjectΪOBJECTSQUARE
  elseif( strcmpi( opts.object, 'CrossEntropy' ) )%���opts.object��CrossEntropy�ַ����������
   Object = OBJECTCROSSENTROPY;%ObjectΪOBJECTCROSSENTROPY
  end
 end
end
 
est = v2h( dbn, IN );%est����v2h����

if( Object == OBJECTSQUARE )%���Object�����OBJECTSQUARE
    err = ( OUT - est );%errΪOUT-est
    err = err .* err;%errΪerr.*err
    err = sum(err(:)) / size(OUT,1) / 2;%errΪerr(:)���ܺͳ���OUT��������2

elseif( Object == OBJECTCROSSENTROPY )%���Object�����OBJECTCROSSENTROPY
    e1 = OUT .* log( est );%e1ΪOUT.*log(est)Matlab�е�log������Ĭ�����������eΪ�ף���loge�������Ҫ������10Ϊ�׵Ķ�������ô��Ҫ��log10()������ͬ�������2Ϊ�׵Ķ�����Ҫ��log2()������
    e2 = (1-OUT) .* log( 1 - est );%e2Ϊ(1-OUT).*log(1-est)
    err = -( sum(e1(:)) + sum(e2(:)) ) / size(OUT, 1);%errΪ����e1���ܺ���e2���ܺ͵ĺͳ���OUT�����Ĵ�С
end

end

