function err = ObjectFunc( dbn, IN, OUT, opts )%ObjectFunc函数的调用格式

OBJECTSQUARE = 1;%OBJECTSQUARE为1
OBJECTCROSSENTROPY = 2;%OBJECTCROSSENTROPY为2
Object = OBJECTSQUARE;%Object为OBJECTSQUARE

if( exist('opts' ) )%如果存在类型opts
 if( isfield(opts,'Object') )%检查结构体opts是否包含由Object指定的域， 如果包含，返回逻辑1; 如果opts不包含Object域或者opts不是结构体类型的， 返回逻辑0。
  if( strcmpi( opts.object, 'Square' ) )%如果opts.object与Square字符串长度相等
   Object = OBJECTSQUARE;%Object为OBJECTSQUARE
  elseif( strcmpi( opts.object, 'CrossEntropy' ) )%如果opts.object与CrossEntropy字符串长度相等
   Object = OBJECTCROSSENTROPY;%Object为OBJECTCROSSENTROPY
  end
 end
end
 
est = v2h( dbn, IN );%est调用v2h函数

if( Object == OBJECTSQUARE )%如果Object恒等于OBJECTSQUARE
    err = ( OUT - est );%err为OUT-est
    err = err .* err;%err为err.*err
    err = sum(err(:)) / size(OUT,1) / 2;%err为err(:)的总和除以OUT行数除以2

elseif( Object == OBJECTCROSSENTROPY )%如果Object恒等于OBJECTCROSSENTROPY
    e1 = OUT .* log( est );%e1为OUT.*log(est)Matlab中的log函数在默认情况下是以e为底，即loge，如果需要计算以10为底的对数，那么需要用log10()函数。同理计算以2为底的对数需要用log2()函数。
    e2 = (1-OUT) .* log( 1 - est );%e2为(1-OUT).*log(1-est)
    err = -( sum(e1(:)) + sum(e2(:)) ) / size(OUT, 1);%err为负的e1的总和与e2的总和的和除以OUT行数的大小
end

end

