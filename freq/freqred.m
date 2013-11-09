function s = freqred(c)
k = int32(sqrt(numel(c)));
kk = k*k;
h = k/2;
hh = h*h;
tmp = repmat ((0:(h-1)), h, 1);
base = (1:2:2*hh) + k.*(tmp(:))';
s = c(base) + c(base+1) + c(base+k) + c(base+k+1);
end
