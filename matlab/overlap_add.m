function y = overlap_add(x,h,nfft)
% x, input
% h, impulse response
% nfft, FFT length
% y, output

x = x(:);
h = h(:);

nh = length(h);




% length of segments of x  (hopsize)
L = nfft - (nh - 1);

if L < 1
    error('fft size is too small');
end

H = fft(h,nfft);

x = [x; zeros(nh,1)];
padL = L-mod(length(x),L);
x = [x; zeros(padL,1)];
y = zeros(size(x));

nx = length(x);


istart = 1;
while istart <= nx
    iend = min(istart+L-1,nx);
    if (iend - istart) == 0
        X = x(istart(ones(nfft,1)));
    else
        X = fft(x(istart:iend),nfft);
    end
    Y = ifft(X.*H);
    yend = min(nx,istart+nfft-1);
    y(istart:yend) = y(istart:yend) + Y(1:(yend-istart+1));
    istart = istart + L;
end

