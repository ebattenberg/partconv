function y = overlap_save2(x,h,nfft)
% x, input
% h, impulse response
% nfft, FFT length
% y, output

x = x(:);
h = h(:);

nh = length(h);


% length of portion of y kept for each segment
% (hopsize)
L = nfft - (nh - 1);

if L < 1
    error('fft size is too small');
end

% H = fft(h,nfft);

% circular shift to get valid y-values at beginning of array.
% (circular shift by L)
H = fft([h(end); zeros(L-1,1); h(1:end-1)]);

x = [x; zeros(nh,1)];
padL = L-mod(length(x),L);
x = [zeros(nh,1); x; zeros(padL,1)];
y = zeros(size(x));

nx = length(x);


istart = 1;
while istart <= nx

    iend = min(istart+nfft-1,nx);
    if (iend - istart) == 0
        X = x(istart(ones(nfft,1)));
    else
        X = fft(x(istart:iend),nfft);
    end
    Y = ifft(X.*H);
    ystart = istart+nh-1;
    y(ystart:iend) = Y(1:(iend-ystart+1));
    istart = istart + L;

end

y = y(nh+1:end);
