function y = overlap_save(x,h,nfft)
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

H = fft(h,nfft);

x = [zeros(nh,1); x; zeros(nh,1)];
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
    y(ystart:iend) = Y(nh:(iend-ystart+nh));
    istart = istart + L;

end

y = y(nh+1:end);
