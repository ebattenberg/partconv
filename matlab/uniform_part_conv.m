function y = uniform_part_conv(x,h_full,N)
% x, input
% h_full, impulse response
% N, partition length
% y, output

x = [x(:); zeros(length(h_full),1)];
h_full = h_full(:);

num_parts = ceil(length(h_full)/N);


h_full = [h_full; zeros(N*num_parts-length(h_full),1)];
h = reshape(h_full,[N num_parts]);

% circular shift to get valid y-values at beginning of array.
% (circular shift by N)
H = fft([zeros(N,num_parts); h]); 

nfft = 2*N;

padN = N-mod(length(x),N);
x = [zeros(N,1); x; zeros(padN,1)];
y = zeros(size(x));

nx = length(x);
num_blocks = nx/N - 1;

% circular fft buffer
X_fft = zeros(nfft,num_parts);


bcurr = 1;  % current block
while bcurr <= num_blocks
    
    istart = 1 + (bcurr-1)*N;
    iend = istart + nfft - 1;
    
    slot = mod(bcurr-1,num_parts)+1;
    X_fft(:,slot) = fft(x(istart:iend));
    
    ystart = istart + N;
    
    for p = 1:num_parts
        slot = mod(bcurr-p,num_parts)+1;
        Y = ifft(X_fft(:,slot).*H(:,p));
        y(ystart:iend) = y(ystart:iend) + Y(1:N);
    end
        
    
    bcurr = bcurr + 1;

end

y = y(N+1:end);

