function [y, pout] = FDL_part_conv(x,h_full,N)
% Frequency Delay Line Partitioned Convolution
% (improvement on uniform part. conv.)
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



xbuf = buffer(x,N); %split x up into frames in columns
ybuf = zeros(size(xbuf));

nx = length(x);
num_blocks = nx/N - 1;

% circular fft buffer
X_fft = zeros(2*N,num_parts);

% sample buffers
inbuffer2 = zeros(N, 1);
% don't need to allocate these two:
% inbuffer1 = zeros(N, 1);
% outbuffer = zeros(N, 1);


bcurr = 1;  % current block
while bcurr <= num_blocks
    
%     istart = 1 + (bcurr-1)*N;
%     iend = istart + N - 1;
    
    %compute output up to time iend
    Yfft = zeros(2*N,1);

    for p = 1:num_parts
        slot = mod(bcurr-p,num_parts)+1;
        Yfft = Yfft + X_fft(:,slot).*H(:,p);
    end
    Y = ifft(Yfft);
    outbuffer = Y(1:N);
    
    % write output buffer to output array
    ybuf(:,bcurr) = outbuffer;
%     y(istart:iend) = outbuffer
    
   
    %buffer input up to time iend
    inbuffer1 = inbuffer2;
    inbuffer2 = xbuf(:,bcurr);
%     inbuffer2 = x(istart:iend);
    
    slot = mod(bcurr,num_parts)+1;
    X_fft(:,slot) = fft([inbuffer1; inbuffer2]);
    
    bcurr = bcurr + 1;


end

y = ybuf(:);

if nargin > 1
    pout = struct;
    pout.N = N;
    pout.numN = num_parts;
end


