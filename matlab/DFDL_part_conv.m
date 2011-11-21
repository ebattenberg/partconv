function [y, pout] = DFDL_part_conv(x,h_full,p)
% Double Frequency Delay Line Partitioned Convolution
% x, input
% h_full, impulse response
% p, partition parameters
%   .N, 1st block size
%   .numN, number of blocks of size N
% y, output

N = p.N;
numN = p.numN;

if length(numN) < length(N)-1
    error('must provide number of partitions of each size');
end

if any(N ~= 2.^round(log2(N)))
    error('N must contain powers of 2');
end


if N(1)*numN(1) < N(2)
    error('partitioning error');
end



x = [x(:); zeros(length(h_full),1)];
h_full = h_full(:);

xbuf = buffer(x,N(1)); %split x up into frames in columns
ybuf = zeros(size(xbuf));


%gather filter partitions for each level
h = {};
h{1} = buffer(h_full(1:N(1)*numN(1)),N(1));
cumsize = N(1)*numN(1);
h{2} = buffer(h_full(cumsize+1:end),N(2));
numN(2) = size(h{2},2);

num_parts = numN;

% circular shift to get valid y-values at beginning of array.
% (circular shift by N)
H = {};
H{1} = fft([zeros(N(1),num_parts(1)); h{1}]); 
H{2} = fft([zeros(N(2),num_parts(2)); h{2}]); 

 

% circular fft buffers
X_fft = {};
X_fft{1} = zeros(2*N(1),num_parts(1));
X_fft{2} = zeros(2*N(2),num_parts(2));
X = zeros(N(1),num_parts(1));

%sample buffers
inbuffer1 = cell(2,1);
inbuffer2 = cell(2,1);
inbuffer1{1} = zeros(N(1),1);
inbuffer1{2} = zeros(N(2),1);
inbuffer2{1} = zeros(N(1),1);
inbuffer2{2} = zeros(N(2),1);

outbuffer = cell(2,1);

num_big_blocks = floor(length(x)/N(2));
num_subblocks = N(2)/N(1);

% L1 filtering involves smaller blocks
% L2 filtering involves bigger blocks
bcurr = [1 1];  % current block [L1,L2]
while bcurr(2) <= num_big_blocks
    

   
    % do L2 filtering
    L = 2;

    Yfft = zeros(2*N(L),1);
    for P = 1:num_parts(L)
        slot = mod(bcurr(L)-P,num_parts(L))+1;
        Yfft = Yfft + X_fft{L}(:,slot).*H{L}(:,P);
    end   
    Y = ifft(Yfft);
    outbuffer{L} = Y(1:N(L));
    
    
    % do all L1 processing
    L = 1;
    
    for B = 1:num_subblocks
        
        
        Yfft = zeros(2*N(L),1);
        for P = 1:num_parts(L)
            slot = mod(bcurr(L)-P,num_parts(L))+1;
            Yfft = Yfft + X_fft{L}(:,slot).*H{L}(:,P);
        end
        Y = ifft(Yfft);
        outbuffer{L} = Y(1:N(L));

        % write output to audio device
        ybuf(:,bcurr(L)) = outbuffer{L} + outbuffer{L+1}(1+(B-1)*N(L):B*N(L));
        
        % read input in from audio device
        inbuffer2{L} = xbuf(:,bcurr(L));
        
        
        slot = mod(bcurr(L),num_parts(L))+1;
        X(:,slot) = inbuffer1{L};
        X_fft{L}(:,slot) = fft([inbuffer1{L}; inbuffer2{L}]); 
        
        inbuffer1{L} = inbuffer2{L};
        

        bcurr(L) = bcurr(L) + 1;
        
    end
    
    
    % acquire L2 fft block
    L = 2;
    
    istart = bcurr(L-1)-num_parts(L-1);
    slots = mod(istart:istart+num_subblocks-1,num_parts(L-1))+1;
    inbuffer2{L} = reshape(X(:,slots),[],1);
    

    slot = mod(bcurr(L),num_parts(L))+1;
    X_fft{L}(:,slot) = fft([inbuffer1{L}; inbuffer2{L}]); %#ok<AGROW>
    
    inbuffer1{L} = inbuffer2{L};
    
    bcurr(L) = bcurr(L) + 1;
    
    
  
    

end

y = ybuf(:);

if nargout > 1
    pout.N = N(:)';
    pout.numN = num_parts(:)';
end

