function [y, pout] = MFDL_part_conv(x,h_full,p)
% Multiple Frequency Delay Line Partitioned Convolution
% x, input
% h_full, impulse response
% p, partition parameters
%   .N, array, block size of each FDL
% y, output

    global N;
    N = p.N;
    numN = p.numN;
    global M;
    M = length(N);  %number of levels


    if length(numN) < length(N)-1
        error('must provide number of partitions of each size (except for last size)');
    end

    if any(N ~= 2.^round(log2(N)))
        error('N must contain powers of 2');
    end

%     must allow enough delay to fill buffer for next partition size
    cumdelay = 0;
    for L = 1:M-1
        cumdelay = cumdelay + N(L)*numN(L);
        if cumdelay < N(L+1)
            error('partitioning error');
        end
    end
      

    % check for too many partitions
    if sum(N(1:M-1).*numN(1:M-1)) > length(h_full)
        error('last partitioning level not required');
    end

    x = [x(:); zeros(length(h_full),1)];  %so y holds entire reverberation
    h_full = h_full(:);


    global xbuf ybuf;  %global so copies aren't made in recursive calls
    xbuf = buffer(x,N(1)); %split x up into frames in columns
    ybuf = zeros(size(xbuf));


    %gather filter partitions for each level
    h = cell(M,1);
    cumsize = 0;
    for L = 1:M-1
        h{L} = buffer(h_full(cumsize+1:cumsize+N(L)*numN(L)),N(L));
        cumsize = cumsize + N(L)*numN(L);
    end
    h{M} = buffer(h_full(cumsize+1:end),N(M));
    numN(M) = size(h{M},2);

    global num_parts;
    num_parts = numN;


    % circular shift to get valid y-values at beginning of array.
    % (circular shift by N)
    global H;
    H = cell(M,1);
    for L = 1:M
        H{L} = fft([zeros(N(L),num_parts(L)); h{L}]);
    end


    % need to fix these buffers to allow for one less partition in 2+ level
    % circular fft buffers  
    global X_fft;
    X_fft = cell(M,1);
    for L = 1:M
        X_fft{L} = zeros(2*N(L),num_parts(L)); 
    end

    % circular sample buffers
    global X;
    
    
    X = cell(M-1,1);
    for L = 1:M-1
        X{L} = zeros(N(L),num_parts(L)+1);
    end

    global inbuffer1 inbuffer2 outbuffer;
    inbuffer1 = cell(M,1);
    inbuffer2 = cell(M,1);
    outbuffer = cell(M,1);
    for L = 1:M
        inbuffer1{L} = zeros(N(L),1);
        inbuffer2{L} = zeros(N(L),1);
        outbuffer{L} = zeros(N(L),1);
    end

    global num_subblocks;
    num_big_blocks = floor(length(x)/N(M));
    num_subblocks = [N(2:M)./N(1:M-1) 1];




    bcurr = ones(1,M);
    subblock = ones(1,M);

    % MAIN LOOP
    % -----------------
    while bcurr(M) <= num_big_blocks

        [bcurr, subblock] = nested_levels(M,bcurr,subblock);

    end
    % -----------------


    y = ybuf(:);

    if nargout > 1
        pout.N = N;
        pout.numN = num_parts;
    end

end



% recursive function to handle arbitrary number of levels

function [bcurr, subblock] = nested_levels(L,bcurr,subblock)

    global N M num_parts num_subblocks H xbuf;  % constants
    global ybuf X X_fft inbuffer1 inbuffer2 outbuffer; %mutable

    if L == 1
        subblock(1) = 1;
        subb_ind = (subblock-1) .* N;
        for B = 1:num_subblocks(1)
            
            subblock(1) = B;
            
            Yfft = zeros(2*N(1),1);
            for P = 1:num_parts(1)
                slot = mod(bcurr(1)-P-1,num_parts(1))+1;
                Yfft = Yfft + X_fft{1}(:,slot).*H{1}(:,P);
            end
            Y = ifft(Yfft);
            outbuffer{1} = Y(1:N(1));

            % write output to audio device
            
            ybuf(:,bcurr(1)) = outbuffer{1};
            ind = 1 + (B-1)*N(1);
            for l = 2:length(N)
                ind = ind + subb_ind(l-1);
                ybuf(:,bcurr(1)) = ybuf(:,bcurr(1)) ...
                    + outbuffer{l}(ind:ind+N(1)-1);
            end
                
                
           

            % read input in from audio device
            inbuffer2{1} = xbuf(:,bcurr(1));

         

            slot = mod(bcurr(1)-1,num_parts(1)+1)+1;
            X{1}(:,slot) = inbuffer2{1};
            slot = mod(bcurr(1)-1,num_parts(1))+1;
            X_fft{1}(:,slot) = fft([inbuffer1{1}; inbuffer2{1}]); 

            inbuffer1{1} = inbuffer2{1};

            bcurr(1) = bcurr(1) + 1;

        end
        
    elseif L ~= M %if 1 < L < M
        
        for B = 1:num_subblocks(L)
            subblock(L) = B;
        
        
            Yfft = zeros(2*N(L),1);
            for P = 1:num_parts(L)
                slot = mod(bcurr(L)-P-1,num_parts(L))+1;
                Yfft = Yfft + X_fft{L}(:,slot).*H{L}(:,P);
            end   
            Y = ifft(Yfft);
            outbuffer{L} = Y(1:N(L));


            % recursion
            bcurr = nested_levels(L-1,bcurr,subblock);
            % ---------


            istart = bcurr(L-1)-num_parts(L-1)-2;
            slots = mod(istart:istart+num_subblocks(L-1)-1,num_parts(L-1)+1)+1;
            inbuffer2{L} = reshape(X{L-1}(:,slots),[],1);


            slot = mod(bcurr(L)-1,num_parts(L)+1)+1;
            X{L}(:,slot) = inbuffer2{L};
            slot = mod(bcurr(L)-1,num_parts(L))+1;
            X_fft{L}(:,slot) = fft([inbuffer1{L}; inbuffer2{L}]); 
            
            inbuffer1{L} = inbuffer2{L};

            bcurr(L) = bcurr(L) + 1;
        end
    else %if L == M
       
        
            Yfft = zeros(2*N(L),1);
            for P = 1:num_parts(L)
                slot = mod(bcurr(L)-P-1,num_parts(L))+1;
                Yfft = Yfft + X_fft{L}(:,slot).*H{L}(:,P);
            end   
            Y = ifft(Yfft);
            outbuffer{L} = Y(1:N(L));


            % recursion
            bcurr = nested_levels(L-1,bcurr,subblock);
            % ---------


            istart = bcurr(L-1)-num_parts(L-1)-2;
            slots = mod(istart:istart+num_subblocks(L-1)-1,num_parts(L-1)+1)+1;
            inbuffer2{L} = reshape(X{L-1}(:,slots),[],1);


            slot = mod(bcurr(L)-1,num_parts(L))+1;
            X_fft{L}(:,slot) = fft([inbuffer1{L}; inbuffer2{L}]);
            
            inbuffer1{L} = inbuffer2{L};

            bcurr(L) = bcurr(L) + 1;
    end
end
        
    
        
        

