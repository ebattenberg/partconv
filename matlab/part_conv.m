function [y, pout] = part_conv(x,h_full,p)
% Multiple Frequency Delay Line Partitioned Convolution
% x, input
% h_full, impulse response
% p, partition parameters
%   .N, array, block size of each FDL
% y, output

    if isstruct(p)
        M = length(p.N);
        if M == 1
            [y,pout] = FDL_part_conv(x,h_full,p.N);
        elseif M == 2
            [y,pout] = DFDL_part_conv(x,h_full,p);
        elseif M > 2
            [y,pout] = MFDL_part_conv(x,h_full,p);
        end
    elseif isnumeric(p)
        [y,pout] = FDL_part_conv(x,h_full,p(1));
    end
            