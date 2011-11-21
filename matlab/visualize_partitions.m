function visualize_partitions(pout,h)
%VISUALIZE_PARTITIONS visualize partitioning used in convolution
%   Detailed explanation goes here
    N = pout.N;
    numN = pout.numN;
    parts = [];
    color = 1;
    for i = 1:length(N)
        for j = 1:numN(i)
            parts = [parts color*ones(1,N(i))];
            color = color + 1;
        end
    end

    h_temp = [h; zeros(length(parts)-length(h),1)];
    h_labeled = zeros(length(h_temp),max(parts));
    for i = 1:max(parts)
        h_labeled(i==parts,i) = h_temp(i==parts);
    end
    h_labeled = h_labeled(1:length(h),:);
    
    
    % constants ----
    k = 1.5;
    Ospecmult = 4;
    % -------------
    
    ind = 1;
    cost = zeros(sum(N.*numN),1);
    M = length(N);
    for L = 1:M
        cost(ind) = 4*k*log2(2*N(L)) + Ospecmult;
        ind = ind + N(L);
        for P = 2:numN(L)
            cost(ind) = Ospecmult;
            ind = ind + N(L);
        end
    end

    cumcost = cumsum(cost);
    

%     subplot(3,1,1)
figure
    plot(h_labeled)
    title(['Impulse Response: length = ' num2str(length(h))])
    axis([1 length(parts) min(h) max(h)])
    
    
%     subplot(3,1,2)
    figure
    image(parts);
    title(['Partitioning: size = ' int2str(N) ', number = ' int2str(numN)])
    colormap(lines(128))
    
%     subplot(3,1,3)
    figure
    stairs(cumcost);
    axis([0 length(parts) 0 max(cumcost)+10])
    title(['Cumulative Cost = ' int2str(cumcost(length(cumcost)))]);

end
