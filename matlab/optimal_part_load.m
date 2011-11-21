function [pout, cost_min] = optimal_part_load(L,N)

% optimal partition finder algorithm
% uses dynamic programming (Viterbi search)

k = 1.5;
%nh = length(h);
nh = L;
% number of pointer positions
T = ceil(nh/N);

% max block size
MB = 2^(nextpow2(floor(T/2))-1);


% number of states
% (allow maximum block size to be the highest power of 2
% multiple of N that is less than half the length of h)
I = 2*MB - 1;

cost = zeros(T,I);
best_index = zeros(T,I);




% compute transition cost matrix
C = zeros(I,I);
for j = 1:I
    C(j,:) = trans_cost(j,N,I,k);
end
VI = ~isinf(C); % valid transitions
indices = 1:I;

state_block = zeros(1,I); 
for i = 1:I
    state_block(i) = 2^(nextpow2(i+1)-1);
end
    

fprintf('\ncomputing optimal partitioning\n');
% dynamic programming loop
cost(1,1) = 4*k*log2(2*N) + 4;
cost(1,2:I) = inf;
best_index(2,:) = 1;
cost(2,1) = cost(1,1) + 4;
cost(2,2:I) = inf;
for t = 3:T
    if ~mod(t,100)
        fprintf('step %u of %u\n',t,T);
    end
    for i = 1:I
        if 2*state_block(i) < t+2
            valid_indices = indices(VI(:,i));            
            [cost(t,i) IX] = min(C(valid_indices,i)' + cost(t-1,valid_indices));
            best_index(t,i) = valid_indices(IX);
            
        else
            cost(t,i) = inf;
        end

    end
end



% backtrack through cost matrix
[cost_min ind_min] = min(cost(T,:));
state_sequence = zeros(1,T);
cum_cost = zeros(1,T);
state_sequence(T) = ind_min;
cum_cost(T) = cost_min;
for t = T-1:-1:1
    state_sequence(t) = best_index(t+1,state_sequence(t+1));
    cum_cost(t) = cost(t,state_sequence(t));
end


% create partitioning struct from optimal state sequence
pout = struct;
L = 1;
pout.N = 1;
pout.numN = 0;
for t = 1:T
    X = 2^(nextpow2(state_sequence(t)+1)-1);
    Y = state_sequence(t) - X + 1;
    if Y == 1
        if X == pout.N(L)
            pout.numN(L) = pout.numN(L) + 1;
        else
            L = L + 1;
            pout.N(L) = X;
            pout.numN(L) = 1;
        end
    end
end
        
pout.N = pout.N*N;

fprintf('\nfor length %u filter \nwith N(1) = %u, \noptimal partitioning:\n',nh,N);
disp(pout);
fprintf('cost = %u\n',cost_min);


    
  




function C = trans_cost(j,N,I,k)
% j - previous state
% i - current state
% state - [x.y]
xj = 2^(nextpow2(j+1)-1);
yj = j-xj+1;

C = inf*ones(1,I);
if yj < xj % then there's only one valid transition
    C(j+1) = 0;

else

    for i = 1:I
        xi = 2^(nextpow2(i+1)-1);
        yi = i-xi+1;

        if xj == xi && yj == xj && yi ==1   % same block size 
            C(i) = 4;  %start new block of same size
            % A.A->B.1; B>A
        elseif xj == yj && yi == 1 && xi > xj
            C(i) = 4*k*log2(2*xi*N) + 4;
        end
    end
end
    


%%
% %state [x.y] -> index (>=1)
% 
% x = [1 2 2 4 4];
% y = [1 1 2 1 2];
% 
% 
% ind = x + y - 1;
% 
% 
% %index -> state [x.y]
% % ind =  1;
% x = 2^(nextpow2(ind+1)-1);
% y = ind-x+1;


