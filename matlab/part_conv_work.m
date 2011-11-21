function cost = part_conv_work(pout)


N = pout.N(:)';
numN = pout.numN(:)';


k = 1.5;
Offt = 2*k*log2(2*N);  %cost per output point for N-pt fft
Ospecmult = 4;         %number of madds for a complex mult 


cost = sum(2*Offt + numN*Ospecmult);


            
        
