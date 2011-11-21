#ifdef _CUDA_


#include "runlevel.h"

#include "partconv.h"

#include "cuda.h"
#include "cutil.h"

#include <sys/time.h>

//#define _DEBUG_

typedef enum{
    compute,
    cleanup_mem
} action_t;

__global__ void filter(float2* xfft, float2* h, float2* yfft, const int bcurr, const int parts, const int cfft);
template <unsigned int blockSize>
__global__ void filter2DStrided(float2 *xfft, float2 *h, float2 *yfft, const int bcurr, const int parts, const int cfft);
template <unsigned int blockSize>
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int N, int stride);
void filter_reduce(action_t action, float* xfft, float* h, float* yfft, 
        const int bcurr, const int parts, const int cfft,  int* params);
void filter_reduce2(fftwf_complex* xfft, fftwf_complex* h, fftwf_complex* yfft, 
        const int bcurr, const int parts, const int cfft,  int* params);



double get_time_cuda()
{
    //output time in microseconds

    //the following line is required for function-wise timing to work,
    //but it slows down overall execution time.
    //comment out for faster execution
    cudaThreadSynchronize(); 

    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}




int PartConv::setupGPU(int L)
{

    // variables to put on GPU:
    // Xfft[L].Cdata - Cvec: Cmatrix stored in row-major order (num_parts[L] x cfft[L])
    // H[L].Cdata - Cvec: Cmatrix store in row-major order (num_parts[L] x cfft[L])
    // fftbuffer[L].Cdata - Cvec
    // Yfft[L].Cdata - Cvec
    // inbuffer[L].data - vec
    // outbuffer[L].data - vec


    //Xfft_d
    CUDA_SAFE_CALL(cudaMalloc(&(Xfft_d[L].Cdata), sizeof(fftwf_complex)*Xfft_d[L].size));
    CUDA_SAFE_CALL(cudaMemset(Xfft_d[L].Cdata, 0, Xfft_d[L].size*sizeof(fftwf_complex)));

    //H_d
    CUDA_SAFE_CALL(cudaMalloc(&(H_d[L].Cdata), sizeof(fftwf_complex)*H_d[L].size));
    for (int i = 0; i < H[L].rows; i++)
    {
        // copy data from H[L]
        //CUDA_SAFE_CALL(cudaMemcpy(&H_d[L].Cdata[i*cfft[L]],H[L].Cdata[i],sizeof(fftwf_complex)*cfft[L],cudaMemcpyHostToDevice));
        // transpose into GPU memory (for memory coalescing: to avoid strided memory access)
        CUDA_SAFE_CALL( cudaMemcpy2D( 
                    &H_d[L].Cdata[i],
                    sizeof(fftwf_complex)*num_parts[L],
                    H[L].Cdata[i],
                    sizeof(fftwf_complex)*1,
                    sizeof(fftwf_complex)*1,
                    cfft[L],
                    cudaMemcpyHostToDevice) );
    }
    //-----debug
    /*
    float* test = (float*)malloc(H_d[L].size*sizeof(fftwf_complex));
    CUDA_SAFE_CALL(cudaMemcpy(test,H_d[L].Cdata,sizeof(fftwf_complex)*H_d[L].size,cudaMemcpyDeviceToHost));
    FILE *fp = fopen("testH.txt","w");
    for (int i = 0; i < 2*H_d[L].size; i+=2)
        fprintf(fp,"%f+%fi \n",test[i],test[i+1]);
    fclose(fp);
    free(test);
    */
    //-------------


    //fftbuffer_d
    CUDA_SAFE_CALL(cudaMalloc(&(fftbuffer_d[L].Cdata), sizeof(cufftComplex)*fftbuffer_d[L].size));
    CUDA_SAFE_CALL(cudaMemset(fftbuffer_d[L].Cdata,0, fftbuffer_d[L].size*sizeof(fftwf_complex)));

    //Yfft_d
    CUDA_SAFE_CALL(cudaMalloc(&(Yfft_d[L].Cdata), sizeof(cufftComplex)*Yfft_d[L].size));
    CUDA_SAFE_CALL(cudaMemset(Yfft_d[L].Cdata,0, Yfft_d[L].size*sizeof(fftwf_complex)));

    //inbuffer_d
    CUDA_SAFE_CALL(cudaMalloc(&(inbuffer_d[L].data), sizeof(float)*inbuffer_d[L].size));
    CUDA_SAFE_CALL(cudaMemset(inbuffer_d[L].data,0, inbuffer_d[L].size*sizeof(float)));

    //outbuffer_d
    CUDA_SAFE_CALL(cudaMalloc(&(outbuffer_d[L].data), sizeof(float)*outbuffer_d[L].size));
    CUDA_SAFE_CALL(cudaMemset(outbuffer_d[L].data,0, outbuffer_d[L].size*sizeof(float)));


    // setup CUFFT plans
    CUFFT_SAFE_CALL(cufftPlan1d ( &(fwd_d[L]), nfft[L], CUFFT_R2C, 1 ));
    CUFFT_SAFE_CALL(cufftPlan1d ( &(bck_d[L]), nfft[L], CUFFT_C2R, 1 ));

    // allocate page-locked host memory to gpu transfer regions
    /*
    fftwf_free(inbuffer[L].data);
    CUDA_SAFE_CALL( cudaMallocHost(&inbuffer[L].data, sizeof(float)*inbuffer[L].size) );
    fftwf_free(outbuffer[L].data);
    CUDA_SAFE_CALL( cudaMallocHost(&outbuffer[L].data, sizeof(float)*outbuffer[L].size) );
    */



    return 0;
}

int PartConv::cleanupGPU(int L)
{
    //Xfft_d
    CUDA_SAFE_CALL(cudaFree(Xfft_d[L].Cdata));

    //H_d
    CUDA_SAFE_CALL(cudaFree(H_d[L].Cdata));

    //fftbuffer_d
    CUDA_SAFE_CALL(cudaFree(fftbuffer_d[L].Cdata));

    //Yfft_d
    CUDA_SAFE_CALL(cudaFree(Yfft_d[L].Cdata));

    //inbuffer_d
    CUDA_SAFE_CALL(cudaFree(inbuffer_d[L].data));

    //outbuffer_d
    CUDA_SAFE_CALL(cudaFree(outbuffer_d[L].data));

    // CUFFT plans
    CUFFT_SAFE_CALL(cufftDestroy(fwd_d[L]));
    CUFFT_SAFE_CALL(cufftDestroy(bck_d[L]));

    filter_reduce(cleanup_mem,NULL,NULL,NULL,0,0,0,NULL);

    // free page-locked host memory, and refill with fftwf memory
    /*
    CUDA_SAFE_CALL( cudaFreeHost(inbuffer[L].data) );
    inbuffer[L].data = (float*)fftwf_malloc(sizeof(float)*inbuffer[L].size);
    CUDA_SAFE_CALL( cudaFreeHost(outbuffer[L].data) );
    outbuffer[L].data = (float*)fftwf_malloc(sizeof(float)*outbuffer[L].size);
    */


    return 0;
}

int PartConv::runLevelGPU(int L)
{
#ifdef _DEBUG_
    double mem, ffts, filters;
    mem = 0; ffts = 0; filters = 0;
#endif

    // copy to gpu
#ifdef _DEBUG_
    mem -= get_time_cuda();
#endif
    CUDA_SAFE_CALL(cudaMemcpy(inbuffer_d[L].data,inbuffer[L].data,sizeof(float)*nfft[L],cudaMemcpyHostToDevice));
#ifdef _DEBUG_
    mem += get_time_cuda();
#endif

    int slot;

#ifdef _DEBUG_
    ffts -= get_time_cuda();
#endif

    //fft slot
    slot = bcurr[L] % num_parts[L];
    //take fft of inbuffer_d, save in fftbuffer_d
    CUFFT_SAFE_CALL( cufftExecR2C(fwd_d[L], inbuffer_d[L].data, (cufftComplex*)fftbuffer_d[L].Cdata) ); 
    //CUDA_SAFE_CALL( cudaMemcpy( &Xfft_d[L].Cdata[slot*cfft[L]], fftbuffer_d[L].Cdata, sizeof(fftwf_complex)*cfft[L], cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy2D( 
                &Xfft_d[L].Cdata[slot],
                sizeof(fftwf_complex)*num_parts[L],
                fftbuffer_d[L].Cdata,
                sizeof(fftwf_complex)*1,
                sizeof(fftwf_complex)*1,
                cfft[L],
                cudaMemcpyDeviceToDevice) );
#ifdef _DEBUG_
    ffts += get_time_cuda();
#endif



#ifdef _DEBUG_
    filters -= get_time_cuda();
#endif
    // do filtering on gpu
    if (0)
    {
        dim3 dimblock(CUDA_BLOCK);
        dim3 dimgrid(ceil((float)cfft[L]/CUDA_BLOCK));
        filter<<<dimgrid,dimblock>>>((float2*)Xfft_d[L].Cdata, (float2*)H_d[L].Cdata, (float2*)Yfft_d[L].Cdata, bcurr[L], num_parts[L], cfft[L]);
    }
    else
    {

        int temp = num_parts[L]/CUDA_BLOCK;
        temp +=  ((num_parts[L] == temp*CUDA_BLOCK) ? 0 : 1);
        int params[] = {CUDA_BLOCK, temp};
        //int params[] = {128, 128, 16};
        filter_reduce2(Xfft_d[L].Cdata, H_d[L].Cdata, Yfft_d[L].Cdata, bcurr[L], num_parts[L], cfft[L],  params);
        //filter_reduce(compute,(float*)Xfft_d[L].Cdata, (float*)H_d[L].Cdata, (float*)Yfft_d[L].Cdata, bcurr[L], num_parts[L], cfft[L],  params);
    }
#ifdef _DEBUG_
    filters += get_time_cuda();
#endif




#ifdef _DEBUG_
    ffts -= get_time_cuda();
#endif
    // take ifft of FDL
    //take ifft of Yfft_d, save in outbuffer_d
    CUFFT_SAFE_CALL( cufftExecC2R(bck_d[L], (cufftComplex*)Yfft_d[L].Cdata, outbuffer_d[L].data) ); 
#ifdef _DEBUG_
    ffts += get_time_cuda();
#endif

#ifdef _DEBUG_
    mem -= get_time_cuda();
#endif


    // copy buffer to cpu
    CUDA_SAFE_CALL(cudaMemcpy(outbuffer[L].data,outbuffer_d[L].data,nfft[L]*sizeof(float),cudaMemcpyDeviceToHost));
#ifdef _DEBUG_
    mem += get_time_cuda();
#endif


#ifdef _DEBUG_
    double total = mem + ffts + filters;
    printf("mem: %f.2%% (%f), ffts: %f.2%% (%f), filters: %f.2%% (%f)\n",100*mem/total,mem,100*ffts/total,ffts,100*filters/total,filters);
#endif



    return 0;
}


__global__ void filter(float2* xfft, float2* h, float2* yfft, const int bcurr, const int parts, const int cfft)
{
    //const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int i = blockDim.x*blockIdx.x +  threadIdx.x;

    int slot;

    if (i < cfft)
    {

        yfft[i].x = 0;
        yfft[i].y = 0;
        for (int p = 0; p < parts; p++)
        {

            slot = (bcurr - p + parts) % parts;

            const float2 ab = xfft[i*parts + slot];
            const float2 cd = h[i*parts + p];

            yfft[i].x += ab.x*cd.x - ab.y*cd.y;
            yfft[i].y += ab.x*cd.y + ab.y*cd.x;

        }
    }


}

__global__ void filter2D(float* xfft, float* h, float* r, const int bcurr, const int parts, const int cfft)
{
    //const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int element = blockDim.x*blockIdx.x +  threadIdx.x;
    const int p = blockIdx.y;




    if (element < cfft) //don't need to check p (blockDim.y=1)
    {

        const int ind = 2*element;
        //const int ind1 = ind+1;
        const int offset = 2*cfft;

        const int slotoffset = offset*((bcurr - p + parts) % parts);
        const int poffset = p*offset;

        float2 *ptr;

        ptr = (float2*)&xfft[slotoffset+ind];
        const float2 ab = *ptr;

        ptr = (float2*)&h[poffset+ind];
        const float2 cd = *ptr;
        

        //result[poffset+ind] = a*c - b*d;
        //result[poffset+ind1] = a*d + b*c;

        float2 result;
        result.x = ab.x*cd.x - ab.y*cd.y;
        result.y = ab.x*cd.y + ab.y*cd.x;

        ptr = (float2*)&r[poffset+ind];
        *ptr = result;



        

        //result[poffset+ind] = ab.x*cd.x - ab.y*cd.y;
        //result[poffset+ind1] = ab.x*cd.y + ab.y*cd.x;

    }


}

void filter_reduce(action_t action, float* xfft, float* h, float* yfft, 
        const int bcurr, const int parts, const int cfft,  int* params)
{
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    //lapt2 - "" for 2nd ""

    static int r1size = 0;
    static float *r1 = NULL;
    if(action==cleanup_mem){
        if(r1!=NULL){
            cudaFree(r1);
            r1 = NULL;
        }
        r1size = 0;
        return;
    }

    int block1 = params[0];
    int block2 = params[1];
    int lapt2 = params[2];


    const int N = parts;	//size of each reduction
    const int M = cfft;	//number of reductions (actually twice this because complex=float[2])

    //printf("input: %ix%i\n",M,N);

    dim3 dimBlock(block1,1);
    int temp = M/block1;
    dim3 dimGrid( temp  +  ((M == temp*block1) ? 0 : 1)  , N );

    //printf("1: %ix%i %ix%i\n",dimBlock.x,dimGrid.x,dimBlock.y, dimGrid.y);

    dim3 dimBlock2(block2,1); //
    temp = N/(block2*lapt2);
    dim3 dimGrid2( temp + (N == temp*block1*lapt2) ? 0 : 1 , 2*M);

    //printf("2: %ix%i %ix%i\n",dimBlock2.x,dimGrid2.x,dimBlock2.y, dimGrid2.y);

    //allocate memory for filter results
    if (r1size < 2*dimGrid.x*dimGrid.y){
        if(r1 != NULL)
            cudaFree(r1);
        r1size = 2*M*N; //complex=float[2]
        cudaMalloc((void**) &r1, sizeof(float)*r1size);
    }

    if (dimGrid2.x != 1){
        fprintf(stderr,"filter_reduce: dimGrid2.x != 1\n");
        exit(1);
    }

    double t0 = get_time_cuda();
    filter2D<<< dimGrid, dimBlock  >>>(xfft,h,r1,bcurr,parts,cfft);
    t0 = get_time_cuda() - t0;

    double t1 = get_time_cuda();
    switch (block2)
    {
        case 512:
            reduce2DStrided<512><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 256:
            reduce2DStrided<256><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 128:
            reduce2DStrided<128><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 64:
            reduce2DStrided<64><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 32:
            reduce2DStrided<32><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 16:
            reduce2DStrided<16><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
        case 8:
            reduce2DStrided<8><<< dimGrid2, dimBlock2, dimBlock2.x*sizeof(float) >>>(r1,yfft,dimGrid.y,2*M); break;
    }
    t1 = get_time_cuda() - t1;

    printf("filter: %f (%f.2%%), reduce: %f (%f.2%%)\n",t0,100*t0/(t0+t1), t1, 100*t1/(t0+t1));


}

void filter_reduce2(fftwf_complex* xfft, fftwf_complex* h, fftwf_complex* yfft, 
        const int bcurr, const int parts, const int cfft,  int* params)
{
    //memory allocated and not freed
    //block1 - block size for first reduction level
    //lapt1 - load/adds per thread for first red. lev.


    int block1 = params[0];
    int lapt1 = params[1];


    const int N = parts;	//size of each reduction
    const int M = cfft;	//number of reductions (actually twice this because complex=float[2])

    //printf("input: %ix%i\n",M,N);

    dim3 dimBlock(block1,1); //
    dim3 dimGrid((int)ceil((float)N/(block1*lapt1)), M);

    //printf("1: %i(%i)x%i %ix%i\n",dimBlock.x,lapt1,dimGrid.x,dimBlock.y, dimGrid.y);




    if(dimGrid.x != 1)
    { 
        printf("dimGrid.x != 1\n");
        exit(-1);
    }
    else
    {
        switch (block1)
        {
            case 512:
                filter2DStrided<512><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 256:
                filter2DStrided<256><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 128:
                filter2DStrided<128><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 64:
                filter2DStrided<64><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 32:
                filter2DStrided<32><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 16:
                filter2DStrided<16><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
            case 8:
                filter2DStrided<8><<< dimGrid, dimBlock, dimBlock.x*2*sizeof(float) >>>((float2*)xfft,(float2*)h,(float2*)yfft,bcurr,parts,cfft); break;
        }
    }


}



template <unsigned int blockSize>
__global__ void filter2DStrided(float2 *xfft, float2 *h, float2 *yfft, const int bcurr, const int parts, const int cfft)
{
//__global__ void filter2DStrided(float *g_idata, float *g_odata, int N, int stride){
    extern __shared__ float2 shared[];

    const unsigned int tid = threadIdx.x;

    int p = blockIdx.x*blockSize + threadIdx.x; //index of part
    const unsigned int element = blockIdx.y; //element within part, complex=float[2], blockSize.y = 1
    const unsigned int gridSize = blockSize*gridDim.x;



    shared[tid].x = 0;
    shared[tid].y = 0;
    while (p < parts)
    {
        const unsigned slot = (bcurr - p + parts) % parts;
       
        
        const float2 ab = xfft[element*parts+slot]; //non-strided
        const float2 cd = h[element*parts+p]; //non-strided


        shared[tid].x += ab.x*cd.x - ab.y*cd.y;
        shared[tid].y += ab.x*cd.y + ab.y*cd.x;

        p += gridSize; 
    }
    __syncthreads();

    //float *sdata = (float*)shared;


    // do reduction in shared mem
    // shared memory bank conflicts:
    //  convert shared to (float*) then reorganize accesses
    if (blockSize >= 512) { 
        if (tid < 256) {    
            shared[tid].x += shared[tid + 256].x;
            shared[tid].y += shared[tid + 256].y;
            //sdata[tid] += sdata[tid + 512];
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) {    
            shared[tid].x += shared[tid + 128].x;
            shared[tid].y += shared[tid + 128].y;
            //sdata[tid] += sdata[tid + 256];
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid < 64) {    
            shared[tid].x += shared[tid + 64].x;
            shared[tid].y += shared[tid + 64].y;
            //sdata[tid] += sdata[tid + 128];
        } 
        __syncthreads(); 
    }

    if (tid < 32) {
        if (blockSize >= 64) {  
            shared[tid].x += shared[tid + 32].x;
            shared[tid].y += shared[tid + 32].y;
            //sdata[tid] += sdata[tid + 32];
        }
        if (blockSize >= 32) {  
            shared[tid].x += shared[tid + 16].x;
            shared[tid].y += shared[tid + 16].y;
            //sdata[tid] += sdata[tid + 32];
        }
        if (blockSize >= 16) {  
            shared[tid].x += shared[tid + 8].x;
            shared[tid].y += shared[tid + 8].y;
            //sdata[tid] += sdata[tid + 16];
        }
        if (blockSize >= 8) {  
            shared[tid].x += shared[tid + 4].x;
            shared[tid].y += shared[tid + 4].y;
            //sdata[tid] += sdata[tid + 8];
        }
        if (blockSize >= 4) {  
            shared[tid].x += shared[tid + 2].x;
            shared[tid].y += shared[tid + 2].y;
            //sdata[tid] += sdata[tid + 4];
        }
        if (blockSize >= 2) {  
            shared[tid].x += shared[tid + 1].x;
            shared[tid].y += shared[tid + 1].y;
            //sdata[tid] += sdata[tid + 2];
        }

        // write result for this block to global mem
        int offset = blockIdx.x*gridDim.y;
        if (tid == 0) {     
            yfft[offset+element] = shared[0];
        }
    }

}




template <unsigned int blockSize>
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int N, int stride)
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockSize*2 + threadIdx.x;
    const unsigned int offset = blockIdx.y;
    const unsigned int gridSize = blockSize*2*gridDim.x;
    int n = N - blockSize;

    sdata[tid] = 0;

    while (i < n) { 
        sdata[tid] += g_idata[i*stride+offset] + g_idata[(i+blockSize)*stride+offset];
        i += gridSize; 
    }
    if(i<N)
        sdata[tid] += g_idata[i*stride+offset];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.y + blockIdx.x*gridDim.y] = sdata[0];
}




#endif
