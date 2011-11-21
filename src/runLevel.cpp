#include "partconvMulti.h"
#include <xmmintrin.h>
#include <pmmintrin.h>

int PartConv::runLevel(int L)
{

    int slot;

    start_timer(timer,kLockWait,L);
    memcpy(inbuffer[L].data, inbuffer[L].data+N[L], sizeof(float)*N[L]);
    Xbuf[L]->read(inbuffer[L].data + N[L]);
    stop_timer(timer,kLockWait,L);

    start_timer(timer,kFFT,L);
    //fft slot
    slot = bcurr[L] % num_parts[L];
    fftwf_execute(fwd[L]); // take fft of inbuffer, save in fftbuffer
    memcpy(Xfft[L].Cdata[slot], fftbuffer[L].Cdata, sizeof(fftwf_complex)*cfft[L]);
    stop_timer(timer,kFFT,L);

    start_timer(timer,kCMAdd,L);
    // reset Yfft to zeros
    memset(Yfft[L].Cdata, 0, sizeof(fftwf_complex)*cfft[L]);

    // do filtering
    for (int p = 0; p < num_parts[L]; p++) 
    {
        slot = (bcurr[L]-p + num_parts[L]) % num_parts[L];
        const float *Aptr = (const float *)Xfft[L].Cdata[slot];
        const float *Bptr = (const float *)H[L].Cdata[p];
        float *Cptr = (float *)Yfft[L].Cdata;

        
        __m128 A, B, C, D;
#ifndef _DONT_UNROLL_CMULT_LOOP
        for (int i = 0; i < cfft[L]-1; i+=8)
        {
            A = _mm_load_ps(Aptr);
            B = _mm_load_ps(Bptr);
            C = _mm_load_ps(Cptr);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+4);
            B = _mm_load_ps(Bptr+4);
            C = _mm_load_ps(Cptr+4);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+4, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+8);
            B = _mm_load_ps(Bptr+8);
            C = _mm_load_ps(Cptr+8);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+8, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+12);
            B = _mm_load_ps(Bptr+12);
            C = _mm_load_ps(Cptr+12);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+12, C);
            
            Aptr += 16;
            Bptr += 16;
            Cptr += 16;
        }
#else
        for (int i = 0; i < cfft[L]-1; i+=2)
        {
            A = _mm_load_ps(Aptr);
            B = _mm_load_ps(Bptr);
            C = _mm_load_ps(Cptr);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr, C);
            
            Aptr += 4;
            Bptr += 4;
            Cptr += 4;
        }
#endif
        Cptr[0]  += (Aptr[0] * Bptr[0]) - (Aptr[1] * Bptr[1]);
        Cptr[1]  += (Aptr[0] * Bptr[1]) + (Aptr[1] * Bptr[0]); 

/*        for (int i = 0; i < cfft[L]; i++)
            Cptr[i] += Aptr[i]*Bptr[i]; */
        /*
        for (int i = 0; i < cfft[L]; i++)
        {
            // (a + b I) * (c + d I) = (ac - bd) + (ad + bc) I
            Cptr[i][0] += (Aptr[i][0] * Bptr[i][0]) - (Aptr[i][1] * Bptr[i][1]);
            Cptr[i][1] += (Aptr[i][0] * Bptr[i][1]) + (Aptr[i][1] * Bptr[i][0]);
        }
        */
        
    }
    stop_timer(timer,kCMAdd,L);

    start_timer(timer,kIFFT,L);
    // take ifft of FDL
    fftwf_execute(bck[L]); //take ifft of Yfft, save in outbuffer
    stop_timer(timer,kIFFT,L);
    
    //copy output into double buffer
    start_timer(timer,kLockWait,L);
    memcpy(Ybuf[L]->getWriteBuffer(),outbuffer[L].data, N[L]*sizeof(float));
    stop_timer(timer,kLockWait,L);

    bcurr[L]++;

    return 0;

}
