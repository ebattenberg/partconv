
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include "complexC.h"

#define MAX_TRIALS (100000000)
#define MAX_TIME (1.0)

#define NAIVE

double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

int main()
{

    int N[] = {16,32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    //, 131072,262144,524288,1048576};
    int sizeN = 13;
    int cfft[sizeN];
    double t[sizeN];
    for (int l = 0; l < sizeN; l++)
        cfft[l] = N[l]+1;

    FILE *fp = fopen("cmult.csv","w");

    for (int l = 0; l < sizeN; l++)
    {
        printf("\n\nTiming N = %u, cfft = %u\n",N[l],cfft[l]);
        float *a = new float[2*cfft[l]];
        float *b = new float[2*cfft[l]];
        float *c = new float[2*cfft[l]];


        int trial;
        double tavg;
        double tprev;
        float temp;

        fprintf(fp,"%u, ",N[l]);
        // COMPLEX_C ------------------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("COMPLEX_C VERSION: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;

            t[l] -= get_time();

            cmultComplexC(a,b,c,cfft[l]);

            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g\n",trial,t[l],tavg);
        fprintf(fp,"%g, ",tavg);
        tprev = tavg;

        /*
        // COMPLEX_C UNROLL ------------------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("COMPLEX_C UNROLL VERSION: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;

            t[l] -= get_time();

            cmultComplexCUnroll(a,b,c,cfft[l]);

            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t[l],tavg,100*tavg/tprev,100*tprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tprev = tavg;
        */

        //NAIVE VERSION 1 --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("NAIVE VERSION: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;


            const float *Aptr = (const float *)a;
            const float *Bptr = (const float *)b;
            float *Cptr = (float *)c;

            t[l] -= get_time();

            for (int i = 0; i < cfft[l]; i++)
            {
                // (a + b I) * (c + d I) = (ac - bd) + (ad + bc) I
                Cptr[2*i] += (Aptr[2*i] * Bptr[2*i]) - (Aptr[2*i+1] * Bptr[2*i+1]);
                Cptr[2*i+1] += (Aptr[2*i] * Bptr[2*i+1]) + (Aptr[2*i+1] * Bptr[2*i]);
            }
            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t[l],tavg,100*tavg/tprev,100*tprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tprev = tavg;
        
        //NAIVE UNROLL VERSION --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("NAIVE UNROLL VERSION: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;


            const float * Ar = (const float * )a;
            const float * Ai = (const float * )(a + 1);
            const float * Br = (const float * )b;
            const float * Bi = (const float * )(b+1);
            float * Cr = (float *)c;
            float * Ci= (float *)(c+1);

            t[l] -= get_time();

            for (int i = 0; i < cfft[l]-1; i+=4)
            {
                // (a + b I) * (c + d I) = (ac - bd) + (ad + bc) I
                *Cr += (*(Ar) * *(Br)) - (*(Ai) * *(Bi));
                *Ci += (*(Ar) * *(Bi)) + (*(Ai) * *(Br));

                Ar+=2; Ai+=2;
                Br+=2; Bi+=2;
                Cr+=2; Ci+=2;

                *Cr += (*(Ar) * *(Br)) - (*(Ai) * *(Bi));
                *Ci += (*(Ar) * *(Bi)) + (*(Ai) * *(Br));

                Ar+=2; Ai+=2;
                Br+=2; Bi+=2;
                Cr+=2; Ci+=2;

                *Cr += (*(Ar) * *(Br)) - (*(Ai) * *(Bi));
                *Ci += (*(Ar) * *(Bi)) + (*(Ai) * *(Br));

                Ar+=2; Ai+=2;
                Br+=2; Bi+=2;
                Cr+=2; Ci+=2;

                *Cr += (*(Ar) * *(Br)) - (*(Ai) * *(Bi));
                *Ci += (*(Ar) * *(Bi)) + (*(Ai) * *(Br));

                Ar+=2; Ai+=2;
                Br+=2; Bi+=2;
                Cr+=2; Ci+=2;
            }
            *Cr += (*(Ar) * *(Br)) - (*(Ai) * *(Bi));
            *Ci += (*(Ar) * *(Bi)) + (*(Ai) * *(Br));

            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t[l],tavg,100*tavg/tprev,100*tprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tprev = tavg;

        // SSE_ROLL VERSION --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("SSE_ROLL: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;


            const float *Aptr = (const float *)a;
            const float *Bptr = (const float *)b;
            float *Cptr = (float *)c;

            t[l] -= get_time();
            __m128 A, B, C, D;
            for (int i = 0; i < cfft[l]-1; i+=2)
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
            Cptr[0]  += (Aptr[0] * Bptr[0]) - (Aptr[1] * Bptr[1]);
            Cptr[1]  += (Aptr[0] * Bptr[1]) + (Aptr[1] * Bptr[0]); 
            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t[l],tavg,100*tavg/tprev,100*tprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tprev = tavg;



        //SSE_UNROLL VERSION --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < 2*cfft[l]; i++)
        {
            a[i] = temp;
            b[i] = temp;
            c[i] = temp;
        }
        t[l] = 0;
        printf("SSE_UNROLL: \n");
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t[l] > MAX_TIME)
                break;


            const float *Aptr = (const float *)a;
            const float *Bptr = (const float *)b;
            float *Cptr = (float *)c;

            t[l] -= get_time();

            __m128 A, B, C, D;
            for (int i = 0; i < cfft[l]-1; i+=8)
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
            Cptr[0]  += (Aptr[0] * Bptr[0]) - (Aptr[1] * Bptr[1]);
            Cptr[1]  += (Aptr[0] * Bptr[1]) + (Aptr[1] * Bptr[0]); 

            t[l] += get_time();
        }
        tavg = t[l]/trial;
        printf("trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t[l],tavg,100*tavg/tprev,100*tprev/tavg);
        fprintf(fp,"%g\n",tavg);


        delete[] a;
        delete[] b;
        delete[] c;

    }

    fclose(fp);

    return 0;

}
