#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include "fftw3.h"

#define MAX_TRIALS (10000000)
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

    if (FILE* wisdomfile = fopen("wisdom.wis","r"))
    {
        fftwf_import_wisdom_from_file(wisdomfile);
        fclose(wisdomfile);
    }
 

    const int N[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    const int sizeN = 13;
    int cfft[sizeN];
    int nfft[sizeN];
    double t;
    double tavg;
    double tfprev;
    double tbprev;

    for (int l = 0; l < sizeN; l++)
    {
        cfft[l] = N[l]+1;
        nfft[l] = N[l]*2;
    }

    FILE *fp = fopen("fftw.csv","w");

    for (int l = 0; l < sizeN; l++)
    {
        printf("\n\nTiming N = %u, cfft = %u\n",N[l],cfft[l]);
        float *a = new float[nfft[l]];
        fftwf_complex *b = (fftwf_complex*)fftwf_malloc(cfft[l]*sizeof(fftwf_complex));

        fftwf_plan fwd;
        fftwf_plan bck;

        int trial;
        float temp;

        fprintf(fp,"%u, ",N[l]);

        //ESTIMATE --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < nfft[i]; i++)
            a[i] = temp;
        float* bptr = (float*)b;
        for (int i = 0; i < 2*cfft[i]; i++)
            bptr[i] = temp;

        printf("ESTIMATE: \n");
        fwd = fftwf_plan_dft_r2c_1d(nfft[l],a,b,FFTW_ESTIMATE);
        bck = fftwf_plan_dft_c2r_1d(nfft[l],b,a,FFTW_ESTIMATE);

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(fwd);
            t += get_time();
        }
        tavg = t/trial;
        printf("FWD trials: %u, total time: %5g, avg time: %5g\n",trial,t,tavg);
        fprintf(fp,"%g, ",tavg);
        tfprev = tavg;

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(bck);
            t += get_time();
        }
        tavg = t/trial;
        printf("BCK trials: %u, total time: %5g, avg time: %5g\n",trial,t,tavg);
        fprintf(fp,"%g, ",tavg);
        tbprev = tavg;

        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bck);

        //MEASURE --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < nfft[i]; i++)
            a[i] = temp;
        bptr = (float*)b;
        for (int i = 0; i < 2*cfft[i]; i++)
            bptr[i] = temp;

        printf("MEASURE: \n");
        fwd = fftwf_plan_dft_r2c_1d(nfft[l],a,b,FFTW_MEASURE);
        bck = fftwf_plan_dft_c2r_1d(nfft[l],b,a,FFTW_MEASURE);

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(fwd);
            t += get_time();
        }
        tavg = t/trial;
        printf("FWD trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t,tavg,100*tavg/tfprev,100*tfprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tfprev = tavg;

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(bck);
            t += get_time();
        }
        tavg = t/trial;
        printf("BCK trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t,tavg,100*tavg/tbprev,100*tbprev/tavg);
        fprintf(fp,"%g, ",tavg);
        tbprev = tavg;

        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bck);

        //PATIENT --------------- ---------------
        temp = (float)get_time();
        for (int i = 0; i < nfft[i]; i++)
            a[i] = temp;
        bptr = (float*)b;
        for (int i = 0; i < 2*cfft[i]; i++)
            bptr[i] = temp;

        printf("PATIENT: \n");
        fwd = fftwf_plan_dft_r2c_1d(nfft[l],a,b,FFTW_PATIENT);
        bck = fftwf_plan_dft_c2r_1d(nfft[l],b,a,FFTW_PATIENT);

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(fwd);
            t += get_time();
        }
        tavg = t/trial;
        printf("FWD trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t,tavg,100*tavg/tfprev,100*tfprev/tavg);
        fprintf(fp,"%g, ",tavg);

        t = 0;
        for (trial = 0; trial < MAX_TRIALS; trial++)
        {

            if (t > MAX_TIME)
                break;

            t -= get_time();
            fftwf_execute(bck);
            t += get_time();
        }
        tavg = t/trial;
        printf("BCK trials: %u, total time: %5g, avg time: %5g (%5g,%5g)\n",trial,t,tavg,100*tavg/tbprev,100*tbprev/tavg);
        fprintf(fp,"%g\n",tavg);

        fftwf_destroy_plan(fwd);
        fftwf_destroy_plan(bck);

        delete[] a;
        fftwf_free(b);

    }

    if (FILE* wisdomfile = fopen("wisdom.wis","w"))
    {
        fftwf_export_wisdom_to_file(wisdomfile);
        fclose(wisdomfile);
    }
    

    fclose(fp);

    return 0;

}
