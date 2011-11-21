#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "complex.h"
#include "fftw3.h"

#define N 10000
#define ITERS 1000

double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

int main()
{

    fftwf_complex *a,*b,*c;
    fftwf_complex sum = 0;

    a = (fftwf_complex*)malloc(N*sizeof(fftwf_complex));
    b = (fftwf_complex*)malloc(N*sizeof(fftwf_complex));
    c = (fftwf_complex*)malloc(N*sizeof(fftwf_complex));




    int i;
    for (i = 0; i < N; i++)
    {
        a[i] = i + I*i;
        b[i] = i - I*i;
    }

    double t = get_time();
    for (int iter = 0; iter < ITERS; iter++)
    {
        for (i = 0; i < N; i++)
        {
           c[i]  += a[i] * b[i];
        }
        sum = 0;
        for (i = 0; i < N; i++)
            sum += c[i];
    }
    t = get_time() - t;
    

    printf("time: %g\n",t);







    free(a);
    free(b);
    free(c);


}
