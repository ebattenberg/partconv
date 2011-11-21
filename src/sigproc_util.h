#ifndef __sigproc_util__
#define __sigproc_util__

#ifdef __cplusplus
extern "C" {
#endif 

#include "sndfile.h"


#ifdef _SNDFILE_
typedef struct sndFileData{
  SNDFILE *sndFile;
  SF_INFO sfInfo;
  int position;
} sndFileData;
#endif

typedef struct vec{
    float* data;
    int size;
} vec;

typedef struct Cvec{
    fftwf_complex* Cdata;
    int size;
} Cvec;

typedef struct matrix{
    float** data;
    int rows;
    int cols;
} matrix;

typedef struct Cmatrix{
    fftwf_complex** Cdata;
    int rows;
    int cols;
} Cmatrix;


double get_time();
int start_timer(double* timers, int ind);
int stop_timer(double* timers, int ind);

#ifdef _SNDFILE_
vec read_mono_wav(const char* filename, const int fs, const int frameCount);
int write_mono_wav(vec sig, const char* filename, int fs);
void write_wav(const char* filename, vec data, double samplerate, int channels);
#endif

Cvec zero_pad_complex(Cvec vector, int padded_length, int offset);
Cmatrix splitArrayComplex(Cvec x, int rowlength,int paddedRowLength,int offset);

matrix splitArray(vec x, int rowLength, int paddedRowLength, int offset); 
vec zero_pad(vec vector, int padded_length, int offset);
int nextpow2(int x);
int log2i(int x);

Cmatrix create_Cmatrix(int rows, int cols);
matrix create_matrix(int rows, int cols);
Cvec create_Cvec(int size);
vec create_vec(int size);

void reset_Cmatrix(Cmatrix A);
void reset_matrix(matrix A);
void reset_Cvec(Cvec A);
void reset_vec(vec A);

void free_vec(vec* x);
void free_Cvec(Cvec* x);
void free_matrix(matrix* A);
void free_Cmatrix(Cmatrix* A);

void malloc_check(void* ptr);
float error_norm(vec x, vec y, float scale);

void write_matrix(matrix A, const char* file);
void write_vec(vec A, const char* file);

void naive_convolution(float *y, float *x, float *h, int x_len, int h_len);


#ifdef __cplusplus
}
#endif //__cplusplus
#endif //__sigproc_util__
