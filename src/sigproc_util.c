#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#ifdef _SNDFILE_
#include "sndfile.h"
#endif 
#include "fftw3.h"

#include "sigproc_util.h"

double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}


Cmatrix splitArrayComplex(Cvec x, int rowLength,int paddedRowLength,int offset)
{
    fftwf_complex** split;
    int numRows = x.size/rowLength;
    if (x.size % numRows != 0)
        numRows++;
    int index;
    split = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*)*numRows);
    malloc_check(split);
    int i;
    for (i = 0; i < numRows; i++) {
        Cvec in, out;
        index = i*rowLength;
        in.Cdata = x.Cdata+index;
        in.size = rowLength;
        out = zero_pad_complex(in,paddedRowLength,offset);
        split[i] = out.Cdata;
    }
    Cmatrix A;
    A.rows = numRows;
    A.cols = paddedRowLength;
    A.Cdata = split;
    return A;
}

matrix splitArray(vec x, int rowLength, int paddedRowLength, int offset) 
{
    // for no padding, set rowLength == paddedRowLength,  offset == 0
    float** split;
    int numRows = x.size/rowLength;
    if (x.size % numRows != 0) 
        numRows++;

    int index;
    split = (float**)fftwf_malloc(sizeof(float*)*numRows);
    malloc_check(split);
    int i;
    for (i = 0; i < numRows; i++) {
        vec in, out;
        index = i*rowLength;

        in.data = x.data+index;
        in.size = rowLength;

        out = zero_pad(in, paddedRowLength, offset);
        split[i] = out.data;
    }

    matrix A;
    A.rows = numRows;
    A.cols = paddedRowLength;
    A.data = split;
    return A;
}



#ifdef _SNDFILE_
void write_wav(const char* filename, vec x, double samplerate, int channels) 
{
    // we strip metadata from the input wav files when we save out...
    SNDFILE *outfile;
    SF_INFO sfinfo;

    sfinfo.samplerate = samplerate;
    sfinfo.channels = channels;
    // SF_FORMAT_FLOAT is not what we want for 16 bit wav
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    if (!(outfile = sf_open(filename,SFM_WRITE, &sfinfo))) {
        printf("Unable to open output wav file: %s\n",filename);
        sf_perror(NULL);
        exit(-1);
    }
    sf_write_float(outfile, x.data, x.size);
    sf_close(outfile);

    printf("wrote %s\n",filename); 
}

vec read_mono_wav(const char* filename, const int fs, const int frameCount)
{
    sndFileData data;
    data.sndFile = sf_open(filename,SFM_READ,&data.sfInfo);
    if (data.sfInfo.samplerate != fs)
        printf("read_mono_wav: samplerate is incorrect\n");

    if (!data.sndFile) {
        fprintf(stderr,"Can't open sound file: %s\n",filename);
        sf_perror(NULL);
        exit(-1);
    }
    int numFrames;
    if (frameCount == -1)
        numFrames =  data.sfInfo.frames;
    else
        numFrames = frameCount;

    int numChannels = data.sfInfo.channels;

    float* out_array = (float*)malloc(numFrames*sizeof(float));

    int i;
    if (numChannels > 1){ 
        //remove extra channels
        float* temp = (float*)malloc(numChannels*numFrames*sizeof(float));
        sf_readf_float(data.sndFile,temp,numFrames);

        for(i=0;i<numFrames;i++)
            out_array[i] = temp[numChannels*i];
        free(temp);
    }
    else
        sf_readf_float(data.sndFile,out_array,numFrames);

    sf_close(data.sndFile);
    vec vector;
    vector.size = numFrames;
    vector.data = out_array;

    return vector;
}

int write_mono_wav(vec sig, const char* filename, int fs)
{
    //data returned as (*data)[channel][frame]
    SNDFILE *sndFile;
    SF_INFO sfInfo;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    sfInfo.channels = 1;
    sfInfo.samplerate = fs;

    sndFile = sf_open(filename,SFM_WRITE,&sfInfo);

    int err = sf_error(sndFile);
    if (err) {
        fprintf(stderr,"write_mono_wav: Can't open file: %s, Error: %u\n",filename,err);
        return 0;
    }


    sf_writef_float(sndFile,sig.data,sig.size);
    sf_close(sndFile);

    return sig.size;
}
#endif

int nextpow2(int y) 
{
    unsigned x;
    if (y >= 0)
        x = (unsigned)y;
    else 
        return -1;
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return (int)(x + 1);

}

int log2i(int x)
{
    if (x <= 0)
        return -1;

    int isPow2 = x && !( (x-1) & x);
    if (!isPow2)
    {
        fprintf(stderr, "log2i: %d is not a positive power of 2\n",x);
        return -1;
    }

    int pow = 0;
    unsigned y = (unsigned) x;
    while (y > 1)
    {
        y >>= 1;
        pow++;
    }
    return pow;
}
            



Cvec zero_pad_complex(Cvec vector, int padded_length, int offset)
{
    //doesn't destroy input vector

    if(padded_length < vector.size){
        fprintf(stderr,"zero_pad_complex: pad_length is shorter than array length\n");
        exit(-1);
    }
    if(padded_length - vector.size < offset){
        fprintf(stderr,"zero_pad_complex: not enoug hpadding for offset\n");
        exit(-1);
    }

    fftwf_complex* padded = (fftwf_complex*)fftwf_malloc(padded_length*sizeof(fftwf_complex)); 
    malloc_check(padded);

    // zero memory (0.0f is equiv to all zero bytes)
    memset(padded,0,padded_length*sizeof(fftwf_complex));
    memcpy(padded+offset, vector.Cdata, sizeof(fftwf_complex)*vector.size);

    Cvec padded_vector;
    padded_vector.Cdata = padded;
    padded_vector.size = padded_length;

    return padded_vector;
}

vec zero_pad(vec vector, int padded_length, int offset)
{
    //doesn't destroy input vector

    if(padded_length < vector.size){
        fprintf(stderr,"zero_pad: pad_length is shorter than array length\n");
        exit(-1);
    }
    if(padded_length - vector.size < offset){
        fprintf(stderr,"zero_pad: not enough padding for offset\n");
        exit(-1);
    }


    float* padded = (float*)fftwf_malloc(padded_length*sizeof(float)); malloc_check(padded);
    malloc_check(padded);

    // zero memory (0.0f is equiv to all zero bytes)
    memset(padded,0,padded_length*sizeof(float));
    memcpy(padded+offset, vector.data, sizeof(float)*vector.size);

    vec padded_vector;
    padded_vector.data = padded;
    padded_vector.size = padded_length;

    return padded_vector;
}

vec create_vec(int size)
{
    // create vector and set all elements to zero
    vec vector;
    vector.size = size;
    vector.data = (float*)fftwf_malloc(size*sizeof(float));
    malloc_check(vector.data);

    memset(vector.data, 0, sizeof(float)*size);

    return vector;
}

matrix create_matrix(int rows, int cols)
{
    matrix A;
    A.rows = rows;
    A.cols = cols;

    A.data = (float**)fftwf_malloc(sizeof(float*)*rows);
    malloc_check(A.data);
    int i;
    for (i = 0; i < rows; i++) {
        vec a_row = create_vec(cols);
        A.data[i] = a_row.data;
    }

    return A;
}

Cvec create_Cvec(int size)
{
    // create complex vector and set all values to zero
    Cvec vector;
    vector.size = size;
    vector.Cdata = (fftwf_complex*)fftwf_malloc(size*sizeof(fftwf_complex));
    malloc_check(vector.Cdata);

    memset(vector.Cdata, 0, sizeof(fftwf_complex)*size);

    return vector;
}

Cmatrix create_Cmatrix(int rows, int cols)
{
    Cmatrix A;
    A.rows = rows;
    A.cols = cols;

    A.Cdata = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*)*rows);
    malloc_check(A.Cdata);
    int i;
    for (i = 0; i < rows; i++) {
        Cvec a_row = create_Cvec(cols);
        A.Cdata[i] = a_row.Cdata;
    }

    return A;

}

void reset_vec(vec vector)
{
    memset(vector.data, 0, vector.size*sizeof(float));
}

void reset_Cvec(Cvec vector)
{
    memset(vector.Cdata, 0, vector.size*sizeof(fftwf_complex));
}

void reset_matrix(matrix A)
{
    int row;
    for (row = 0; row < A.rows; row++)
        memset(A.data[row],0,A.cols*sizeof(float));
}

void reset_Cmatrix(Cmatrix A)
{
    int row;
    for (row = 0; row < A.rows; row++)
        memset(A.Cdata[row],0,A.cols*sizeof(fftwf_complex));
}


void malloc_check(void* ptr)
{
    if(ptr == NULL){
        fprintf(stderr,"malloc error\n");
        exit(-1);
    }
}

float error_norm(vec x, vec y, float scale)
{
    int i;
    float s = 0;
    float diff;

    if(x.size != y.size){
        fprintf(stderr,"error_norm: size mismatch\n");
        exit(-1);
    }
    int N = x.size;

    for(i=0;i<N;i++){
        diff = x.data[i] - y.data[i]/scale;
        s += diff*diff;
    }

    return sqrt(s)/N;
}

void free_vec(vec* x)
{
    if(x->data != NULL)
        fftwf_free(x->data);
    x->data = NULL;
    x->size = 0;
}

void free_matrix(matrix* A)
{
    if(A->data != NULL){
        int i;
        for (i = 0; i < A->rows; i++)
            fftwf_free(A->data[i]);
    }
    fftwf_free(A->data);
    A->rows = 0;
    A->cols = 0;
}

void free_Cmatrix(Cmatrix* A)
{
    if(A->Cdata != NULL){
        int i;
        for (i = 0; i < A->rows; i++)
            fftwf_free(A->Cdata[i]);
    }
    fftwf_free(A->Cdata);
    A->rows = 0;
    A->cols = 0;
}

void free_Cvec(Cvec* x)
{
    if(x->Cdata != NULL)
        fftwf_free(x->Cdata);
    x->Cdata = NULL;
    x->size = 0;
}

void write_matrix(matrix A, const char* file)
{
    //write matrix to file using row-major order
    //dimensions are written as leading ints
    //(so we can look at it in matlab (see bin_interface.m in ../matlab)

    FILE* fp;    
    size_t count;

    fp = fopen(file,"wb");
    count = fwrite(&A.rows,sizeof(int),1,fp); 
    count = fwrite(&A.cols,sizeof(int),1,fp); 

    int i;
    for(i = 0; i < A.rows; i++){
        count = fwrite(A.data[i],sizeof(float),A.cols,fp);
        if(count < A.cols)
            fprintf(stderr,"write_matrix: fwrite error\n");
    }
    fclose(fp);

    printf("wrote %s [%ix%i]\n",file,A.rows,A.cols); 
}

void write_vec(vec A, const char* file)
{

    FILE* fp;    
    size_t count;

    fp = fopen(file,"wb");
    count = fwrite(&A.size,sizeof(int),1,fp); 

    count = fwrite(A.data,sizeof(float),A.size,fp);
    if(count < A.size)
        fprintf(stderr,"write_matrix: fwrite error\n");
    fclose(fp);

    printf("wrote %s [%i]\n",file,A.size);
}

void naive_convolution(float *y, float *x, float *h, int x_len, int h_len)
{
    const int print_interval = x_len/100;
    int count = 0;

    memset(y, 0, (x_len+h_len-1)*sizeof(float));
    for (int i=0;i<x_len;i++)
    {
        if (count++ == print_interval)
        {
            printf("naive_convolution: step %u of %u\n",i,x_len);
            count = 0;
        }
        float *yy = &y[i];
        float xi  = x[i];
        for (int j=0;j<h_len;j++)
            yy[j] += xi * h[j];
    }
}
