
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<errno.h>

#include "partconvMulti.h"

#define OUTPUT_NAME "./output.wav"
#define INPUT_NAME "../audio/input.wav"
#define IMPULSE_NAME "../reverbs/impulse.wav"
#define FS 44100

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))




int main(int argc, char** argv) 
{

    vec x = read_mono_wav(INPUT_NAME,FS,5*FS);
    vec h = read_mono_wav(IMPULSE_NAME,FS,-1);
    vec y = create_vec(x.size+h.size);

    naive_convolution(y.data,x.data,h.data,x.size,h.size);

    write_mono_wav(y,OUTPUT_NAME,FS);

    
    
    return 0;
}

