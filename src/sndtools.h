#ifndef __sndtools_h__
#define __sndtools_h__

#include "stdlib.h"

#include "sndfile.h"
#include "buffers.h"

struct sndFileData
{
  SNDFILE *sndFile;
  SF_INFO sfInfo;
  int position;
};

int readImpulseResponse(Vector & v, const char* filename, int verbosity = 0);
void write_wav(const char* filename, vec data, double samplerate, int channels);

/*
class IOData
{
    public:
        IOData();
        ~IOData();
        IOData* setup(PartConvMulti* pc, const char* infilename, const char* outfilename);
        int cleanup();
        int run(const float* const output);
        float* getInput();

        sndFileData* infile; //input audio file
        sndFileData* outfile; //output audio file

        int num_output_blocks;
        int num_input_blocks;
        int block_size;
        int io_ind;

        int verbosity;

    private:
        // input buffers
        vec xbuf, xread;
        vec xtemp;

        // output buffers
        vec ybuf, ywrite;

        pthread_t ith;
        pthread_t oth;  
        pthread_attr_t ioth_attr;

        pthread_mutex_t ith_mutex;
        pthread_cond_t ith_cond;
        pthread_mutex_t oth_mutex;
        pthread_cond_t oth_cond;

        int ith_state;
        int oth_state;

        static void *readChunkThreadEntry(void *arg);
        int readChunk(void);
        static void *writeChunkThreadEntry(void *arg);
        int writeChunk(void);
};
*/


#endif //__sndtools_h__
