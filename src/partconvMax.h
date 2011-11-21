#ifndef __partconvMax_h__
#define __partconvMax_h__


#ifndef _PORTAUDIO_
#define printf post
#include "ext.h"
#else
#include<stdio.h>
#endif

#include<stdlib.h>
#include <string.h>
#include <math.h>

#include <pthread.h>
//#include <semaphore.h>


#include "fftw3.h"
#include "buffers.h"


class PartConvFilter;
class PartConvMax;
class DelayBuffer;
class DoubleBuffer;





class PartConvFilter 
{
    public: 

        //methods
        PartConvFilter(int inVerbosity = 0, int inSampleRate = 44100) : FS(inSampleRate), verbosity(inVerbosity) {}
        ~PartConvFilter() {}
        int cleanup(void);
        int setup( 
                int M_in, int* N_in, int* num_parts_in, 
                Vector *impulse = NULL,
                int fftwOptLevel = 1, //FFTW_MEASURE
                const char* wisdom = NULL
                );
        //int run(float* const output, const float* const input);
        double bench(const int L, const int trials = 100, const int pollute_every = 0, 
                const float max_time = 0.1, double *benchData = NULL);
        void reset();
        void sync_levels();
        void input(const float* const input);
        void output(float* const output);

        //members
        int M; // number of partition levels
        int *N; // int[L]: length of block at level L
        int *num_parts; // int[L]: number of partitions at level L
        int FS; // sample rate
        int verbosity; // = {0,1,2} amount of info printed, default:1
        //float scaleFactor;
        //int inputChannel;
        //int outputChannel;
        //int muted;



        //float* output_mix;
        //float* input_mix;
	
    private:
        // methods
        //static void *runLevelThreadEntry(void *arg);
        int runLevel(int L);
        friend class PartConvMax;

        // members
        int *nfft; // int[L]: fft length at level L
        int *cfft; // int[L]: elements in output of real fft at level L
        unsigned *bcurr; // bcurr[L]: current block at level L
        int *num_N0_in_level; // int[L]: num blocks of size N[0] in N[L]
        int *ind_N0_in_level; // int[L]: index of subblock of size N[0] within block of size N[L]

        int start_sync_depth;
        int end_sync_depth;

        DoubleBuffer** Ybuf;
        DelayBuffer** Xbuf;

        ComplexVectorArray *Xfft; 
        ComplexVectorArray *H;
        Vector *inbuffer, *outbuffer;
        ComplexVector *Yfft, *fftbuffer;

        fftwf_plan *fwd, *bck;

        pthread_t *pcth;
        int *pcth_state; //pcth_state[L] = {-1,0,1}, -1=exit, 0=stop, 1=run
        //pthread_mutex_t *pcth_mutex;
        pthread_cond_t *worker_cond;
        pthread_mutex_t levels_mutex;
        pthread_cond_t *main_cond;
        int first_thread_level;
};

class PartConvMax
{
    public: 

        //methods
        PartConvMax(int inVerbosity = 0) :  verbosity(inVerbosity) {}
        ~PartConvMax() {}
        int setup(int inSampleRate, int n, int m, int k, float* impulses, int filterLength, int stride, int* blockSizes, int levels, int fftwOptLevel, const char* wisdom = NULL);
        int cleanup();
        int run(float** const output, float** const input);
        void reset();

        float** outbuffers;
        float** inbuffers;

        int FS; // sample rate
        int buffer_size;
        int max_block_size;
        int verbosity; // = {0,1,2} amount of info printed, default:1

        int numInputChannels;
        int numOutputChannels;

        unsigned long frameNum; 


	
    private:

        static void *workerThreadEntry(void *arg);


        // members

        PartConvFilter *pc;


        int numPCs;
        int numRepeats;

        int *inputChannel;
        int *outputChannel;
        int *muteChannel;
        float *scaleFactor;

        int num_levels;         //number of distinct partitioning levels in all pc instances
        unsigned *level_counter;     //keeps track of where current buffer lies in a level's block size
        unsigned *level_size;        //block size of level in multiples of buffer size
        int **level_map;        //level_map[pc][level]: how each global level maps to the levels within a pc instance   
        int max_threads_per_level;  //number of work threads per level
        int *threads_in_level;  //number of worker threads per level

        int first_thread_level; //lowest FDL level that gets separate worker threads


        int *thread_counter;    //thread_counter[sync_depth]: decremented when each work thread finishes
        int *sync_target;

        pthread_t **worker_thread;
        int **worker_state; //work_state[level][thread]: {-1,0,1}, -1=exit, 0=stop, 1=run
        unsigned terminate_flag;
        pthread_cond_t *worker_cond;
        pthread_mutex_t *worker_mutex;
        pthread_cond_t *main_cond;
        pthread_mutex_t *main_mutex;

        struct WorkerThreadData // for passing both arguments to a pthread
        {
            PartConvMax *This; // pointer to "this"
            int levelNum; // level number to run
            int threadNum; // worker thread to run

        } **workerThreadData;   // workerThreadData[level][thread] is WorkerThreadData 

        struct WorkList         // linked list for per thread work
        {
            WorkList(WorkList *head = NULL) { next = head; };

            int arg;            //level to run 
            int ind;            //index of PartConvFilter instance
            PartConvFilter *obj;      //pointer to PartConvFilter instance
            double time;        //measured work time
            WorkList *next;     //pointer to next list member   

        } ***workList;          // workList[level][thread] is pointer to WorkList
};


class DelayBuffer
{
    public:
        DelayBuffer(int frameSize, int blockSize, int delay);
        ~DelayBuffer();
        void write(const float* const x);
        void read(float* y);
        void prepNextRead();
        void reset();

    private:
        float *bufferMem;
        float **buffer;
        int bufferSize;
        int frameSize, framesPerBlock, numBlocks;
        int readBlock, writeBlock, writeFrame;
        int readFrameDelay;
};
        

class DoubleBuffer
{
    public:
        DoubleBuffer(int inSize);
        ~DoubleBuffer();
        void swap();
        float* getReadBuffer();
        float* getWriteBuffer();
        void reset();

    private:
        float *readBuffer, *writeBuffer;
        int size;
};



#endif
