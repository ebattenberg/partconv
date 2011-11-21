#ifndef __partconv_h__
#define __partconv_h__

#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

#include <pthread.h>
#include <semaphore.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _PORTAUDIO_
extern "C" {
#include "portaudio.h"
#ifndef __APPLE__
#include "pa_linux_alsa.h"
#endif
}
#endif
//extern "C" {
//#include "/usr/include/complex.h"
//}
#include "fftw3.h"
#ifdef _SNDFILE_
#include "sndfile.h"
#endif

#include "sigproc_util.h"
//#include "threading.h"

#ifdef _CUDA_
#include "cufft.h"
#endif

class PartConv;
class PartConvMulti;
#ifdef _SNDFILE_
class IOData;
#endif
class DelayBuffer;
class DoubleBuffer;
class Waiter;

enum TimerType
{
    kLockWait,
    kFFT,
    kIFFT,
    kCMAdd
};

class Timer
{
    
    public:
        Timer(int inM);
        ~Timer();
        void tally(PartConv* pc);
        void reset();
        void display(int FS, int* N);

        double *lockWait;
        double *fft;
        double *ifft;
        double *cmadd;
        unsigned *count;
        int M;
};

/*
 * implemented outside of timer class
 * to allow passing of NULL timers when timing isn't needed
 */
int start_timer(Timer* timer, TimerType type, int ind);
int stop_timer(Timer* timer, TimerType type, int ind);


#ifdef _SNDFILE_
vec readImpulseResponse(const char* filename,int verbosity=0);
#endif

class PartConv 
{

    public: 

        //methods
        PartConv(int inVerbosity = 0, int inSampleRate = 44100) : FS(inSampleRate), verbosity(inVerbosity) {}
        ~PartConv() {}
        int cleanup(void);
        int setup( 
                int M_in, int* N_in, int* num_parts_in, 
                vec *impulse = NULL,
                bool inCreateThreads = false,
                double* benching = NULL, //if used, ptr to double[M] (externally allocated)
                Timer* inTiming = NULL //if used, ptr to double[M+2] (externally allocated)
                );
        int run(float* const output, const float* const input);
        double bench(const int L, const int trials = 100, const int pollute_every = 0, 
                const float max_time = 0.1, double *benchData = NULL);
        void reset();
        void sync_levels();
        void input(const float* const input);
        void start();
        void end();
        void output(float* const output);

        //members
        int M; // number of partition levels
        int *N; // int[L]: length of block at level L
        int *num_parts; // int[L]: number of partitions at level L
        int FS; // sample rate
        int verbosity; // = {0,1,2} amount of info printed, default:1
        double *t; //double[M] extra benching array
        Timer *timer; //for profiling, externally allocated
        float scaleFactor;
        int inputChannel;
        int outputChannel;
        int muted;

        DelayBuffer** Xbuf;

#ifdef _SNDFILE_
        IOData* io; // class for buffering disk read/writes
#endif
#ifdef _PORTAUDIO_
        PaStream* stream;
#endif

        float* output_mix;
        float* input_mix;
	
    private:
        // methods
        static void *runLevelThreadEntry(void *arg);
        int runLevel(int L);
        friend class PartConvMulti;
        friend class PartConvMax;

        // members
        int *nfft; // int[L]: fft length at level L
        int *cfft; // int[L]: elements in output of real fft at level L
        unsigned *bcurr; // bcurr[L]: current block at level L
        int *num_N0_in_level; // int[L]: num blocks of size N[0] in N[L]
        int *ind_N0_in_level; // int[L]: index of subblock of size N[0] within block of size N[L]

        int start_sync_depth;
        int end_sync_depth;

        Cmatrix *Xfft, *H;
        DoubleBuffer** Ybuf;
        vec *inbuffer, *outbuffer;
        Cvec *Yfft, *fftbuffer;

        fftwf_plan *fwd, *bck;

        pthread_t *pcth;
        int *pcth_state; //pcth_state[L] = {-1,0,1}, -1=exit, 0=stop, 1=run
        //pthread_mutex_t *pcth_mutex;
        pthread_cond_t *worker_cond;
        pthread_mutex_t levels_mutex;
        pthread_cond_t *main_cond;
        int first_thread_level;

        struct PartConvLevel // for passing both arguments to a pthread
        {
            PartConv *This; // pointer to "this"
            int L; // level number to run
        } *pcl;

        //unsigned runningBits;  // bits hold state of worker threads
        unsigned terminate_flag;
        bool createThreads;

#ifdef _CUDA_
        Cvec *Xfft_d, *H_d;
        vec *inbuffer_d, *outbuffer_d;
        Cvec *Yfft_d, *fftbuffer_d;
        int *useGPU;

        cufftHandle *fwd_d, *bck_d;

        int setupGPU(int L);
        int cleanupGPU(int L);
        int runLevelGPU(int L);
#endif
        friend class Timer;
};

class PartConvMulti
{

    public: 

        //methods
#ifdef _SNDFILE_
        PartConvMulti(int inVerbosity = 0, int inSampleRate = 44100) :  FS(inSampleRate), verbosity(inVerbosity), io(NULL) {}
#else
        PartConvMulti(int inVerbosity = 0, int inSampleRate = 44100) :  FS(inSampleRate), verbosity(inVerbosity) {}
#endif
        ~PartConvMulti() {}
        int cleanup(void);
        int setup( const char* config_file, int threads_per_level = 1, double min_work = 0);
        int run(float* const output, const float* const input, const int frameCount);
        void reset();

        //members
        double *t; //double[M] extra benching array
        Timer *timer; //for profiling, externally allocated

        int FS; // sample rate
        int buffer_size;
        int max_block_size;
        int verbosity; // = {0,1,2} amount of info printed, default:1

        int numInputChannels;
        int numOutputChannels;

        unsigned long frameNum; 
        Waiter *doneWaiter;
        unsigned long lastFrame;



#ifdef _SNDFILE_
        IOData* io; // class for buffering disk read/writes
#endif
#ifdef _PORTAUDIO_
        PaStream* stream;
#endif

        float* output_mix;
        float* input_mix;

	
    protected:

        static void *workerThreadEntry(void *arg);


        // members

        PartConv *pc;


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
            PartConvMulti *This; // pointer to "this"
            int levelNum; // level number to run
            int threadNum; // worker thread to run

        } **workerThreadData;   // workerThreadData[level][thread] is WorkerThreadData 

        struct WorkList         // linked list for per thread work
        {
            WorkList(WorkList *head = NULL) { next = head; };

            int arg;            //level to run 
            int ind;            //index of PartConv instance
            PartConv *obj;      //pointer to PartConv instance
            double time;        //measured work time
            WorkList *next;     //pointer to next list member   

        } ***workList;          // workList[level][thread] is pointer to WorkList



};

class PartConvMax: public PartConvMulti
{
    public:
        PartConvMax(int inVerbosity = 0, int inSampleRate = 44100) : PartConvMulti(inVerbosity,inSampleRate) {}
        ~PartConvMax() {}
        int setup(int n, int m, int k, float* impulses, int filterLength, int stride, int* blockSizes, int levels);
        int cleanup();
        int run(float** const output, float** const input);

        float** outbuffers;
        float** inbuffers;



};




#ifdef _SNDFILE_
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
#endif

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

class Waiter
{
    public:
        Waiter(int inVal = 0);
        ~Waiter();
        void waitFor(int target);
        void changeAndSignal(int newVal);

    private:
        pthread_cond_t cond;
        pthread_mutex_t mutex;
        int val;
        int initVal;
};


#endif
