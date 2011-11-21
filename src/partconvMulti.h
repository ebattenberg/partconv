#ifndef __partconvMulti_h__
#define __partconvMulti_h__

#include "partconv.h"




class PartConvMulti;
class PartConvTest;
class Waiter;
//class Timer;

/*
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
        void tally(PartConvFilter* pc);
        void reset();
        void display(int FS, int* N);

        double *lockWait;
        double *fft;
        double *ifft;
        double *cmadd;
        unsigned *count;
        int M;
};
*/

 /*
 * implemented outside of timer class
 * to allow passing of NULL timers when timing isn't needed
 */
/*
int start_timer(Timer* timer, TimerType type, int ind);
int stop_timer(Timer* timer, TimerType type, int ind);
*/



class PartConvMulti
{
    public: 

        //methods
        PartConvMulti(int inVerbosity = 0) :  verbosity(inVerbosity) {}
        ~PartConvMulti() {}
        int setup(const char * config_file_name, int max_threads_per_level = 1, int max_level0_threads = 0, double min_work = 0);
        int cleanup();
        virtual int run(float* const output, const float* const input, const int frameCount);
        void reset();

        float* output_mix;
        float* input_mix;

        int FS; // sample rate
        int buffer_size;
        int max_block_size;
        int verbosity; // = {0,1,2} amount of info printed, default:1

        int numInputChannels;
        int numOutputChannels;

        unsigned long frameNum; 
        Waiter *doneWaiter;
        unsigned long lastFrame;



	
    protected:

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
        //int max_threads_per_level;  //number of work threads per level
        int *threads_in_level;  //number of worker threads per level
        int total_level0_work_units;  //number of work units at FDL level 0
        int first_thread_level; // {0,1} lowest FDL level that gets separate worker threads


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
            int ind;            //index of PartConvFilter instance
            PartConvFilter *obj;      //pointer to PartConvFilter instance
            double time;        //measured work time
            WorkList *next;     //pointer to next list member   

        } ***workList;          // workList[level][thread] is pointer to WorkList
};

class PartConvMultiRelaxed : public PartConvMulti
{
    public:
        PartConvMultiRelaxed(int inVerbosity = 0) : PartConvMulti(inVerbosity) {}
        ~PartConvMultiRelaxed() {}
        virtual int run(float* const output, const float* const input, const int frameCount);
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
