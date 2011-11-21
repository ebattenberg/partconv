#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>

#define TRIALS (10000000)
#define USE_ATOMICS


double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

void* threadEntry(void *data);

int *array;
int counter;
int *thrCounter;
pthread_mutex_t mutex;
pthread_cond_t cond;
int goFlag;

int main()
{

    const int numThreads = 2;

    int tid[numThreads];
    int thrCounterData[numThreads];
    pthread_mutex_init(&mutex,NULL);
    pthread_cond_init(&cond,NULL);
    //int arrayData[TRIALS*numThreads];
    //array = arrayData;
    thrCounter = thrCounterData;
    counter = 0;
    goFlag = 0;

    pthread_t thr[numThreads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    struct sched_param param;
    pthread_attr_getschedparam(&attr, &param);
#ifdef __APPLE__
    const int actual_max_priority = 63;
    param.sched_priority = actual_max_priority;
#else
    param.sched_priority = 99;
#endif
    pthread_attr_setschedparam(&attr, &param);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);


    for (int i = 0; i < numThreads; i++)
    {
        tid[i] = i;
        thrCounter[i] = 0;
        pthread_create(&thr[i],&attr,threadEntry,(void*)&tid[i]);
    }

    pthread_mutex_lock(&mutex);
    goFlag = 1;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

    double t = 0;

    t -= get_time();

    for (int i = 0; i < numThreads; i++)
        pthread_join(thr[i],NULL);
    t += get_time();



    printf("\ntime: %g\n\n",t);
    //for (int i = 0; i < TRIALS*numThreads; i++)
    //{
    //    printf("%d",array[i]);
    //}
    printf("\n\n");
    for (int i = 0; i < numThreads; i++)
        printf("%u: %u, ",i,thrCounter[i]);
    printf("\n\n");
#ifdef USE_ATOMICS
    printf("using atomics\n");
#else
    printf("using mutexes\n");
#endif
    printf("\n\ntime: %g\n\n",t);


    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);


    return 0;

}


void* threadEntry(void *data)
{
    const int tid = *(int*)data;
    pthread_mutex_lock(&mutex);
    while (goFlag < 1)
        pthread_cond_wait(&cond,&mutex);
    pthread_mutex_unlock(&mutex);
    //printf("\ntid: %u awake!\n",tid);


#ifdef USE_ATOMICS
    int ret;
    while(1)
    {
        ret = __sync_fetch_and_add(&counter,1);
        //array[ret] = tid;
        thrCounter[tid]++;
        if (thrCounter[tid] == TRIALS)
            break;
    }

#else
    while(1)
    {
        pthread_mutex_lock(&mutex);
        //array[counter] = tid;
        counter++;
        thrCounter[tid]++;
        if (thrCounter[tid] == TRIALS)
        {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
    }
#endif
    return 0;
}


    




