#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>

#define TRIALS (10000)
#define THREADS (2)
#define NAIVE
//#define BROADCAST
//#define ATOMIC

double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

int checkSignals(int *signal, int N, int target)
{
    for (int i = 0; i < N; i++)
    {
        if (signal[i] != target)
            return 0;
    }
    return 1;
}

void naive_start();
void naive_sync();
void naive_finish();
int naive_wait(int tid);
void naive_signal(int tid);
void broadcast_start();
void broadcast_sync();
void broadcast_finish();
int broadcast_wait(int tid);
void broadcast_signal(int tid);
void atomic_start();
void atomic_sync();
void atomic_finish();
//atomic_wait == atomic_broadcast
void atomic_signal(int tid);

void* threadEntry(void *data);

int array[TRIALS];
int counter;
const int sync_target = THREADS;
pthread_mutex_t mutex[THREADS];
pthread_cond_t cond[THREADS];
pthread_cond_t main_cond;
pthread_mutex_t main_mutex;
int signal[THREADS];
int thrCounter[THREADS];

int main()
{

    

    int tid[THREADS];



    pthread_cond_init(&main_cond,NULL);
    pthread_mutex_init(&main_mutex,NULL);
    for (int i = 0; i < THREADS; i++)
    {
        pthread_mutex_init(&mutex[i],NULL);
        pthread_cond_init(&cond[i],NULL);
        signal[i] = 0;
        tid[i] = i;
        thrCounter[i] = 0;
    }

    pthread_t thr[THREADS];

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


    pthread_setschedparam(pthread_self(),SCHED_FIFO,&param);

    for (int i = 0; i < THREADS; i++)
    {
        pthread_create(&thr[i],&attr,threadEntry,(void*)&tid[i]);
    }

    double t = 0;
    t -= get_time();
    for (int j = 0; j < TRIALS; j++)
    {


#ifdef NAIVE
        naive_start();
        naive_sync();
#else
#ifdef BROADCAST
        broadcast_start();
        broadcast_sync();
#else
#ifdef ATOMIC
        atomic_start();
        atomic_sync();
#endif //ATOMIC
#endif //BROADCAST
#endif //NAIVE


    }
    t += get_time();
    
    // kill threads
#ifdef NAIVE
    naive_finish();
#else
#ifdef BROADCAST
    broadcast_finish();
#else
#ifdef ATOMIC
    atomic_finish();
#endif //ATOMIC
#endif //BROADCAST
#endif //NAIVE

    for (int i = 0; i < THREADS; i++)
        pthread_join(thr[i],NULL);


    printf("\ntime: %g\n\n",t);
    for (int i = 0; i < THREADS; i++)
        printf("%u: %u, ",i,thrCounter[i]);
    printf("\n\ntime: %g\n\n",t);


    for (int i = 0; i < THREADS; i++)
    {
        pthread_mutex_destroy(&mutex[i]);
        pthread_cond_destroy(&cond[i]);
    }
    pthread_cond_destroy(&main_cond);
    pthread_mutex_destroy(&main_mutex);


    return 0;

}

//ret = __sync_fetch_and_add(&counter,1);

void* threadEntry(void *data)
{
    const int tid = *(int*)data;

    while(1)
    {
#ifdef NAIVE
        int ret = naive_wait(tid);
        if (ret)
            break;

        thrCounter[tid]++;
        
        naive_signal(tid);

#else
#ifdef BROADCAST
        int ret = broadcast_wait(tid);
        if (ret)
            break;

        thrCounter[tid]++;

        broadcast_signal(tid);

#else
#ifdef ATOMIC

        int ret = broadcast_wait(tid);
        if (ret)
            break;

        thrCounter[tid]++;

        atomic_signal(tid);



#endif //ATOMIC
#endif //BROADCAST
#endif //NAIVE
    }

    return 0;
}


    
void naive_start()
{
    // start work threads
    for (int i = 0; i < THREADS; i++)
    {
        pthread_mutex_lock(&mutex[i]);
        signal[i] = 1;
        pthread_cond_signal(&cond[i]);
        pthread_mutex_unlock(&mutex[i]);
    }

}

void naive_sync()
{
    // sync threads
    for (int i = 0; i < THREADS; i++)
    {
        pthread_mutex_lock(&mutex[i]);
        while (signal[i] == 1)
            pthread_cond_wait(&cond[i],&mutex[i]);
        pthread_mutex_unlock(&mutex[i]);
    }
}

void broadcast_start()
{
    // start work threads
    pthread_mutex_lock(&mutex[0]);
    for (int i = 0; i < THREADS; i++)
    {
        signal[i] = 1;
    }
    pthread_cond_broadcast(&cond[0]);
    pthread_mutex_unlock(&mutex[0]);
}

void atomic_start()
{
    // start work threads
    pthread_mutex_lock(&mutex[0]);
    counter = 0;
    for (int i = 0; i < THREADS; i++)
    {
        signal[i] = 1;
    }
    pthread_cond_broadcast(&cond[0]);
    pthread_mutex_unlock(&mutex[0]);
}


void broadcast_sync()
{
    // sync threads
    pthread_mutex_lock(&mutex[0]);
    while (checkSignals(signal,THREADS,0) != 1)
    {
        //printf("main wait\n");
        pthread_cond_wait(&main_cond,&mutex[0]);
    }
    //printf("main awake!\n");
    pthread_mutex_unlock(&mutex[0]);
}

void atomic_sync()
{
    // sync threads
    if (counter != sync_target)
    {   
        pthread_mutex_lock(&main_mutex);
        while (counter != sync_target)
        {
            pthread_cond_wait(&main_cond,&main_mutex);
        }
        pthread_mutex_unlock(&main_mutex);
    }
}

void naive_finish()
{
    for (int i = 0; i < THREADS; i++)
    {
        pthread_mutex_lock(&mutex[i]);
        signal[i] = -1;
        pthread_cond_signal(&cond[i]);
        pthread_mutex_unlock(&mutex[i]);
    }
}
void broadcast_finish()
{
    pthread_mutex_lock(&mutex[0]);
    for (int i = 0; i < THREADS; i++)
    {
        signal[i] = -1;
    }
    pthread_cond_broadcast(&cond[0]);
    pthread_mutex_unlock(&mutex[0]);
}

void atomic_finish()
{
    pthread_mutex_lock(&mutex[0]);
    counter = 0;
    for (int i = 0; i < THREADS; i++)
    {
        signal[i] = -1;
    }
    pthread_cond_broadcast(&cond[0]);
    pthread_mutex_unlock(&mutex[0]);
}

int naive_wait(int tid)
{
    //wait for signal
    pthread_mutex_lock(&mutex[tid]);
    while (signal[tid] == 0)
        pthread_cond_wait(&cond[tid],&mutex[tid]);

    if (signal[tid] == -1)
    {
        pthread_mutex_unlock(&mutex[tid]);
        return 1;
    }
    pthread_mutex_unlock(&mutex[tid]);
    return 0;
}

void naive_signal(int tid)
{
    //signal completion
    pthread_mutex_lock(&mutex[tid]);
    signal[tid] = 0;
    pthread_cond_signal(&cond[tid]);
    pthread_mutex_unlock(&mutex[tid]);
}

int broadcast_wait(int tid)
{
    //wait for signal
    pthread_mutex_lock(&mutex[0]);
    while (signal[tid] == 0)
    {
        pthread_cond_wait(&cond[0],&mutex[0]);
    }

    if (signal[tid] == -1)
    {
        pthread_mutex_unlock(&mutex[0]);
        return 1;
    }
    pthread_mutex_unlock(&mutex[0]);
    return 0;
}


void broadcast_signal(int tid)
{
    //signal completion
    pthread_mutex_lock(&mutex[0]);
    signal[tid] = 0;
    if (checkSignals(signal,THREADS,0))
    {
        pthread_cond_signal(&main_cond);
    }
    pthread_mutex_unlock(&mutex[0]);
}

void atomic_signal(int tid)
{
    //signal completion
    __sync_sub_and_fetch (&signal[tid], 1);  //*signal[tid] = 0
    int ret = __sync_add_and_fetch (&counter, 1);  //counter += 1

    if (ret == sync_target)
    {
        // DO NEED MUTEX:
        // if no mutex, sometimes main thread will miss the signal below and never wake up.
        pthread_mutex_lock(&main_mutex);
        pthread_cond_signal(&main_cond);
        pthread_mutex_unlock(&main_mutex);
    }
}

