#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>

#define TRIALS (100000)
//#define USE_ATOMICS


double get_time()
{
    //output time in seconds
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}


pthread_mutex_t mutex;
pthread_cond_t cond;

int main()
{
    pthread_mutex_init(&mutex,NULL);
    pthread_cond_init(&cond,NULL);


    double t = -get_time();
    for (int i = 0; i < TRIALS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&mutex);
    }
    t += get_time();
    printf("signal time: %g (%gms per)\n",t,1000*t/TRIALS);

    t = -get_time();
    for (int i = 0; i < TRIALS; i++)
    {
        pthread_mutex_lock(&mutex);
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex);
    }
    t += get_time();
    printf("broadcast time: %g (%gms per)\n",t,1000*t/TRIALS);



    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);




}

