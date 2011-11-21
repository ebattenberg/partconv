
#include "threading.h"


int barrier_init(barrier_t *barrier,int needed)
{
    barrier->needed = needed;
    barrier->called = 0;
    pthread_mutex_init(&barrier->mutex,NULL);
    pthread_cond_init(&barrier->cond,NULL);
    return 0;
}

int barrier_destroy(barrier_t *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
    return 0;
}
int barrier_wait(barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    barrier->called++;
    if (barrier->called == barrier->needed) {
        barrier->called = 0;
        pthread_cond_broadcast(&barrier->cond);
    } else {
        pthread_cond_wait(&barrier->cond,&barrier->mutex);
    }
    pthread_mutex_unlock(&barrier->mutex);
    return 0;
}


unsigned levelOn(unsigned* ptr, unsigned L)
{
    //flips bit L on
    //return __sync_or_and_fetch(ptr, 1 << L); //atomic
    *ptr |= (1 << L);
    return *ptr;
}

unsigned levelOff(unsigned* ptr, unsigned L)
{
    // flips bit L off
    //return __sync_and_and_fetch(ptr, ~(1 << L)); //atomic
    
    *ptr &= ~(1 << L);
    return *ptr;  
}

unsigned checkLevel(unsigned var, unsigned L)
{
    //checks bit L
    return (var & (1 << L)) > 0;
}

void AtomicSet(unsigned *ptr, unsigned new_value)
{
        while (true)
        {
                unsigned old_value = *ptr;
                if (__sync_bool_compare_and_swap(ptr, old_value, new_value)) 
                    return;
        }
}




