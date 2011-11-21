
#include <pthread.h>

#if  defined(__APPLE__)
#define pthread_barrier_t barrier_t
#define pthread_barrier_attr_t barrier_attr_t
#define pthread_barrier_init(b,a,n) barrier_init(b,n)
#define pthread_barrier_destroy(b) barrier_destroy(b)
#define pthread_barrier_wait(b) barrier_wait(b)
#endif

typedef struct {
    int needed;
    int called;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} barrier_t;

int barrier_init(barrier_t *barrier,int needed);
int barrier_destroy(barrier_t *barrier);
int barrier_wait(barrier_t *barrier);



unsigned levelOn(unsigned* ptr, unsigned L);
unsigned levelOff(unsigned* ptr, unsigned L);
unsigned checkLevel(unsigned var, unsigned L);
void AtomicSet(unsigned *ptr, unsigned new_value);
