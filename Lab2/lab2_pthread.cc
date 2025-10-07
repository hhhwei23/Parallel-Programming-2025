#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <unistd.h>

typedef unsigned long long ull;

typedef struct {
    ull r;
    ull k;
    ull x_lo, x_hi;
    ull partial_mod;
} Task;

static inline ull umin(ull a, ull b){
    return a < b ? a : b;
}

static void* worker(void* arg){
    Task* t = (Task*)arg;
    const long double rr = (long double)t->r;
    ull local = 0;

    for(ull x = t->x_lo; x<t->x_hi; x++){
        long double dx = (long double)x;
        long double val = rr*rr - dx*dx;
        if(val < 0) val = 0;
        ull y = (ull)ceil(sqrtl(val));
        local += y;
    }
    t->partial_mod = (t->k ? (local % t->k) : local);
    return NULL;
}

int main(int argc, char** argv){
    char *ep = NULL;
    ull r = strtoull(argv[1], &ep, 10);
    ull k = strtoull(argv[2], &ep, 10);

    int T = (argc >= 4) ? atoi(argv[3]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if(T <= 0) T = 1;
    if((ull)T > r && r > 0) T = (int)r;

    ull base = r / (ull)T, rem = r % (ull)T, cur = 0;

    pthread_t *ths = (pthread_t *)malloc(sizeof(pthread_t) *T);
    Task *tsk = (Task*) malloc(sizeof(Task)*T);
    
    for(int i=0; i<T; i++){
        ull len = base + (i < (int)rem ? 1ull : 0ull);
        tsk[i].r = r; tsk[i].k = k;
        tsk[i].x_lo = cur;
        tsk[i].x_hi = cur + len;
        cur += len;
        tsk[i].partial_mod = 0;
        pthread_create(&ths[i], NULL, worker, &tsk[i]);
    }

    ull quad_mod = 0;
    for(int i=0; i<T; i++){
        pthread_join(ths[i], NULL);
        if(k){
            quad_mod += tsk[i].partial_mod;
            quad_mod %= k;
        }
        else{
            quad_mod += tsk[i].partial_mod;
        }
    }

    ull ans = k ? ((4ull * (quad_mod % k)) % k) : (4ull * quad_mod);
    printf("%llu\n", ans);

    return 0;
}