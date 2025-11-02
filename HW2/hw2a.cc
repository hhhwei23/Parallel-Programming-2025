#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <bits/stdc++.h>
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <atomic>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace std;

static inline double wall_time_now(){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static inline double thread_cpu_now(){
    struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

struct Task{
    int iters, width, height;
    double left, right, lower, upper;
    int* image;
    // int* next_row;
    atomic<int>* next_row;
    int chunk;
    // pthread_mutex_t* mtx;
    int cpu_avail, tid;
    int T; // # of total threads;
    double wall_s = 0.0;
    double cpu_s = 0.0;
    long long rows = 0;
    long long pixels = 0;
    long long iters_sum = 0;
    int core_first = -1;
};


static void* worker2(void* argv){
    Task* t = (Task*) argv;

    const double dx = (t->right - t->left) / t->width;
    const double dy = (t->upper - t->lower) / t->height;

    while(true){
        int j0 = t->next_row->fetch_add(t->chunk, memory_order_relaxed);
        if(j0 >= t->height) break;
        int j1 = min(j0 + t->chunk, t->height);

        double w0 = wall_time_now();
        double c0 = thread_cpu_now();
        if(t->core_first < 0) t->core_first = sched_getcpu();

        for(int j = j0; j<j1; j++){
            const double y0 = t->lower + j * dy;
            int* rowp = t->image + j * t->width;
            
            long long iter_sum_row = 0;

            int i = 0;
            for(; i+1 < t->width; i += 2){
                const double x0a = t->left + (i + 0) * dx;
                const double x0b = t->left + (i + 1) * dx;

                const __m128d vx0 = _mm_setr_pd(x0a, x0b);
                const __m128d vy0 = _mm_set1_pd(y0);
                const __m128d vfour = _mm_set1_pd(4.0);
                const __m128d vone = _mm_set1_pd(1.0);
                const __m128d viter = _mm_set1_pd((double) t->iters);

                __m128d vx = _mm_setzero_pd();
                __m128d vy = _mm_setzero_pd();
                __m128d vlen2 = _mm_setzero_pd();
                __m128d vrep = _mm_setzero_pd();

                while(true){
                    const __m128d cond1 = _mm_cmplt_pd(vrep, viter);
                    const __m128d cond2 = _mm_cmplt_pd(vlen2, vfour);
                    const __m128d active = _mm_and_pd(cond1, cond2);

                    if(_mm_movemask_pd(active) == 0) break;

                    __m128d xx = _mm_sub_pd(_mm_mul_pd(vx, vx), _mm_mul_pd(vy, vy));
                    xx = _mm_add_pd(xx, vx0);

                    const __m128d xy = _mm_mul_pd(vx, vy);
                    const __m128d y_new = _mm_add_pd(_mm_add_pd(xy, xy), vy0);

                    const __m128d len2_new = _mm_add_pd(_mm_mul_pd(xx, xx), _mm_mul_pd(y_new, y_new));

                    vx    = _mm_or_pd(_mm_and_pd(active, xx), _mm_andnot_pd(active, vx));
                    vy    = _mm_or_pd(_mm_and_pd(active, y_new), _mm_andnot_pd(active, vy));
                    vlen2 = _mm_or_pd(_mm_and_pd(active, len2_new), _mm_andnot_pd(active, vlen2));
                    vrep  = _mm_add_pd(vrep, _mm_and_pd(active, vone));
                }

                const __m128i rep_i32 = _mm_cvtpd_epi32(vrep);
                _mm_storel_epi64((__m128i*)(rowp + i), rep_i32);
                iter_sum_row += rowp[i] + rowp[i+1];
            }
            for (; i < t->width; ++i){
                const double x0 = t->left + i * dx;
                int repeats = 0;
                double x = 0.0, y = 0.0, len2 = 0.0;
                while (repeats < t->iters && len2 < 4.0){
                    const double xx = x*x - y*y + x0;
                    y = 2.0 * x * y + y0;
                    x = xx;
                    len2 = x*x + y*y;
                    ++repeats;
                }
                rowp[i] = repeats;
                iter_sum_row += repeats;
            }

            t->rows += 1;
            t->pixels += t->width;
            t->iters_sum += iter_sum_row;
        }

        double c1 = thread_cpu_now();
        double w1 = wall_time_now();
        t->cpu_s += (c1 - c0);
        t->wall_s += (w1 - w0);
    }

    return nullptr;
}

int main(int argc, char** argv){
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int cpu_avail = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", cpu_avail);

    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], nullptr, 10);
    double left = strtod(argv[3], nullptr);
    double right = strtod(argv[4], nullptr);
    double lower = strtod(argv[5], nullptr);
    double upper = strtod(argv[6], nullptr);
    int width = strtol(argv[7], nullptr, 10);
    int height = strtol(argv[8], nullptr, 10);

    vector<int> image(width * height);

    int T = min<int>(cpu_avail > 0 ? cpu_avail : 1, height);

    if(const char* env = getenv("NTHREADS")){
        int want = atoi(env);
        if(want >= 1) T = min(want, height);
    }
    const int chunk = 1;
    // int next_row = 0;
    atomic<int> next_row{0};

    // pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    vector<pthread_t> th(T);
    vector<Task> tk(T);

    double t_total0 = wall_time_now();
    double t_comp0 = wall_time_now();

    for(int t = 0; t < T; t++){
        tk[t].iters = iters; tk[t].width = width; tk[t].height = height;
        tk[t].left = left; tk[t].right = right; tk[t].lower = lower; tk[t].upper = upper;
        tk[t].image = image.data();
        tk[t].next_row = &next_row;
        tk[t].chunk = chunk;
        // tk[t].mtx = &mtx;
        tk[t].cpu_avail = cpu_avail;
        tk[t].tid = t;
        tk[t].T = T;
        pthread_create(&th[t], nullptr, worker2, &tk[t]);
    }

    for(int t = 0; t < T; t++) pthread_join(th[t], nullptr);

    double t_comp = wall_time_now() - t_comp0;
    double t_io0 = wall_time_now();
    write_png(filename, iters, width, height, image.data());
    double t_io = wall_time_now() - t_io0;
    double t_total = wall_time_now() - t_total0;

    /*
    FILE* sf = fopen("prof_summary.csv", "w");
    if (sf) {
        fprintf(sf, "threads,width,height,iters,left,right,lower,upper,total_s,compute_s,io_s\n");
        fprintf(sf, "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.6f,%.6f,%.6f\n",
                T, width, height, iters, left, right, lower, upper,
                t_total, t_comp, t_io);
        fclose(sf);
    }

    FILE* tf = fopen("prof_threads.csv", "w");
    if (tf) {
        fprintf(tf, "thread,core,wall_s,cpu_s,rows,pixels,iters\n");
        for (int t = 0; t < T; ++t) {
            fprintf(tf, "%d,%d,%.6f,%.6f,%lld,%lld,%lld\n",
                t, tk[t].core_first, tk[t].wall_s, tk[t].cpu_s,
                (long long)tk[t].rows, (long long)tk[t].pixels, (long long)tk[t].iters_sum);
        }
        fclose(tf);
    }

    printf("hw2a,%d,%d,%d,%.6f,%.6f,%.6f\n", T, width, height, t_comp, t_io, t_total);
    fflush(stdout);
    */

    return 0;
}