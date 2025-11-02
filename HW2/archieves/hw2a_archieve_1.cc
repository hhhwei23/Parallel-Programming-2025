/*
Revision :

worker -> completed

time 730.9

worker2 -> some testcase WA (1-4% difference)
*/

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

using namespace std;

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
};

static void* worker(void* arg){
    Task* t = (Task*) arg;

    const double dx = (t->right - t->left) / t->width;
    const double dy = (t->upper - t->lower) / t->height;

    while(true){
    //    int j0, j1;
    //    pthread_mutex_lock(t->mtx);
        int j0 = t->next_row->fetch_add(t->chunk, memory_order_relaxed);
    //    j0 = *(t->next_row);
       if(j0 >= t->height){
        // pthread_mutex_unlock(t->mtx);
        break;
       }
       int j1 = min(j0 + t->chunk, t->height);

    //    j1 = j0 + t->chunk;
    //    if(j1 > t->height) j1 = t->height;
    //    *(t->next_row) = j1;
    //    pthread_mutex_unlock(t->mtx);

       for(int j = j0; j < j1; j++){
        const double y0 = t->lower + j * dy;
        // const int base = j * t->width;
        int* rowp = t->image + j * t->width;
        // double x0 = t->left;
        for(int i = 0 ; i < t->width; i++){
            const double x0 = t->left + i * dx;

            int repeats = 0;
            double x = 0.0, y = 0.0, len2 = 0.0;

            while(repeats < t->iters && len2 < 4.0){
                const double xx = x*x - y * y + x0;
                y = 2.0 * x * y + y0;
                x = xx;
                len2 = x * x + y * y;
                repeats++;
            }

            rowp[i] = repeats;
        }
       }
    }

    return nullptr;
}

static void* worker2(void* argv){
    Task* t = (Task*) argv;

    if(t->cpu_avail > 0){
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(t->tid % t->cpu_avail, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    const int width = t->width;
    const int height = t->height;
    const int iters = t->iters;
    int* const image = t->image;
    const double left = t->left;
    const double lower = t->lower;
    const double dx = (t->right - t->left) / t->width;
    const double dy = (t->upper - t->lower) / t->height;

    const __m128d four = _mm_set1_pd(4.0);
    const __m128d iters_v = _mm_set1_pd((double)iters);
    const __m128d one_pd = _mm_set1_pd(1.0);
    const __m128i one_i = _mm_castpd_si128(one_pd);


    for(int j = t->tid; j < height; j += t->T){
        const double y0 = lower + j * dy;
        int* rowp = image + j * width;

        // SIMD
        int i = 0;
        for(; i + 1 < width; i += 2){
            const double x0_scalar = left + i * dx;

            const __m128d x0v = _mm_setr_pd(x0_scalar, x0_scalar + dx);
            const __m128d y0v = _mm_set1_pd(y0);

            __m128d x = _mm_setzero_pd();
            __m128d y = _mm_setzero_pd();
            __m128d rep = _mm_setzero_pd();
            __m128d len2 = _mm_setzero_pd();

            while(true){
                const __m128d can_iter_pd = _mm_and_pd(_mm_cmplt_pd(rep, iters_v), _mm_cmplt_pd(len2, four));
                if(_mm_movemask_pd(can_iter_pd) == 0) break;

                const __m128d xx = _mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
                const __m128d xy = _mm_mul_pd(x, y);
                
                x = _mm_add_pd(xx, x0v);
                y = _mm_add_pd(_mm_add_pd(xy, xy), y0v);
                
                const __m128i can_i = _mm_castpd_si128(can_iter_pd);
                const __m128i inc_i = _mm_and_si128(can_i, one_i);
                rep = _mm_add_pd(rep, _mm_castsi128_pd(inc_i));

                len2 = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
            }

            __m128i reps_i32 = _mm_cvttpd_epi32(rep);
            alignas(16) int tmp[4];
            _mm_storeu_si128((__m128i*)tmp, reps_i32);
            rowp[i] = tmp[0];
            rowp[i+1] = tmp[1];
        }

        if(i < width){
            double x0 = left + i * dx;
            int repeats = 0;
            double x = 0.0, y = 0.0, len2 = 0.0;
            while(repeats < iters && len2 < 4.0){
                const double xx = x*x - y*y + x0;
                y = 2.0 * x * y + y0;
                x = xx;
                len2 = x*x + y*y;
                ++repeats;
            }
            rowp[i] = repeats;
        }
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
    const int chunk = 1;
    // int next_row = 0;
    atomic<int> next_row{0};

    // pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    vector<pthread_t> th(T);
    vector<Task> tk(T);

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

    write_png(filename, iters, width, height, image.data());
}