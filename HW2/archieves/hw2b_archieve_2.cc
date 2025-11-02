/*
Revision : Use MPI + OpenMP and SIMD
Also use atomic operation to access next row

time 656.11
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
#include <mpi.h>
#include <omp.h>
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

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc != 9){
        if(rank == 0){
            fprintf(stderr, "Usage : %s output.png ......", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int cpu_avail = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", cpu_avail);

    const char* filename = argv[1];
    int iters = strtol(argv[2], nullptr, 10);
    double left = strtod(argv[3], nullptr);
    double right = strtod(argv[4], nullptr);
    double lower = strtod(argv[5], nullptr);
    double upper = strtod(argv[6], nullptr);
    int width = strtol(argv[7], nullptr, 10);
    int height = strtol(argv[8], nullptr, 10);

    vector<int> image(width * height);

    // row-partition
    int base = height / size;
    int rem = height % size;
    int row_start = rank * base + (rank < rem ? rank : rem);
    int local_rows = base + (rank < rem ? 1 : 0);
    int row_end = row_start + local_rows;

    vector<int> local(local_rows * width);
    const double dx = (right - left) / width;
    const double dy = (upper - lower) / height;

    atomic<int> next_row{0};
    const int chunk = 8;

    #pragma omp parallel
    {
        while(true){
            int jj0 = next_row.fetch_add(chunk, memory_order_relaxed);
            if(jj0 >= local_rows) break;
            int jj1 = min(jj0 + chunk, local_rows);

            for(int jj = jj0; jj < jj1; jj++){
                const int j_abs = row_start + jj;
                const double y0 = lower + j_abs * dy;
                int* rowp = local.data() + jj * width;

                int i = 0;
                for(; i+1 < width; i+= 2){
                    const double x0a = left + (i + 0) * dx;
                    const double x0b = left + (i + 1) * dx;

                    const __m128d vx0 = _mm_setr_pd(x0a, x0b);
                    const __m128d vy0 = _mm_set1_pd(y0);
                    const __m128d vfour = _mm_set1_pd(4.0);
                    const __m128d vone = _mm_set1_pd(1.0);
                    const __m128d viter = _mm_set1_pd((double)iters);

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

                        vx = _mm_or_pd(_mm_and_pd(active, xx), _mm_andnot_pd(active, vx));
                        vy = _mm_or_pd(_mm_and_pd(active, y_new), _mm_andnot_pd(active, vy));
                        vlen2 = _mm_or_pd(_mm_and_pd(active, len2_new), _mm_andnot_pd(active, vlen2));
                        vrep = _mm_add_pd(vrep, _mm_and_pd(active, vone));
                    }
                    const __m128i rep_i32 = _mm_cvtpd_epi32(vrep);
                    _mm_storel_epi64((__m128i*)(rowp + i), rep_i32);
                }

                for(; i < width; i++){
                    double x = 0.0, y = 0.0, len2 = 0.0;
                    const double x0 = left + i * dx;
                    int repeats = 0;
                    while(repeats < iters && len2 < 4.0){
                        const double xx = x * x - y * y + x0;
                        y = 2.0 * x * y + y0;
                        x = xx;
                        len2 = x * x + y * y;
                        repeats++;
                    }
                    rowp[i] = repeats;
                }
            }
        }
    }
       

    vector<int> recvcounts, displs;
    if(rank == 0){
        recvcounts.resize(size);
        displs.resize(size);

        for(int r = 0; r < size; r++){
            int r_base = height / size;
            int r_rem = height % size;
            int r_rows = r_base + (r < r_rem ? 1 : 0);
            int r_start = r * r_base + (r < r_rem ? r : r_rem);
            recvcounts[r] = r_rows * width;
            displs[r] = r_start * width;
        }
    }

    int mycount = local_rows * width;
    MPI_Gatherv(local.data(), mycount, MPI_INT, rank == 0 ? image.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr, 
                rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) write_png(filename, iters, width, height, image.data());

    MPI_Finalize();
    return 0;
}