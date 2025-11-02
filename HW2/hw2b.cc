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
#include <xmmintrin.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

// 656.11 -> 441.46

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
    // printf("%d cpus available\n", cpu_avail);

    double T_total0 = MPI_Wtime();
    double T_compute = 0.0, T_comm = 0.0, T_post = 0.0, T_io = 0.0;

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
    // int base = height / size;
    // int rem = height % size;
    // int row_start = rank * base + (rank < rem ? rank : rem);
    // int local_rows = base + (rank < rem ? 1 : 0);
    int local_rows = (height - rank + size - 1)/size;
    if(local_rows < 0) local_rows = 0;
    vector<int> local(local_rows * width);

    const double dx = (right - left) / width;
    const double dy = (upper - lower) / height;

    double T_comp0 = MPI_Wtime();

    int T = omp_get_max_threads();
    vector<double> thr_wall(T, 0.0), thr_cpu(T, 0.0);
    vector<long long> thr_pixels(T, 0), thr_iters(T, 0);
    vector<int> thr_core(T, -1);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double wall_local = 0.0, cpu_local = 0.0;
        long long pixels_local = 0, iters_local = 0;
        int core_first = -1;
        #pragma omp for schedule(dynamic)
        for (int jj = 0; jj < local_rows; ++jj) {
            double w0 = wall_time_now();
            double c0 = thread_cpu_now();
            if(core_first < 0) core_first = sched_getcpu();

            const int j_abs = rank + jj * size;
            const double y0 = lower + j_abs * dy;
            int* rowp = local.data() + jj * width;

            int i = 0;
            const __m128d vfour = _mm_set1_pd(4.0);
            const __m128d vy0   = _mm_set1_pd(y0);
            const __m128d vone  = _mm_set1_pd(1.0);
            const __m128d viter = _mm_set1_pd((double)iters);

            long long iter_sum_row = 0;

            for (; i + 1 < width; i += 2) {
                const double x0a = left + (i + 0) * dx;
                const double x0b = left + (i + 1) * dx;
                const __m128d vx0 = _mm_setr_pd(x0a, x0b);

                __m128d vx = _mm_setzero_pd(), vy = _mm_setzero_pd();
                __m128d vlen2 = _mm_setzero_pd(), vrep = _mm_setzero_pd();

                while (true) {
                    // scalr : (repeats < iters && len2 < 4.0)
                    const __m128d cond1  = _mm_cmplt_pd(vrep,  viter);
                    const __m128d cond2  = _mm_cmplt_pd(vlen2, vfour);
                    const __m128d active = _mm_and_pd(cond1, cond2);
                    if (_mm_movemask_pd(active) == 0) break;

                    /*
                    xx = x * x;
                    xy = x * y;
                    y_new = 2 * xy + y0;
                    x_new = xx ;
                    len2new = x_new*x_new + y_new*y_new;
                    */

                    __m128d xx = _mm_sub_pd(_mm_mul_pd(vx, vx), _mm_mul_pd(vy, vy));
                    xx = _mm_add_pd(xx, vx0);

                    const __m128d xy      = _mm_mul_pd(vx, vy);
                    const __m128d y_new   = _mm_add_pd(_mm_add_pd(xy, xy), vy0);
                    const __m128d len2new = _mm_add_pd(_mm_mul_pd(xx, xx), _mm_mul_pd(y_new, y_new));
                    
                    // scaler : if(active) {x = x_new; y = y_new; len2 = len2new; repeats++;}
                    vx = _mm_or_pd(_mm_and_pd(active, xx), _mm_andnot_pd(active, vx));
                    vy = _mm_or_pd(_mm_and_pd(active, y_new), _mm_andnot_pd(active, vy));
                    vlen2 = _mm_or_pd(_mm_and_pd(active, len2new), _mm_andnot_pd(active, vlen2));
                    vrep = _mm_add_pd(vrep, _mm_and_pd(active, vone));
                }
                
                // scalar : rowp[i] = repeats(i), rowp[i+1] = repeats(i+1)
                const __m128i rep_i32 = _mm_cvtpd_epi32(vrep);
                _mm_storel_epi64((__m128i*)(rowp + i), rep_i32);

                iter_sum_row += rowp[i] + rowp[i+1];
            }

            for (; i < width; ++i) {
                double x = 0.0, y = 0.0, len2 = 0.0;
                const double x0 = left + i * dx;
                int repeats = 0;
                while (repeats < iters && len2 < 4.0) {
                    const double xx = x*x - y*y + x0;
                    y = 2.0 * x * y + y0;
                    x = xx;
                    len2 = x*x + y*y;
                    ++repeats;
                }
                rowp[i] = repeats;
                iter_sum_row += repeats;
            }

            pixels_local += width;
            iters_local += iter_sum_row;

            double c1 = thread_cpu_now();
            double w1 = wall_time_now();

            cpu_local += (c1 - c0);
            wall_local += (w1 - w0);
        }

        thr_core[tid] = core_first;
        thr_wall[tid] = wall_local;
        thr_cpu[tid] = cpu_local;
        thr_pixels[tid] = pixels_local;
        thr_iters[tid] = iters_local;
    }

    T_compute += MPI_Wtime() - T_comp0;


    int mycount = local_rows * width;

    vector<int> recvcounts, displs, tmp;
    if (rank == 0) {
        recvcounts.resize(size);
        for (int r = 0; r < size; ++r) {
            int rows_r = (r < height) ? ((height - 1 - r) / size + 1) : 0;
            recvcounts[r] = rows_r * width;
        }
        displs.resize(size); displs[0] = 0;
        for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
        tmp.resize(width * height);
    }

    double tc0 = MPI_Wtime();

    MPI_Gatherv(local.data(), mycount, MPI_INT,
                rank == 0 ? tmp.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, MPI_COMM_WORLD);

    T_comm += MPI_Wtime() - tc0;

    double tp0 = MPI_Wtime();
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < size; ++r) {
            int rows_r = recvcounts[r] / width;
            for (int k = 0; k < rows_r; ++k) {
                int j = r + k * size;
                memcpy(&image[j * width], &tmp[off + k * width],
                            sizeof(int) * width);
            }
            off += recvcounts[r];
        }
    }
    T_post += MPI_Wtime() - tp0;

    if(rank == 0){
        double tio0 = MPI_Wtime();
        write_png(filename, iters, width, height, image.data());
        T_io += (MPI_Wtime() - tio0);
    }

    /*
    double T_total = MPI_Wtime() - T_total0;

    double io_max = 0.0;
    double post_max = 0.0;
    double comm_sum = 0.0, comp_sum = 0.0, total_max = 0.0;

    MPI_Reduce(&T_io, &io_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&T_post, &post_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&T_comm, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&T_compute, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&T_total, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    long long rank_iters = 0, rank_pixels = 0;
    for(int t = 0; t<(int)thr_iters.size(); t++){
        rank_iters += thr_iters[t];
        rank_pixels += thr_pixels[t];
    }

    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int hostlen = 0;
    MPI_Get_processor_name(host, &hostlen);


    const int HN = MPI_MAX_PROCESSOR_NAME;
    double my_times[5] = {T_total, T_compute, T_comm, T_post, T_io};
    int my_threads = omp_get_max_threads();
    int my_rows = local_rows;

    vector<double> all_times(5 * size);
    vector<int> all_threads, all_rows;
    vector<long long> all_iters, all_pixels;
    vector<char> all_hosts;

    if(rank == 0){
        all_times.resize(size * 5);
        all_threads.resize(size);
        all_rows.resize(size);
        all_iters.resize(size);
        all_pixels.resize(size);
        all_hosts.resize(size * HN);
    }

    MPI_Gather(my_times, 5, MPI_DOUBLE,
            rank==0 ? all_times.data() : nullptr, 5, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
    MPI_Gather(&my_threads, 1, MPI_INT,
            rank==0 ? all_threads.data() : nullptr, 1, MPI_INT,
            0, MPI_COMM_WORLD);
    MPI_Gather(&my_rows, 1, MPI_INT,
            rank==0 ? all_rows.data() : nullptr, 1, MPI_INT,
            0, MPI_COMM_WORLD);
    MPI_Gather(&rank_iters, 1, MPI_LONG_LONG,
            rank==0 ? all_iters.data() : nullptr, 1, MPI_LONG_LONG,
            0, MPI_COMM_WORLD);
    MPI_Gather(&rank_pixels, 1, MPI_LONG_LONG,
            rank==0 ? all_pixels.data() : nullptr, 1, MPI_LONG_LONG,
            0, MPI_COMM_WORLD);
    MPI_Gather(host, HN, MPI_CHAR,
            rank==0 ? all_hosts.data() : nullptr, HN, MPI_CHAR,
            0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* pr = fopen("prof_ranks.csv", "w");
        if (pr) {
            fprintf(pr, "rank,node,threads,local_rows,pixels,iters,total_s,compute_s,comm_s,post_s,io_s\n");
            for (int r = 0; r < size; ++r) {
                const double *t = &all_times[r*5];
                const char*   h = &all_hosts[r*HN];
                fprintf(pr, "%d,%s,%d,%d,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r, h, all_threads[r], all_rows[r],
                    (long long)all_pixels[r], (long long)all_iters[r],
                    t[0], t[1], t[2], t[3], t[4]);
            }
            fclose(pr);
        }
    }  

    if (rank == 0) {
        const double comm_avg = comm_sum / size;
        const double comp_avg = comp_sum / size;
        const int threads = omp_get_max_threads();

        fprintf(stderr, "================= Performance Report =================\n");
        fprintf(stderr, "nprocs=%d threads=%d  size=%dx%d  iters=%d\n",
                size, threads, width, height, iters);
        fprintf(stderr, "Compute (avg): %.6f s\n", comp_avg);
        fprintf(stderr, "Comm    (avg): %.6f s\n", comm_avg);
        fprintf(stderr, "Post   (root): %.6f s\n", post_max);
        fprintf(stderr, "I/O    (root): %.6f s\n", io_max);
        fprintf(stderr, "Total  (max) : %.6f s\n", total_max);
        fprintf(stderr, "======================================================\n");

        FILE* sf = fopen("prof_summary.csv", "w");
        if (sf) {
            fprintf(sf, "nprocs,threads,width,height,iters,left,right,lower,upper,total_s,compute_avg_s,comm_avg_s,post_root_s,io_root_s\n");
            fprintf(sf, "%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    size, threads, width, height, iters,
                    left, right, lower, upper,
                    total_max, comp_avg, comm_avg, post_max, io_max);
            fclose(sf);
        }
    }
    */

    MPI_Finalize();
    return 0;
}