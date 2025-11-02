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

    #pragma omp parallel for schedule(dynamic, 16)
    for(int jj = 0; jj< local_rows; jj++){
        int j = row_start + jj;
        double y0 = j * dy + lower;
        int* rowp = local.data() + jj * width;

        for(int i=0; i<width; i++){
            double x0 = i * dx + left;
            int repeats = 0;
            double x = 0.0, y = 0.0;
            double len2 = 0.0;

            while(repeats < iters && len2 < 4.0){
                double xx = x * x, yy = y * y;
                double x_new = x*x - y*y + x0;
                double xy = x*y;
                x = xx - yy + x0;
                y = 2.0*xy + y0;
                len2 = x*x + y*y;
                repeats++;
            }
            rowp[i] = repeats;
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