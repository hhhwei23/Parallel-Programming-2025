#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cmath>
#include <cuda_runtime.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

/* Hint 7 */
// this variable is used by device
int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

__constant__ int d_mask[MASK_N][MASK_X][MASK_Y];

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* Hint 5 */
// this function is called by host and executed by device
__global__
void sobel_kernel(const unsigned char* __restrict__ s,
                  unsigned char* __restrict__ t,
                  unsigned height, unsigned width, unsigned channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)width || y >= (int)height) return;

    int xBound = MASK_X / 2, yBound = MASK_Y / 2;
    int adjustX = (MASK_X % 2) ? 1 : 0;
    int adjustY = (MASK_Y % 2) ? 1 : 0;

    double acc[MASK_N*3] = {0.0}; // [Gy/Gx] × [B,G,R]

    for (int i = 0; i < MASK_N; ++i) {
        for (int v = -yBound; v < yBound + adjustY; ++v) {
            int yy = y + v;
            if (yy < 0 || yy >= (int)height) continue;
            for (int u = -xBound; u < xBound + adjustX; ++u) {
                int xx = x + u;
                if (xx < 0 || xx >= (int)width) continue;

                int base = channels * (width * yy + xx);
                int B = s[base + 0];
                int G = s[base + 1];
                int R = s[base + 2];
                int w = d_mask[i][u + xBound][v + yBound];

                acc[i*3 + 0] += (double)B * w;
                acc[i*3 + 1] += (double)G * w;
                acc[i*3 + 2] += (double)R * w;
            }
        }
    }

    double tb=0, tg=0, tr=0;
    for (int i = 0; i < MASK_N; ++i) {
        tb += acc[i*3+0] * acc[i*3+0];
        tg += acc[i*3+1] * acc[i*3+1];
        tr += acc[i*3+2] * acc[i*3+2];
    }

    tb = sqrt(tb) / SCALE;
    tg = sqrt(tg) / SCALE;
    tr = sqrt(tr) / SCALE;

    unsigned char cB = (tb > 255.0) ? 255 : (unsigned char)tb;
    unsigned char cG = (tg > 255.0) ? 255 : (unsigned char)tg;
    unsigned char cR = (tr > 255.0) ? 255 : (unsigned char)tr;

    int out = channels * (width * y + x);
    t[out + 0] = cB;
    t[out + 1] = cG;
    t[out + 2] = cR;
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    size_t nbytes = (size_t)height * width * channels * sizeof(unsigned char);
    unsigned char *d_s = nullptr, *d_t = nullptr;
    cudaMalloc(&d_s, nbytes);
    cudaMalloc(&d_t, nbytes);

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(d_s, host_s, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, mask, sizeof(mask));

    dim3 block(16, 16);
    dim3 grid((width + block.x-1) / block.x, (height + block.y - 1) / block.y);

    /* Hint 3 */
    // acclerate this function
    // sobel(host_s, host_t, height, width, channels);
    sobel_kernel<<<grid, block>>>(d_s, d_t, height, width, channels);
    cudaDeviceSynchronize();
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, d_t, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_s);
    cudaFree(d_t);
    write_png(argv[2], host_t, height, width, channels);

    return 0;
}
