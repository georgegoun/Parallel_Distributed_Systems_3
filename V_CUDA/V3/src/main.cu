#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 12
#define k 2
#define b 8
// 8-2-2-4-4-36
// 6-2-1-9-4-16
#define nb 9
#define bs 4
// type shared_size (sqrt((n*n)/nb)+2)^2
#define shared_size 36

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void vector_add(int* ising_sign_d, int* ising_out, int block_elems, int side_blocks, int side_block_elems);

int main(int argc, char* argv[])
{
    int* sign;
    int* ising_out_h;

    // CUDA lines
    int* ising_sign_d;
    int* ising_out;

    time_t t;

    /* Intializes random number generator */
    srand((unsigned)time(&t));

    int sign_size = (n + 2) * (n + 2);
    int ising_sign_size = n * n;
    // Malloc 1D Arrays

    sign = (int*)malloc(sign_size * sizeof(int));

    // Could use module but better surround the array with 1 line of values
    // Less cost in CPI

    // Initialize 2D array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }
    printf("\n");

    // boundaries set

    for (int k_count = 0; k_count < k; k_count++) {
        // 1st column
        sign[0] = 0;
        sign[(n + 2) * (n + 1)] = 0;
        for (int l = 0, i = n + 2, j = n + 2 + n; l < n; l++) {
            sign[i] = sign[j];
            i += n + 2;
            j += n + 2;
        }

        // 1st row
        for (int l = 0, i = 1, j = n * (n + 2) + 1; l < n; l++) {
            sign[i] = sign[j];
            i++;
            j++;
        }

        // 2nd column
        sign[n + 1] = 0;
        sign[(n + 2) * (n + 2) - 1] = 0;
        for (int l = 0, i = n + 2 + n + 1, j = n + 2 + 1; l < n; l++) {
            sign[i] = sign[j];
            i += n + 2;
            j += n + 2;
        }

        // 2nd row
        for (int l = 0, i = (n + 2) * (n + 1) + 1, j = (n + 2) + 1; l < n; l++) {
            sign[i] = sign[j];
            i++;
            j++;
        }

        // print sign array
        for (int i = 0; i < sign_size; i++) {
            printf("%d\t", sign[i]);
            if ((i + 1) % (n + 2) == 0) {
                printf("\n");
            }
        }
        printf("\n\n");

        // CUDA

        gpuErrchk(cudaMalloc((void**)&ising_sign_d, sign_size * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d, sign, sign_size * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out, sign_size * sizeof(int)));

        int block_elems = (sqrt((n * n) / nb) + 2) * (sqrt((n * n) / nb) + 2);
        int side_blocks = sqrt(nb);
        int side_block_elems = sqrt(block_elems) - 2;

        if ((nb * bs * b * b) != (n * n) && ((side_block_elems * side_block_elems) / bs) == (b * b)) {
            printf("error in assigned values\n");
        }
        vector_add<<<nb, bs>>>(ising_sign_d, ising_out, block_elems, side_blocks, side_block_elems);
        ising_out_h = (int*)malloc(sign_size * sizeof(int));
        gpuErrchk(cudaMemcpy(ising_out_h, ising_out, sign_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Print ising kernel output
        printf("\t");
        for (int i = 0; i < sign_size; i++) {
            printf("%d\t", ising_out_h[i]);
            if (((i + 1) % (n + 2)) == 0) {
                printf("\n\t");
            }
        }
        printf("\n\n");
        for (int i = 0; i < sign_size; i++) {
            sign[i] = ising_out_h[i];
        }

        // Free cuda memory
        cudaFree(ising_sign_d);
        cudaFree(ising_out);
    }
    free(sign);
    return 0;
}

__global__ void vector_add(int* ising_sign_d, int* ising_out, int block_elems, int side_blocks, int side_block_elems)
{
    __shared__ int shared_mem[shared_size];
    unsigned int xBlock = blockDim.x * blockIdx.x;
    unsigned int yBlock = blockDim.y * blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // example of 4 blocks on 8 * 8 = 64 array
    // threads inside cooperate to fill shared memory
    int thread_shared_count = block_elems / bs;
    int idx_pos = (idx % bs) * thread_shared_count; // 0-9-18-27
    int thread_shared_count_temp = thread_shared_count;
    if ((idx + 1) % bs == 0) {
        thread_shared_count += block_elems % thread_shared_count_temp;
    }
    // fill shared with block's values

    // each block:
    // 16 values
    // 4 threads
    // 2 * 2 bb
    // side_blocks = 2 (sqrt(nb))

    // so 4 blocks -> 2 lines 2 columns
    // block1 -> r(0-3)c(0-3)       0       0-3     -> 0-3 || 0-3   ->  0 || 0
    // block2 -> r(0-3)c(4-7)       4       4-7     -> 0-3 || 4-7   ->  0 || 1
    // block3 -> r(4-7)c(0-3)       8       8-11    -> 4-7 || 0-3   ->  1 || 0
    // block4 -> r(4-7)c(4-7)       12      12-15   -> 4-7 || 4-7   ->  1 || 1

    // Block's grid
    int line0 = bs * side_blocks; // 8
    int line = idx / line0;
    int column0 = idx % line0;
    int column = column0 / bs;

    int line_index = ((side_block_elems * side_block_elems * side_blocks) + (side_block_elems * 2)) * line + n + 2 + 1;
    int column_index = column * (side_block_elems); // 0*4, 1*4

    int j = line_index + column_index - n - 2 - 1;

    int j_prev = 0;

    if (idx_pos > 0) {
        j_prev = idx_pos / (side_block_elems + 2) * (n + 2) + idx_pos % (side_block_elems + 2);
    }
    j += j_prev;

    // Fill shared_mem

    for (int i = idx_pos; i < idx_pos + thread_shared_count; i++, j++) {
        shared_mem[i] = ising_sign_d[j];

        if ((i + 1) % (side_block_elems + 2) == 0) {
            j = j + n + 2 - side_block_elems - 2;
        }
    }
    __syncthreads();

    // Find ising model in shared

    j = line_index + column_index - n - 2 - 1;

    j_prev = 0;

    if (idx_pos > 0) {
        j_prev = idx_pos / (side_block_elems + 2) * (n + 2) + idx_pos % (side_block_elems + 2);
    }
    j += j_prev;

    for (int i = idx_pos; i < idx_pos + thread_shared_count; i++, j++) {
        if ((i > side_block_elems + 2) && (i < (block_elems - side_block_elems - 2 - 1)) && ((i + 1) % (side_block_elems + 2) != 0) && (i % (side_block_elems + 2) != 0)) {
            ising_out[j] = shared_mem[i] + shared_mem[i - 1] + shared_mem[i + 1] + shared_mem[i - side_block_elems - 2] + shared_mem[i + side_block_elems + 2];
            ising_out[j] /= abs(ising_out[j]);
        } else {
            if (xBlock == 0) {
                printf("ok ");
            }
            ising_out[j] = shared_mem[i];
        }
        if ((i + 1) % (side_block_elems + 2) == 0) {
            j = j + n + 2 - side_block_elems - 2;
        }
    }
    __syncthreads();
}