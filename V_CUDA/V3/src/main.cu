#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 8
#define k 2
#define b 2
// 8-2-2-4-4
// 6-2-1-9-4
#define nb 4
#define bs 4

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

__global__ void vector_add(int* ising_sign_d, int* ising_out, int block_elems, int side_blocks);

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
    // Example cost of 40000 X 40000 array in CPI

    // Initialize 2D array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }
    printf("\n");

    // boundaries set

    for (int k_count = 0; k_count < k; k_count++) {
        printf("\n\n\nIteration_%d:\n\n", k_count + 1);
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

        gpuErrchk(cudaMalloc((void**)&ising_sign_d, (n + 2) * (n + 2) * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d, sign, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out, n * n * sizeof(int)));

        if ((nb * bs * b * b) != (n * n)) {
            printf("error\n");
        }
        int block_elems = (sqrt((n * n) / nb) + 1) * (sqrt((n * n) / nb) + 1);
        int side_blocks = sqrt(nb);
        vector_add<<<nb, bs>>>(ising_sign_d, ising_out, block_elems, side_blocks);
        ising_out_h = (int*)malloc(ising_sign_size * sizeof(int));
        gpuErrchk(cudaMemcpy(ising_out_h, ising_out, ising_sign_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Print ising kernel output
        printf("\t");
        for (int i = 0; i < (ising_sign_size); i++) {
            printf("%d\t", ising_out_h[i]);
            if ((i % n) == n - 1) {
                printf("\n\t");
            }
        }
        printf("\n\n");
        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                sign[j] = ising_out_h[i];
            }
            j += 2;
        }

        // Free cuda memory
        cudaFree(ising_sign_d);
        cudaFree(ising_out);
    }
    free(sign);
    return 0;
}

__global__ void vector_add(int* ising_sign_d, int* ising_out, int block_elems, int side_blocks)
{
    // n*n/nb
    __shared__ int shared_mem[25];
    unsigned int xBlock = blockDim.x * blockIdx.x;
    unsigned int yBlock = blockDim.y * blockIdx.y;
    unsigned int xIndex = xBlock + threadIdx.x;
    unsigned int yIndex = yBlock + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // threads inside cooperate to fill shared memory
    int thread_shared_count = block_elems / bs;
    int start = (idx % bs) * thread_shared_count;

    if ((idx + 1) % nb == 0) {
        thread_shared_count += block_elems % bs;
    }
    // fill shared with block's values
    // example of 4 blocks on 8 * 8 = 64 array
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
    int line0 = (bs * nb) / side_blocks; // 8
    int line = idx / line0;
    int column0 = idx % line0;
    int column = column0 / bs;

    // TODO: Find ising_sign_d values using column-lines
    // think 2d in the right way
    for (int i = start; i < start + thread_shared_count; i++) {
        shared_mem[i] = ising_sign_d[];
    }

    printf("idx: %d\t|| xBlock: %d\t| Thread Count: %d\t| line: %d\t|| column: %d\t|| size: %d\n", idx, xBlock, thread_shared_count, line, column, block_elems);

    // column = (threads * nb)/(side_blocks* sideblocks) = 4

    // int line = ((k_id + i) / n) + 1;

    // for (int i = start; i < start + thread_shared_count; i++) {
    //     line;
    //     shared_mem[i]
    //         = ising_sign_d[i];
    //     printf("%d: %d\n", idx, shared_mem[i]);
    // }

    __syncthreads();

    // for (int i = 0; i < block_elems; i++) {
    // }
    //  int blocks = b * b;
    //  int k_id = idx * blocks;
    //  int line = 0;
    //  int pos = 0;
    //  for (int i = 0; i < blocks; i++) {
    //      line = ((k_id + i) / n) + 1;
    //      pos = ((k_id + i) % n) + 1;
    //      // self+left+right+up+down
    //      ising_out[k_id + i] = ising_sign_d[line * (n + 2) + pos] + ising_sign_d[line * (n + 2) + (pos - 1)] + ising_sign_d[line * (n + 2) + (pos + 1)] + ising_sign_d[(line - 1) * (n + 2) + pos] + ising_sign_d[(line + 1) * (n + 2) + pos];
    //      ising_out[k_id + i] /= abs(ising_out[k_id + i]);
    //  }
}
