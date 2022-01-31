#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// n - k - nb - bs - b - shared_size
// V0 || V1 || V2 || V3
// 1:   8-2         ||  8-2-16-4            ||  8-2-4-4-2               ||  8-2-4-4-2-36
// 2:   5000-4      ||  4096-4-4096-4096    ||  4096-4-4096-64-8        ||  4096-4-4096-64-32-66564 // // 4096-4-4096-64-8-4356
// 3:   10000-10    ||  10000-10-50000-2000 ||  10000-10-5000-200-10    ||

#define n 4096
#define k 4

// v1
#define nb 4096
#define bs 4096

// v2
#define nb_v2 4096
#define bs_v2 64
#define b_v2 8

// v3
#define nb_v3 4096
#define bs_v3 64
#define b_v3 8
// type shared_size (sqrt((n*n)/nb_v3)+2)^2
#define shared_size 4356

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
// Timer
struct timespec timerStart(struct timespec start)
{
    clock_gettime(CLOCK_MONOTONIC, &start);
    return start;
}
struct timespec timerStop(struct timespec stop)
{
    clock_gettime(CLOCK_MONOTONIC, &stop);
    return stop;
}
double timeDif(struct timespec start_, struct timespec stop_)
{
    double time_dif;
    time_dif = (stop_.tv_sec - start_.tv_sec) * 1e9;
    time_dif = (time_dif + (stop_.tv_nsec - start_.tv_nsec)) * 1e-9;
    return time_dif;
}

__global__ void v1_kernel(int* ising_sign_d_v1, int* ising_out_v1);

__global__ void v2_kernel(int* ising_sign_d_v2, int* ising_out_v2);

__global__ void v3_kernel(int* ising_sign_d_v3, int* ising_out_v3, int block_elems, int side_blocks, int side_block_elems);

int main(int argc, char* argv[])
{
    int* sign;
    int* ising_sign;
    time_t t;

    /* Intializes random number generator */
    srand((unsigned)time(&t));

    int sign_size = (n + 2) * (n + 2);
    int ising_sign_size = n * n;
    // Malloc 1D Arrays

    sign = (int*)malloc(sign_size * sizeof(int));

    // Could use module but is better for CPI to surround the array with 1 line of values

    // Initialize 1D array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }

    struct timespec start_v0 = { 0 }, stop_v0 = { 0 };
    struct timespec start_v1 = { 0 }, stop_v1 = { 0 };
    struct timespec start_v2 = { 0 }, stop_v2 = { 0 };
    struct timespec start_v3 = { 0 }, stop_v3 = { 0 };

    printf("\nv0 starts\n\n");

    /*------------------------V0------------------------*/

    int* sign_v0;
    sign_v0 = (int*)malloc(sizeof(int) * sign_size);

    int* ising_sign_v0;
    ising_sign_v0 = (int*)malloc(sizeof(int) * ising_sign_size);

    for (int i = 0; i < sign_size; i++) {
        sign_v0[i] = sign[i];
    }

    start_v0 = timerStart(start_v0);
    for (int k_count = 0; k_count < k; k_count++) {
        // 1st column
        sign_v0[0] = 0;
        sign_v0[(n + 2) * (n + 1)] = 0;
        for (int l = 0, i = n + 2, j = n + 2 + n; l < n; l++) {
            sign_v0[i] = sign_v0[j];
            i += n + 2;
            j += n + 2;
        }

        // 1st row
        for (int l = 0, i = 1, j = n * (n + 2) + 1; l < n; l++) {
            sign_v0[i] = sign_v0[j];
            i++;
            j++;
        }

        // 2nd column
        sign_v0[n + 1] = 0;
        sign_v0[(n + 2) * (n + 2) - 1] = 0;
        for (int l = 0, i = n + 2 + n + 1, j = n + 2 + 1; l < n; l++) {
            sign_v0[i] = sign_v0[j];
            i += n + 2;
            j += n + 2;
        }

        // 2nd row
        for (int l = 0, i = (n + 2) * (n + 1) + 1, j = (n + 2) + 1; l < n; l++) {
            sign_v0[i] = sign_v0[j];
            i++;
            j++;
        }
        // print sign_v0 array
        // for (int i = 0; i < sign_size; i++) {
        //     printf("%d\t", sign_v0[i]);
        //     if ((i + 1) % (n + 2) == 0) {
        //         printf("\n");
        //     }
        // }
        // printf("\n\n");

        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            // printf("\t");
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                ising_sign_v0[i] = sign_v0[j] + sign_v0[j - 1] + sign_v0[j + 1] + sign_v0[j - n - 2] + sign_v0[j + n + 2];
                ising_sign_v0[i] /= abs(ising_sign_v0[i]);
                // printf("%d\t", ising_sign_v0[i]);
            }
            j += 2;
            // printf("\n");
        }
        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                sign_v0[j] = ising_sign_v0[i];
            }
            j += 2;
        }
        // printf("\n");
    }
    stop_v0 = timerStop(stop_v0);
    // free memory
    free(sign_v0);
    free(ising_sign_v0);

    printf("Time for V0: %lf seconds\n", timeDif(start_v0, stop_v0));
    printf("\nv1 starts\n\n");

    /*------------------------V1------------------------*/

    int* sign_v1;
    int* ising_out_h_v1;

    // CUDA lines
    int* ising_sign_d_v1;
    int* ising_out_v1;

    sign_v1 = (int*)malloc(sign_size * sizeof(int));

    for (int i = 0; i < sign_size; i++) {
        sign_v1[i] = sign[i];
    }

    start_v1 = timerStart(start_v1);
    for (int k_count = 0; k_count < k; k_count++) {
        // 1st column
        sign_v1[0] = 0;
        sign_v1[(n + 2) * (n + 1)] = 0;
        for (int l = 0, i = n + 2, j = n + 2 + n; l < n; l++) {
            sign_v1[i] = sign_v1[j];
            i += n + 2;
            j += n + 2;
        }

        // 1st row
        for (int l = 0, i = 1, j = n * (n + 2) + 1; l < n; l++) {
            sign_v1[i] = sign_v1[j];
            i++;
            j++;
        }

        // 2nd column
        sign_v1[n + 1] = 0;
        sign_v1[(n + 2) * (n + 2) - 1] = 0;
        for (int l = 0, i = n + 2 + n + 1, j = n + 2 + 1; l < n; l++) {
            sign_v1[i] = sign_v1[j];
            i += n + 2;
            j += n + 2;
        }

        // 2nd row
        for (int l = 0, i = (n + 2) * (n + 1) + 1, j = (n + 2) + 1; l < n; l++) {
            sign_v1[i] = sign_v1[j];
            i++;
            j++;
        }

        // print sign_v1 array
        // for (int i = 0; i < sign_size; i++) {
        //     printf("%d\t", sign_v1[i]);
        //     if ((i + 1) % (n + 2) == 0) {
        //         printf("\n");
        //     }
        // }
        // printf("\n\n");

        // CUDA

        gpuErrchk(cudaMalloc((void**)&ising_sign_d_v1, (n + 2) * (n + 2) * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d_v1, sign_v1, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out_v1, n * n * sizeof(int)));

        v1_kernel<<<nb, bs>>>(ising_sign_d_v1, ising_out_v1);
        ising_out_h_v1 = (int*)malloc(ising_sign_size * sizeof(int));
        gpuErrchk(cudaMemcpy(ising_out_h_v1, ising_out_v1, ising_sign_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Print ising kernel output

        //  printf("\t");
        // for (int i = 0; i < (nb * bs); i++) {
        //     printf("%d\t", ising_out_h_v1[i]);
        //     if ((i % n) == n - 1) {
        //         printf("\n\t");
        //     }
        // }
        // printf("\n\n");
        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                sign_v1[j] = ising_out_h_v1[i];
            }
            j += 2;
        }

        // Free cuda memory
        cudaFree(ising_sign_d_v1);
        cudaFree(ising_out_v1);
    }
    stop_v1 = timerStop(stop_v1);

    free(sign_v1);

    printf("Time for V1: %lf seconds\n", timeDif(start_v1, stop_v1));

    printf("\nv2 starts\n\n");

    /*------------------------V2------------------------*/

    int* sign_v2;
    int* ising_out_h_v2;

    // CUDA lines
    int* ising_sign_d_v2;
    int* ising_out_v2;

    sign_v2 = (int*)malloc(sign_size * sizeof(int));

    for (int i = 0; i < sign_size; i++) {
        sign_v2[i] = sign[i];
    }

    start_v2 = timerStart(start_v2);

    for (int k_count = 0; k_count < k; k_count++) {
        // 1st column
        sign_v2[0] = 0;
        sign_v2[(n + 2) * (n + 1)] = 0;
        for (int l = 0, i = n + 2, j = n + 2 + n; l < n; l++) {
            sign_v2[i] = sign_v2[j];
            i += n + 2;
            j += n + 2;
        }

        // 1st row
        for (int l = 0, i = 1, j = n * (n + 2) + 1; l < n; l++) {
            sign_v2[i] = sign_v2[j];
            i++;
            j++;
        }

        // 2nd column
        sign_v2[n + 1] = 0;
        sign_v2[(n + 2) * (n + 2) - 1] = 0;
        for (int l = 0, i = n + 2 + n + 1, j = n + 2 + 1; l < n; l++) {
            sign_v2[i] = sign_v2[j];
            i += n + 2;
            j += n + 2;
        }

        // 2nd row
        for (int l = 0, i = (n + 2) * (n + 1) + 1, j = (n + 2) + 1; l < n; l++) {
            sign_v2[i] = sign_v2[j];
            i++;
            j++;
        }

        // print sign_v2 array

        // for (int i = 0; i < sign_size; i++) {
        //     printf("%d\t", sign_v2[i]);
        //     if ((i + 1) % (n + 2) == 0) {
        //         printf("\n");
        //     }
        // }
        // printf("\n\n");

        // CUDA

        gpuErrchk(cudaMalloc((void**)&ising_sign_d_v2, (n + 2) * (n + 2) * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d_v2, sign_v2, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out_v2, n * n * sizeof(int)));

        if ((nb_v2 * bs_v2 * b_v2 * b_v2) != (n * n)) {
            printf("error\n");
        }

        v2_kernel<<<nb_v2, bs_v2>>>(ising_sign_d_v2, ising_out_v2);
        ising_out_h_v2 = (int*)malloc(ising_sign_size * sizeof(int));
        gpuErrchk(cudaMemcpy(ising_out_h_v2, ising_out_v2, ising_sign_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Print ising kernel output
        // printf("\t");
        // for (int i = 0; i < (ising_sign_size); i++) {
        //     printf("%d\t", ising_out_h_v2[i]);
        //     if ((i % n) == n - 1) {
        //         printf("\n\t");
        //     }
        // }
        // printf("\n\n");
        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                sign_v2[j] = ising_out_h_v2[i];
            }
            j += 2;
        }

        // Free cuda memory
        cudaFree(ising_sign_d_v2);
        cudaFree(ising_out_v2);
    }
    stop_v2 = timerStop(stop_v2);

    free(sign_v2);

    printf("Time for V2: %lf seconds\n", timeDif(start_v2, stop_v2));

    printf("\nv3 starts\n\n");

    /*------------------------V3------------------------*/

    int* sign_v3;
    int* ising_out_h_v3;

    // CUDA lines
    int* ising_sign_d_v3;
    int* ising_out_v3;

    sign_v3 = (int*)malloc(sign_size * sizeof(int));

    for (int i = 0; i < sign_size; i++) {
        sign_v3[i] = sign[i];
    }

    start_v3 = timerStart(start_v3);

    for (int k_count = 0; k_count < k; k_count++) {
        // 1st column
        sign_v3[0] = 0;
        sign_v3[(n + 2) * (n + 1)] = 0;
        for (int l = 0, i = n + 2, j = n + 2 + n; l < n; l++) {
            sign_v3[i] = sign_v3[j];
            i += n + 2;
            j += n + 2;
        }

        // 1st row
        for (int l = 0, i = 1, j = n * (n + 2) + 1; l < n; l++) {
            sign_v3[i] = sign_v3[j];
            i++;
            j++;
        }

        // 2nd column
        sign_v3[n + 1] = 0;
        sign_v3[(n + 2) * (n + 2) - 1] = 0;
        for (int l = 0, i = n + 2 + n + 1, j = n + 2 + 1; l < n; l++) {
            sign_v3[i] = sign_v3[j];
            i += n + 2;
            j += n + 2;
        }

        // 2nd row
        for (int l = 0, i = (n + 2) * (n + 1) + 1, j = (n + 2) + 1; l < n; l++) {
            sign_v3[i] = sign_v3[j];
            i++;
            j++;
        }

        // print sign_v3 array

        // for (int i = 0; i < sign_size; i++) {
        //     printf("%d\t", sign_v3[i]);
        //     if ((i + 1) % (n + 2) == 0) {
        //         printf("\n");
        //     }
        // }
        // printf("\n\n");

        // CUDA

        gpuErrchk(cudaMalloc((void**)&ising_sign_d_v3, sign_size * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d_v3, sign_v3, sign_size * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out_v3, sign_size * sizeof(int)));

        if ((nb_v3 * bs_v3 * b_v3 * b_v3) != (n * n)) {
            printf("error\n");
        }
        int block_elems = (sqrt((n * n) / nb_v3) + 2) * (sqrt((n * n) / nb_v3) + 2);
        int side_blocks = sqrt(nb_v3);
        int side_blocks_elems = sqrt(bs_v3 * b_v3 * b_v3);

        v3_kernel<<<nb_v3, bs_v3>>>(ising_sign_d_v3, ising_out_v3, block_elems, side_blocks, side_blocks_elems);
        printf("ok %d\n", k_count);

        ising_out_h_v3 = (int*)malloc(sign_size * sizeof(int));
        gpuErrchk(cudaMemcpy(ising_out_h_v3, ising_out_v3, sign_size * sizeof(int), cudaMemcpyDeviceToHost));

        // Print ising kernel output

        // printf("\t");
        // for (int i = 0; i < sign_size; i++) {
        //     printf("%d\t", ising_out_h_v3[i]);
        //     if (((i + 1) % (n + 2)) == 0) {
        //         printf("\n\t");
        //     }
        // }
        // printf("\n\n");

        for (int i = 0; i < sign_size; i++) {
            sign_v3[i] = ising_out_h_v3[i];
        }

        // Free cuda memory
        cudaFree(ising_sign_d_v3);
        cudaFree(ising_out_v3);
        free(ising_out_h_v3);
    }

    stop_v2 = timerStop(stop_v2);

    free(sign_v3);
    printf("Time for V3: %lf seconds\n", timeDif(start_v3, stop_v3));
    return 0;
}

__global__ void v1_kernel(int* ising_sign_d_v1, int* ising_out_v1)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int line = (idx / n) + 1;
    int pos = (idx % n) + 1;

    // ising_out_v1 = self+left+right+up+down / abs(self+left+right+up+down)
    ising_out_v1[idx] = ising_sign_d_v1[line * (n + 2) + pos] + ising_sign_d_v1[line * (n + 2) + (pos - 1)] + ising_sign_d_v1[line * (n + 2) + (pos + 1)] + ising_sign_d_v1[(line - 1) * (n + 2) + pos] + ising_sign_d_v1[(line + 1) * (n + 2) + pos];
    ising_out_v1[idx] /= abs(ising_out_v1[idx]);
}

__global__ void v2_kernel(int* ising_sign_d_v2, int* ising_out_v2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocks = b_v2 * b_v2;
    int k_id = idx * blocks;
    int line = 0;
    int pos = 0;
    for (int i = 0; i < blocks; i++) {
        line = ((k_id + i) / n) + 1;
        pos = ((k_id + i) % n) + 1;
        // self+left+right+up+down
        ising_out_v2[k_id + i] = ising_sign_d_v2[line * (n + 2) + pos] + ising_sign_d_v2[line * (n + 2) + (pos - 1)] + ising_sign_d_v2[line * (n + 2) + (pos + 1)] + ising_sign_d_v2[(line - 1) * (n + 2) + pos] + ising_sign_d_v2[(line + 1) * (n + 2) + pos];
        ising_out_v2[k_id + i] /= abs(ising_out_v2[k_id + i]);
    }
}

__global__ void v3_kernel(int* ising_sign_d_v3, int* ising_out_v3, int block_elems, int side_blocks, int side_block_elems)
{
    __shared__ int shared_mem[shared_size];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // example of 4 blocks on 8 * 8 = 64 array
    // threads inside cooperate to fill shared memory
    int thread_shared_count = block_elems / bs_v3;
    int idx_pos = (idx % bs_v3) * thread_shared_count; // 0-9-18-27

    if ((idx + 1) % nb_v3 == 0) {
        thread_shared_count += block_elems % bs_v3;
    }
    // fill shared with block's values

    // each block:
    // 16 values
    // 4 threads
    // 2 * 2 bb
    // side_blocks = 2 (sqrt(nb_v3))

    // so 4 blocks -> 2 lines 2 columns
    // block1 -> r(0-3)c(0-3)       0       0-3     -> 0-3 || 0-3   ->  0 || 0
    // block2 -> r(0-3)c(4-7)       4       4-7     -> 0-3 || 4-7   ->  0 || 1
    // block3 -> r(4-7)c(0-3)       8       8-11    -> 4-7 || 0-3   ->  1 || 0
    // block4 -> r(4-7)c(4-7)       12      12-15   -> 4-7 || 4-7   ->  1 || 1

    // Block's grid
    int line0 = (bs_v3 * nb_v3) / side_blocks; // 8
    int line = idx / line0;
    int column0 = idx % line0;
    int column = column0 / bs_v3;

    int line_index = line * (side_block_elems); // 0*4, 1*4
    int column_index = column * (side_block_elems); // 0*4, 1*4
    int j_line = idx_pos / (side_block_elems + 2); // 0/6 - 9/6 - 18/6 -27/6 || 0-1-3-4
    // idx_pos 0-9-18-27

    // Fill shared_mem

    int j = ((line_index + j_line) * (n + 2)) + column_index; // j 0 - 10 - 30 - 40

    int j_prev = 0;
    if (j > 0) {
        j_prev = ((idx_pos) / (side_block_elems + 2)) * (n + 2) + ((idx_pos) % (side_block_elems + 2)) + column_index;
        j += (j_prev - column_index) % (n + 2);
    }
    for (int i = idx_pos; i < idx_pos + thread_shared_count; i++, j++) {
        if (i > 0 && ((i % (side_block_elems + 2)) == 0) && j_prev != j) {
            j += n + 2 - side_block_elems - 2;
        }
        shared_mem[i] = ising_sign_d_v3[j];
    }
    __syncthreads();

    // Find ising model in shared

    int ising_thread_count = ((n * n) / nb_v3) / (b_v3 * b_v3); // 16/4=4
    int idxx_pos = (idx % bs_v3) * thread_shared_count; // 0-4-8-12

    int line_thread = idxx_pos / (b_v3 * b_v3); // 0/4-4/4-8/4-12/4
    if ((idx + 1) % nb_v3 == 0) {
        ising_thread_count += ((n * n) / nb_v3) % bs_v3;
    }
    // Must b_v3*b_v3==bs_v3 && nb_v3=4
    for (int i = (side_block_elems + 2) * (line_thread + 1) + 1, j = 0; j < b_v3 * b_v3; j++) {
        shared_mem[i + j] = shared_mem[i + j] + shared_mem[i - 1 + j] + shared_mem[i + 1 + j] + shared_mem[i - side_block_elems - 2 + j] + shared_mem[i + side_block_elems + 2 + j];
        shared_mem[i + j] /= abs(shared_mem[i + j]);
    }
    __syncthreads();

    // Fill ising_sign_d_v3

    j = ((line_index + j_line) * (n + 2)) + column_index; // j 0 - 10 - 30 - 40

    j_prev = 0;
    if (j > 0) {
        j_prev = ((idx_pos) / (side_block_elems + 2)) * (n + 2) + ((idx_pos) % (side_block_elems + 2)) + column_index;
        j += (j_prev - column_index) % (n + 2);
    }
    for (int i = idx_pos; i < idx_pos + thread_shared_count; i++, j++) {
        if (i > 0 && ((i % (side_block_elems + 2)) == 0) && j_prev != j) {
            j += n + 2 - side_block_elems - 2;
        }
        ising_out_v3[j] = shared_mem[i];
    }
    __syncthreads();
}
