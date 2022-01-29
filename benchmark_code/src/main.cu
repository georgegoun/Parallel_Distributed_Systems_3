#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// 1:    8-2||       16-4||         4-4-2
// 2:    5000-4||    25000-1000||   250-1000-10
// 3:    10000-10||  50000-2000||   5000-200-10
#define n 10000
#define k 10

// v1
#define nb 50000
#define bs 2000

// v2
#define nb_v2 5000
#define bs_v2 200
#define b_v2 10

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
    ising_sign = (int*)malloc(ising_sign_size * sizeof(int));

    // Could use module but is better for CPI to surround the array with 1 line of values

    // Initialize 1D array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }

    struct timespec start_v0 = { 0 }, stop_v0 = { 0 };
    struct timespec start_v1 = { 0 }, stop_v1 = { 0 };
    struct timespec start_v2 = { 0 }, stop_v2 = { 0 };

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
        // printf("\t");
        for (int i = 0; i < (nb * bs); i++) {
            // printf("%d\t", ising_out_h_v1[i]);
            if ((i % n) == n - 1) {
                // printf("\n\t");
            }
        }
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
