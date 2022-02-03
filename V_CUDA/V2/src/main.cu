#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 6
#define k 3
#define b 3

#define nb 2
#define bs 2

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

__global__ void v2_kernel(int* ising_sign_d, int* ising_out);

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

    // Could use mod for indexes to get border values, but better surround the array with 4 line of values
    // Less cost in CPI for big data

    // Initialize array

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

        gpuErrchk(cudaMalloc((void**)&ising_sign_d, (n + 2) * (n + 2) * sizeof(int)));
        gpuErrchk(cudaMemcpy(ising_sign_d, sign, (n + 2) * (n + 2) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&ising_out, n * n * sizeof(int)));

        if ((nb * bs * b * b) != (n * n)) {
            printf("error\n");
        }
        v2_kernel<<<nb, bs>>>(ising_sign_d, ising_out);
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

__global__ void v2_kernel(int* ising_sign_d, int* ising_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blocks = b * b;
    int k_id = idx * blocks;
    int line = 0;
    int pos = 0;
    for (int i = 0; i < blocks; i++) {
        line = ((k_id + i) / n) + 1;
        pos = ((k_id + i) % n) + 1;
        // self+left+right+up+down
        ising_out[k_id + i] = ising_sign_d[line * (n + 2) + pos] + ising_sign_d[line * (n + 2) + (pos - 1)] + ising_sign_d[line * (n + 2) + (pos + 1)] + ising_sign_d[(line - 1) * (n + 2) + pos] + ising_sign_d[(line + 1) * (n + 2) + pos];
        ising_out[k_id + i] /= abs(ising_out[k_id + i]);
    }
}
