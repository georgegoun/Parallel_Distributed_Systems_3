% % cu
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 8
#define k 5

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

    inline void
    gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void vector_add(int* d_out, int* ising_sign_d, int j);

int main(int argc, char* argv[])
{
    int** sign;
    int** ising_sign;
    int* ising_sign_h;
    // CUDA lines
    int* ising_sign_d;
    time_t t;

    /* Intializes random number generator */
    srand((unsigned)time(&t));

    // Malloc 2D Arrays

    sign = (int**)malloc((n + 2) * sizeof(int*));

    for (int i = 0; i < n + 2; i++) {
        sign[i] = (int*)malloc((n + 2) * sizeof(int));
    }

    ising_sign = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        ising_sign[i] = (int*)malloc(n * sizeof(int));
    }

    ising_sign_h = (int*)malloc(3 * (n + 2) * sizeof(int*));

    // Could use module but better surround the array with 1 line of values
    // Example cost of 40000 X 40000 array in CPI

    // Initialize 2D array

    for (int i = 1; i < n + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            sign[i][j] = 1 - (2 * (rand() % 2));
        }
        printf("\n");
    }

    for (int count = 0; count < k; count++) {
        // boundaries set

        // 1st column
        sign[0][0] = 0;
        sign[n + 1][0] = 0;
        for (int i = 0; i < n; i++) {
            sign[i + 1][0] = sign[i + 1][n];
        }

        // 1st row
        sign[0][n + 1] = 0;
        for (int i = 0; i < n; i++) {
            sign[0][i + 1] = sign[n][i + 1];
        }

        // 2nd column
        sign[n + 1][n + 1] = 0;
        for (int i = 0; i < n; i++) {
            sign[i + 1][n + 1] = sign[i + 1][1];
        }

        // 2nd row
        for (int i = 0; i < n; i++) {
            sign[n + 1][i + 1] = sign[1][i + 1];
        }

        // print sign array
        for (int i = 0; i < n + 2; i++) {
            for (int j = 0; j < n + 2; j++) {
                printf("%d\t", sign[i][j]);
            }
            printf("\n");
        }
        printf("\n\n");

        // CUDA

        gpuErrchk(cudaMalloc((void**)&ising_sign_d, (n + 2) * 3 * sizeof(int)));
        int* d_out;
        gpuErrchk(cudaMalloc((void**)&d_out, sizeof(int)));

        int h_out;

        for (int i = 0; i < n; i++) {
            printf("\t");
            // fill 2d ising_sign_h array with 3 lines
            for (int r = 0; r < 3; r++) {
                for (int l = 0; l < n + 2; l++) {
                    ising_sign_h[r * (n + 2) + l] = sign[i + r][l];
                }
            }
            gpuErrchk(cudaMemcpy(ising_sign_d, ising_sign_h, (n + 2) * 3 * sizeof(int), cudaMemcpyHostToDevice));
            for (int j = 0; j < (n); j++) {
                vector_add<<<1, 1>>>(d_out, ising_sign_d, j);
                gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
                ising_sign[i][j] = h_out;
                printf("%d\t", ising_sign[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        // Free Memory
        cudaFree(ising_sign_d);
        cudaFree(d_out);
        free(ising_sign_h);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sign[i + 1][j + 1] = ising_sign[i][j];
            }
        }
    }

    // free memory

    for (int i = 0; i < n + 2; i++) {
        free(sign[i]);
    }
    free(sign);

    for (int i = 0; i < n; i++) {
        free(ising_sign[i]);
    }
    free(ising_sign);

    return 0;
}

__global__ void vector_add(int* d_out, int* ising_sign_d, int j)
{
    *d_out = 0;
    *d_out = ising_sign_d[j + 1] + ising_sign_d[(n + 2) + j] + ising_sign_d[(n + 2) + j + 1] + ising_sign_d[(n + 2) + j + 2] + ising_sign_d[(2 * (n + 2)) + j + 1];

    *d_out /= abs(*d_out);
}
