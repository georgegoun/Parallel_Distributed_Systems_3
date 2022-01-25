//!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
//%load_ext nvcc_plugin
% % cu
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 3
#define k 5

#define nb 3
#define bs 3

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

__global__ void vector_add(int* ising_sign_d, int* ising_out, int N);

int main(int argc, char* argv[])
{
    int* sign;
    int* ising_sign;
    int* ising_sign_h;
    // CUDA lines
    int* ising_sign_d;
    int* ising_out;

    time_t t;

    /* Intializes random number generator */
    srand((unsigned)time(&t));

    int sign_size = (n + 2) * (n + 2);
    int ising_sign_size = n * n;
    // Malloc 2D Arrays

    sign = (int*)malloc(sign_size * sizeof(int));

    ising_sign = (int*)malloc(ising_sign_size * sizeof(int));

    ising_sign_h = (int*)malloc(3 * (n + 2) * sizeof(int));

    // Could use module but better surround the array with 1 line of values
    // Example cost of 40000 X 40000 array in CPI

    // Initialize 2D array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }
    printf("\n");

    // boundaries set

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

    vector_add<<<nb, bs>>>(ising_sign_d, ising_out, n);
    printf("\n");

    return 0;
}

__global__ void vector_add(int* ising_sign_d, int* ising_out, int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int line = (idx / N) + 1;
    int pos = (idx % N) + 1;

    // self+left+right+up+down

    ising_out[idx] = ising_sign_d[line * (N + 2) + pos] + ising_sign_d[line * (N + 2) + (pos - 1)] + ising_sign_d[line * (N + 2) + (pos + 1)] + ising_sign_d[line * (N + 2) + (pos + 1)] + ising_sign_d[(line - 1) * (N + 2) + pos] + ising_sign_d[(line + 1) * (N + 2) + pos];
    printf("idx: %d line: %d pos: %d\n", idx, line, pos);
}
