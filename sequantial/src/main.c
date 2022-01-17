#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 8
#define k 6

int main(int argc, char* argv[])
{
    int** sign;
    int** ising_sign;
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

        // printf("%d\n", *(sign[8] - 1));

        for (int i = 0; i < n; i++) {
            printf("\t");

            for (int j = 0; j < (n); j++) {
                ising_sign[i][j] = sign[i + 1][j + 1] + sign[i - 1 + 1][j + 1] + sign[i + 1][j - 1 + 1] + sign[i + 1 + 1][j + 1] + sign[i + 1][j + 1 + 1];
                ising_sign[i][j] /= abs(ising_sign[i][j]);
                printf("%d\t", ising_sign[i][j]);
            }
            printf("\n");
        }
        printf("\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sign[i + 1][j + 1] = ising_sign[i][j];
            }
        }
    }

    return 0;
}
// 4, 0 [4, 1][5, 0][6, 1][5, 2]