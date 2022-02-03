#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 8
#define k 3

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

    // Could use mod for idexes to get border values, but better surround the array with 4 line of values
    // Less cost in CPI for big data

    // Initialize array

    for (int i = 0; i < sign_size; i++) {
        sign[i] = 1 - (2 * (rand() % 2));
    }
    printf("\n");

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

        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            printf("\t");
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                ising_sign[i] = sign[j] + sign[j - 1] + sign[j + 1] + sign[j - n - 2] + sign[j + n + 2];
                ising_sign[i] /= abs(ising_sign[i]);
                printf("%d\t", ising_sign[i]);
            }
            j += 2;
            printf("\n");
        }

        // for (int i = 0, j = n + 3; i < n * n; i++) {
        //     printf("\t");
        //     if ((i + 1) % n == 0) {
        //         j += 2;
        //         printf("\n");
        //     }
        //     // self left right up down
        //     ising_sign[i] = sign[j] + sign[j - 1] + sign[j + 1] + sign[j - n - 2] + sign[j + n + 2];
        //     ising_sign[i] /= abs(ising_sign[i]);
        //     printf("%d\t", ising_sign[i]);
        // }

        for (int i = 0, j = n + 3; i < ising_sign_size;) {
            for (int t_i = 0; t_i < n; t_i++, i++, j++) {
                sign[j] = ising_sign[i];
            }
            j += 2;
        }
        printf("\n");
    }

    // free memory

    free(sign);
    free(ising_sign);

    return 0;
}
