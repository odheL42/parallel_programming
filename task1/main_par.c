
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
int main() {
    clock_t begin, end;
    begin = clock();
    long long len = N;

    double* array = (double*)malloc(sizeof(double) * len);
    double step = (M_PI * 2) / 10000000.0;
#pragma acc parallel loop
    for (int i = 0; i < len; i++) {
        array[i] = sin(step * i);
    }

    double sum = 0.0;
#pragma acc parallel loop vector vector_length(128) gang num_gangs(2048) reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += array[i];
    }
    printf("Sum (double): %f\n", sum);

    float* fl_array = (float*)malloc(sizeof(float) * len);
    step = (M_PI * 2) / 10000000.0;
#pragma acc parallel loop
    for (int i = 0; i < len; i++) {
        fl_array[i] = (float)sinf(step * i);
    }

    sum = 0.0;
#pragma acc parallel loop vector vector_length(128) gang num_gangs(2048) reduction(+:sum)
    for (int i = 0; i < len; i++) {
        sum += fl_array[i];
    }
    printf("Sum (float): %f\n\n", sum);

    end = clock();
    printf("Total time: %f sec \n\n\n", ((double)(end - begin) / CLOCKS_PER_SEC));
    free(array);
    free(fl_array); 
    return 0;
}