#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define N 10000000


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>



void FillArray_double(double* my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;

#pragma acc parallel loop
	for (int i = 0; i < len; i++) {
		my_array[i] = sin(sin_arg);
		sin_arg += step;
	}

}
void FillArray_float(float* my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;

#pragma acc parallel loop
	for (int i = 0; i < len; i++) {
		my_array[i] = (float)sin(sin_arg);
		sin_arg += step;
	}
}

/*
double ArraySum_double(double* input_arr, int len) {
	double sum = 0.0;
#pragma acc parallel loop gang num_gangs(2048) reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += input_arr[i];
	}
	return sum;
}

float ArraySum_float(float* input_arr, int len) {
	float sum = 0.0;
#pragma acc parallel loop gang num_gangs(2048) reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += input_arr[i];
	}
	return sum;
}*/

int main() {
	long long len = N;
	double* array = (double*)malloc(sizeof(double) * len);
	FillArray_double(array, len);

	double sum = 0.0;
#pragma acc parallel loop vector vector_length(128) gang num_gangs(2048) reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += array[i];
	}

	printf("Sum (double): %f\n\n", sum);

	float* fl_array = (float*)malloc(sizeof(float) * len);
	FillArray_float(fl_array, len);
	sum = 0.0;
#pragma acc parallel loop vector vector_length(128) gang num_gangs(2048) reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += fl_array[i];
	}

	printf("Sum (float): %f\n", sum);


	return 0;
}