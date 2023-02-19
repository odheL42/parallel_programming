#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define N 10000000


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>



void FillArray_double(double** my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;
	double* temp = (double*)malloc(sizeof(double) * len);

	#pragma acc parallel loop
	for (int i = 0; i < len; i++) {
		temp[i] = sin(sin_arg);
		sin_arg += step;
	}
	*my_array = temp;
}
void FillArray_float(float** my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;
	float* temp = (float*)malloc(sizeof(float) * len);

	#pragma acc parallel loop
	for (int i = 0; i < len; i++) {
		temp[i] = (float)sin(sin_arg);
		sin_arg += step;
	}
	*my_array = temp;
}

double ArraySum_double(double** input_arr, int len) {
	double sum = 0;
	#pragma parallel acc loop reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += input_arr[0][i];
	}
	return sum;
}

float ArraySum_float(float** input_arr, int len) {
	float sum = 0;
	#pragma parallel acc loop reduction(+:sum)
	for (int i = 0; i < len; i++) {
		sum += input_arr[0][i];
	}
	return sum;
}

int main() {
	double** array = (double**)malloc(sizeof(double**));
	long long len = N;
	FillArray_double(array, len);
	printf("Sum (double): %f\n\n", ArraySum_double(array, len));
	float** fl_array = (float**)malloc(sizeof(float**));
	FillArray_float(fl_array, len);
	printf("Sum (float): %f\n", ArraySum_float(fl_array, len));


	return 0;
}