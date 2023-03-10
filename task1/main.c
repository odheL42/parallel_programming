#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define N 10000000

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

void FillArray_double(double* my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;

	for (int i = 0; i < len; i++) {
		my_array[i] = sin(step*i);
	}
}
void FillArray_float(float* my_array, int len) {
	double step = (M_PI * 2) / 10000000.0;
	double sin_arg = 0;

	for (int i = 0; i < len; i++) {
		my_array[i] = sinf(step * i);
	}
}

double ArraySum_double(double* input_arr, int len) {
	double sum = 0.0;
	for (int i = 0; i < len; i++) {
		sum += input_arr[i];
	}
	return sum;
}
float ArraySum_float(float* input_arr, int len) {
	float sum = 0.0;
	for (int i = 0; i < len; i++) {
		sum += input_arr[i];
	}
	return sum;
}

int main() {
	clock_t prog_begin, prog_end;
	prog_begin = clock();
	long long len = N;
	double* array = (double*)malloc(sizeof(double)*len);
	

	clock_t time_mark_begin, time_mark_end;
	time_mark_begin = clock();
	FillArray_double(array, len);
	time_mark_end = clock();
	printf("TYPE:DOUBLE\nFill (time): %f sec\n", ((double)(time_mark_end - time_mark_begin) / CLOCKS_PER_SEC));

	time_mark_begin = clock();
	double sum_result = ArraySum_double(array, len);
	time_mark_end = clock();
	printf("Sum (time):%f sec\nSum (result):%f\n\n", ((double)(time_mark_end - time_mark_begin) / CLOCKS_PER_SEC), sum_result);


	float* fl_array = (float*)malloc(sizeof(float)*len);
	time_mark_begin = clock();
	FillArray_float(fl_array, len);
	time_mark_end = clock();
	printf("TYPE:FLOAT\nFill (time): %f sec\n", ((double)(time_mark_end - time_mark_begin) / CLOCKS_PER_SEC));

	time_mark_begin = clock();
	float sum_result_FL = ArraySum_float(fl_array, len);
	time_mark_end = clock();
	printf("Sum (time):%f sec\nSum (result):%f\n\n", ((double)(time_mark_end - time_mark_begin) / CLOCKS_PER_SEC), sum_result_FL);

	prog_end = clock();
	printf("Total time: %f sec \n", ((double)(prog_end - prog_begin) / CLOCKS_PER_SEC));
	free(array);
	free(fl_array);
	return 0;
}