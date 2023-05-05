#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <nvtx3/nvToolsExt.h>

//следующие фнкции будут вызываться на host, но выполняться на device, поэтому используем __global__
__global__ void getErrorMatrix(double* A, double* Anew, double* end, size_t size){
    //вычисление ошибки
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > size * size) return;
	
	end[idx] = std::abs(Anew[idx] - A[idx]);
}
__global__ void calculateMatrix(double* A, double* Anew, size_t size){
    //Основные вычисления
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i * size + j > (size) * (size)) return;
///!!!!!!!!!!!!!!!!!
	if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1)))
	{
		Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}
//проверить выход за границы памяти ++++++++++++=


int main(int argc, char** argv) {
    int  iters = 1000000, grid_size = 128;
    double accuracy = 1e-6;
        for (int i = 0; i < argc - 1; i++) {
        std::string arg = argv[i];
        if (arg == "-accuracy") {
            std::string dump = std::string(argv[i + 1]);
            accuracy = std::stod(dump);
        }
        else if (arg == "-grid") {
            grid_size = std::stoi(argv[i + 1]);
        }
        else if (arg == "-iters") {
            iters = std::stoi(argv[i + 1]);
        }}
    
    int full_size = grid_size * grid_size;
    double step = 1.0 * (20 - 10) / (grid_size - 1);
    //auto* A = new double[full_size];
    //auto* Anew = new double[full_size];
    //std::memset(A, 0, sizeof(double) * full_size);

	double* A;
	double* Anew;

	cudaMallocHost(&A, full_size * sizeof(double));
	cudaMallocHost(&Anew, full_size * sizeof(double));
	
	std::memset(A, 0, full_size * sizeof(double));

    //инициализируем углы
    A[0] = 10;
    A[grid_size - 1] = 20;
    A[full_size - 1] = 30;
    A[grid_size * (grid_size - 1)] = 20;

    //заполняем(рассчитываем) рамку матрицы
    for (int i = 1; i < grid_size - 1; i++) {
        A[i] = 10 + i * step;
        A[i * (grid_size)] = 10 + i * step;
        A[grid_size * i + (grid_size - 1)] = 20 + i * step;
        A[grid_size * (grid_size - 1) + i] = 20 + i * step;
    }
    std::memcpy(Anew, A, sizeof(double) * full_size);
	double* error;
	cudaMallocHost(&error, sizeof(double));
	*error = 1.0;
	double min_error = accuracy;
    int max_iter = iters, iter = 0;

/////////////////////////////////////////////////////////////////////////////////
    double* ptr_A, *ptr_Anew, *deviceError, *errMx, *buff = NULL;
	size_t sizeofBuff = 0;    
    //выделяем память на gpu для будущих действий, проверяем статусы используемых функций (на всякий случай)
    //место в памяти для матриц
    cudaError_t cudaStatus_1 = cudaMalloc((void**)(&ptr_A), sizeof(double) * full_size);
	cudaError_t cudaStatus = cudaMalloc((void**)(&ptr_Anew), sizeof(double) * full_size);
	cudaError_t cudaStatus_2 = cudaMalloc((void**)&errMx, sizeof(double) * full_size);
	if (cudaStatus_1 != 0 || cudaStatus != 0 || cudaStatus_2 != 0){
		std::cout << "Pu-pu-pu, something is wrong with memory allocation" << std::endl;
		if (ptr_A != NULL) cudaFree(ptr_A);
		if (ptr_Anew != NULL) cudaFree(ptr_Anew);
		if (errMx != NULL) cudaFree(errMx);
		return 42;

		//освободить память ++++++++++++++++++++++++++++++++++++++++++++++++++++
	}    
    //место в памяти для переменной ошибки
    cudaStatus_2 = cudaMalloc((void**)&deviceError, sizeof(double));

	cudaStatus_1 = cudaMemcpy(ptr_A, A, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(ptr_Anew, Anew, sizeof(double) * full_size, cudaMemcpyHostToDevice);
	if (cudaStatus_1 != 0 || cudaStatus != 0 || cudaStatus_2 != 0){
		if (ptr_A != NULL) cudaFree(ptr_A);
		if (ptr_Anew != NULL) cudaFree(ptr_Anew);
		if (errMx != NULL) cudaFree(errMx);
		if (deviceError != NULL) cudaFree(deviceError);
		std::cout << "Pu-pu-pu, something is wrong with memory transfer" << std::endl;
		return 42;
	}//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (IF)

	//получаем значение sizeofBuff, чтобы выделить память для buff 
    // buff пригодится для промежуточных вычислений в функции Max
    // на данном этапе Max ничего не меняет, кроме sizeofBuff, так как *buff = NULL
	cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, full_size);
	cudaMalloc((void**)&buff, sizeofBuff);



    bool graphFlag = false;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;
    nvtxRangePushA("pepe");



	//////
	size_t threads;
	if (grid_size < 1024) threads= grid_size;
	else threads = 1024;
    unsigned int blocks = grid_size / threads;
	dim3 blockD(threads / 32, threads / 32);
    dim3 gridD(blocks * 32, blocks * 32);
	//////
	
		while(iter < max_iter && *error > min_error)
	{
    //основные вычисление с использованием нескольких ядер
    //кол-во потоков: (grid_size - 1)^2
    	if (graphFlag){
			//запускаем выполнения заданных нод
			cudaGraphLaunch(instance, stream);
			//cudaStreamSynchronize(stream);
			cudaMemcpyAsync(error, deviceError, sizeof(double), cudaMemcpyDeviceToHost, stream);
			//ждем, пока выполнится граф
			cudaStreamSynchronize(stream);
			iter += 100;
		}
		else{
			//захватываем граф, наинаем добавлять ноды (функции)
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

			for(size_t i = 0; i < 50; i++)
			{
				calculateMatrix<<<gridD, blockD, 0, stream>>>(ptr_A, ptr_Anew, grid_size);
				calculateMatrix<<<gridD, blockD, 0, stream>>>(ptr_Anew, ptr_A, grid_size);
			}
			// Расчитываем ошибку каждую сотую итерацию
			getErrorMatrix<<<threads*blocks*blocks, threads,  0, stream>>>(ptr_A, ptr_Anew, errMx, grid_size);
			cub::DeviceReduce::Max(buff, sizeofBuff, errMx, deviceError, full_size, stream);
			//заканчиваем работать с графом
			cudaStreamEndCapture(stream, &graph);

			//после закидывания нод, инициализируем (создаем) сам граф
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			graphFlag = true;
  		}
}
	
	
    nvtxRangePop();
	std::cout << "Iter: " << iter << std::endl;
    std::cout << "Error: " << *error << std::endl;

    //обязательно освобождаем память
    free(A);    
    free(Anew);
	cudaFree(ptr_A);
	cudaFree(ptr_Anew);
	cudaStreamDestroy(stream);
	cudaGraphDestroy(graph);
	cudaFree(errMx);
	cudaFree(buff);
    return 0;
}