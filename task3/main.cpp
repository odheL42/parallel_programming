#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    auto* A = new double[full_size];
    auto* Anew = new double[full_size];
    std::memset(A, 0, sizeof(double) * full_size);
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

    double error = 1.0, min_error = accuracy, diff = -1.0;
    int max_iter = iters, iter = 0, idx = 0;

    cublasHandle_t handler;
	cublasStatus_t status;
    status = cublasCreate(&handler);
    if (status != CUBLAS_STATUS_SUCCESS){
        std::cout << "Pu-pu-pu, something is wrong with CUBLAS init." << std::endl;
        return 42;
    }

nvtxRangePushA("pepe");
#pragma acc enter data copyin(Anew[0:full_size], A[0:full_size])
{
    

    while (error > min_error && iter < max_iter) {
        iter++;
        

#pragma acc data present(A, Anew)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) async
            for (int i = 1; i < grid_size - 1; i++)
            {//выполняем основные вычисления - jacobi iteration
                for (int j = 1; j < grid_size - 1; j++)
                {
                    Anew[i * grid_size + j] = 0.25 * (A[i * grid_size + j - 1]
                        + A[(i - 1) * grid_size + j]
                        + A[(i + 1) * grid_size + j]
                        + A[i * grid_size + j + 1]);
                    }
            }

        if (iter % 100 == 0) {// каждую сотую итерацию ищем max
#pragma acc data present (A, Anew) wait

//необходимо передать указатели с device на host, так как функции библиотеки cublas мы вызываем с host, но она выполняется на gpu
#pragma acc host_data use_device(A, Anew)
{
    //вычитаем матрицы (diff = -1, енто коэф, что сложение преобразовывает в вычитание)
    status = cublasDaxpy(handler, full_size, &diff, Anew, 1, A, 1);
    if (status != CUBLAS_STATUS_SUCCESS){
        std::cout << "Pu-pu-pu, something is wrong with daxpy." << std::endl;
        cublasDestroy(handler);
        return 42;
    }

    //max - ?
	status = cublasIdamax(handler, full_size, A, 1, &idx);
    if (status != CUBLAS_STATUS_SUCCESS){
        std::cout << "Pu-pu-pu, something is wrong with idamax" << status <<std::endl;
        cublasDestroy(handler);
        return 42;
    }
}
#pragma acc update host(A[idx - 1]) 
			error = std::abs(A[idx - 1]);

#pragma acc host_data use_device(A, Anew)
//восстановим рамку (граничные условия)
			status = cublasDcopy(handler, full_size, Anew, 1, A, 1);
        }

        double* temp = A;
        A = Anew;
        Anew = temp;
    }}

nvtxRangePop();
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    free(A);
    free(Anew);
    cublasDestroy(handler);
    return 0;
}
