#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string>
#include <vector>
#include <stdio.h>


int main(int argc, char** argv) {
    int accuracy = 10e-6, iters = 1000000, grid_size = 128;
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
        }
    }
    
    clock_t start, end; 
    int full_size = grid_size * grid_size;
    double step = (20 - 10) / (grid_size - 1);
    auto* A = new double[full_size];
    auto* Anew = new double[full_size];
    std::memset(A, 0, sizeof(double) * full_size);

    //инициализируем углы
    A[0] = 10;
    A[grid_size - 1] = 20;
    A[full_size - 1] = 30;
    A[grid_size * (grid_size - 1)] = 20;

    start = clock();
#pragma acc enter data copyin(A[0:full_size]) create(Anew[0:full_size])
    {
#pragma acc parallel loop seq gang num_gangs(size) vector vector_length(size)
        //заполняем(рассчитываем) рамку матрицы
        for (int i = 1; i < grid_size - 1; i++) {
            A[i] = 10 + i * step;
            A[i * (grid_size - 1)] = 20 + i * step;
            A[i * grid_size] = 10 + i * step;
            A[grid_size * (grid_size - 1) + i] = 20 + i * step;
        }
    }
    std::memcpy(Anew, A, sizeof(double) * full_size);


    double error = 1.0, min_error = accuracy;
    int max_iter = iters, iter = 0;

#pragma acc enter data copyin(Anew[0:full_size], A[0:full_size], error)
    
    while (error > min_error && iter < max_iter) {
        iter++;
        if (iter % 100 == 0) { //зануляем ошибку на каждой сотой итерации
#pragma acc kernels async(1)
            error = 0.0;
#pragma acc update device(error) async(1)
        }

#pragma acc data present(A, Anew, error)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:error) async(1)
        
        for (int i = 1; i < grid_size - 1; i++)
        {//выполняем основные вычисления - jacobi iteration
            for (int j = 1; j < grid_size - 1; j++)
            {
                Anew[i * grid_size + j] = 0.25 * (A[i * grid_size + j - 1]
                                            + A[(i - 1) * grid_size + j] 
                                            + A[(i + 1) * grid_size + j] 
                                            + A[i * grid_size + j + 1]);
                error = fmax(error, Anew[i * grid_size + j] - A[i * grid_size + j]); //пересчитываем(узнаём) ошибку
            }
        }
        if (iter % 100 == 0) {// каждую сотую итерацию сохраняем знач-е ошибки на cpu
#pragma acc update host(error) async(1)
#pragma acc wait(1)
        }

        double* temp = A;
        A = Anew;
        Anew = temp;
    }

    end = clock();
#pragma acc update host(A[0:size * size], Anew[0:size * size], error)
    std::cout << "Error: " << error << std::endl;
    std::cout << "Iteration: " << iter << std::endl;
    free(A);
    free(Anew);
    return 0;
}