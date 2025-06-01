#include <iostream>
#include "Tensor.cuh"
#include <vector>
#include <cublas_v2.h>
#include "functions.cuh"

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")


int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	Tensor<double> A(5, 4);
	Tensor<double> B(3, 4);
	Tensor<double> C(5, 3);

	double alpha = static_cast<double>(1.0f);
	double beta = static_cast<double>(0.0f);

	A.Fill((2.1f));
	B.Fill((2.0f));

	Functions::Matmul(handle, A, B, C, false, true, alpha, beta);

	std::cout << "A:" << A.ToString() << std::endl;
	std::cout << "B:" << B.ToString() << std::endl;
	std::cout << "C:" << C.ToString() << std::endl;

	cublasDestroy(handle);


	return 0;
}