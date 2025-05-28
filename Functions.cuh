#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include "Tensor.cuh"

class Functions
{
public:


	static cublasHandle_t CreateCublasHandle();
	static void DestroyCublasHandle(cublasHandle_t handle);

	static cudnnHandle_t CreateCudnnHandle();
	static void DestroyCudnnHandle(cudnnHandle_t handle);


private:

};
