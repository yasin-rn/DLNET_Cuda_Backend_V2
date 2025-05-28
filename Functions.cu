#include "Functions.cuh"

cublasHandle_t Functions::CreateCublasHandle()
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	return handle;
}

void Functions::DestroyCublasHandle(cublasHandle_t handle)
{
	cublasDestroy(handle);
}

cudnnHandle_t Functions::CreateCudnnHandle()
{
	cudnnHandle_t handle;
	cudnnCreate(&handle);
	return handle;
}

void Functions::DestroyCudnnHandle(cudnnHandle_t handle)
{
	cudnnDestroy(handle);
}

