
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

template <typename T>
__global__ void FillKernel(T* data, size_t size, T value) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridStride = static_cast<size_t>(blockDim.x) * gridDim.x; 

	for (size_t i = tid; i < size; i += gridStride) {
		data[i] = value;
	}
}