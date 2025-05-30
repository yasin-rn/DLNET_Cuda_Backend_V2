#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>   
#include <cuda_fp8.h>    
#include <type_traits> 


template <typename T>
__global__ void FillKernel(T* data, size_t size, T value) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridStride = static_cast<size_t>(blockDim.x) * gridDim.x;

	for (size_t i = tid; i < size; i += gridStride) {
		data[i] = value;
	}
}

template <typename T>
__global__ void FillRandomUniformKernel(T* data, size_t size, unsigned long long seed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridStride = static_cast<size_t>(blockDim.x) * gridDim.x;
	curandState_t state;
	curand_init(seed, tid, 0, &state);
	for (size_t i = tid; i < size; i += gridStride) {

		if constexpr (std::is_same_v<T, double>) {
			data[i] = static_cast<T>(curand_uniform_double(&state));
		}
		else
		{
			float rand_val = curand_uniform(&state);

			if constexpr (std::is_same_v<T, float>) {
				data[i] = rand_val;
			}
			else if constexpr (std::is_same_v<T, __half>) {
				data[i] = __float2half(rand_val);
			}
			else if constexpr (std::is_same_v<T, int8_t>) {
				data[i] = static_cast<int8_t>(rand_val);
			}
		}
	}
}