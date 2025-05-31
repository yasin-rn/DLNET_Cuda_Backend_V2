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
			data[i] = static_cast<double>(curand_uniform_double(&state));  
		}
		else if constexpr (std::is_same_v<T, float>) {
			data[i] = curand_uniform(&state); 
		}
		else if constexpr (std::is_same_v<T, __half>) {
			data[i] = __float2half(curand_uniform(&state)); 
		}
		else if constexpr (std::is_same_v<T, int8_t>) {
			int randval = curand(&state) % 256; 
			data[i] = static_cast<int8_t>(randval - 128);
		}
		else if constexpr (std::is_same_v<T, int32_t>) {
			data[i] = static_cast<int32_t>(curand(&state)); 
		}
		else if constexpr (std::is_same_v<T, int64_t>) {
			uint64_t low = static_cast<uint64_t>(curand(&state));
			uint64_t high = static_cast<uint64_t>(curand(&state));
			data[i] = static_cast<int64_t>((high << 32) | low);
		}
		else {
			data[i] = static_cast<T>(curand_uniform(&state));
		}
	}
}