#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>   
#include <cuda_fp8.h>    
#include <type_traits> 
#include <cstdint>      


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
			data[i] = curand_uniform_double(&state);
		}
		else if constexpr (std::is_same_v<T, float>) {
			data[i] = curand_uniform(&state);
		}
		else if constexpr (std::is_same_v<T, __half>) {
			data[i] = __float2half(curand_uniform(&state));
		}
		else if constexpr (std::is_same_v<T, int8_t>) {
			data[i] = static_cast<int8_t>(curand(&state));
		}
		else if constexpr (std::is_same_v<T, int16_t>) {
			data[i] = static_cast<int16_t>(curand(&state));
		}
		else if constexpr (std::is_same_v<T, int32_t>) {
			data[i] = static_cast<int32_t>(curand(&state));
		}
		else if constexpr (std::is_same_v<T, int64_t>) {
			uint32_t low = curand(&state);
			uint32_t high = curand(&state);
			data[i] = static_cast<int64_t>((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low));
		}
		else if constexpr (std::is_same_v<T, uint8_t>) {
			data[i] = static_cast<uint8_t>(curand(&state));
		}
		else if constexpr (std::is_same_v<T, uint16_t>) {
			data[i] = static_cast<uint16_t>(curand(&state));
		}
		else if constexpr (std::is_same_v<T, uint32_t>) {
			data[i] = curand(&state);
		}
		else if constexpr (std::is_same_v<T, uint64_t>) {
			uint32_t low = curand(&state);
			uint32_t high = curand(&state);
			data[i] = (static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low);
		}
		else {
		}
	}
}