#pragma once

#include <cuda_fp16.h>    
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cudnn.h>

template <typename U>
constexpr cudaDataType_t GetCudaDType();

template <>
constexpr cudaDataType_t GetCudaDType<__half>() {
	return CUDA_R_16F;
}

template <>
constexpr cudaDataType_t GetCudaDType<float>() {
	return CUDA_R_32F;
}

template <>
constexpr cudaDataType_t GetCudaDType<double>() {
	return CUDA_R_64F;
}

template <>
constexpr cudaDataType_t GetCudaDType<int8_t>() {
	return CUDA_R_8I;
}
template <>
constexpr cudaDataType_t GetCudaDType<int32_t>() {
	return CUDA_R_32I;
}
template <>
constexpr cudaDataType_t GetCudaDType<int64_t>() {
	return CUDA_R_64I;
}

template <typename U>
constexpr cudnnDataType_t GetCudnnDType();

template <>
constexpr cudnnDataType_t GetCudnnDType<__half>() {
	return CUDNN_DATA_HALF;
}

template <>
constexpr cudnnDataType_t GetCudnnDType<float>() {
	return CUDNN_DATA_FLOAT;
}

template <>
constexpr cudnnDataType_t GetCudnnDType<double>() {
	return CUDNN_DATA_DOUBLE;
}

template <>
constexpr cudnnDataType_t GetCudnnDType<int8_t>() {
	return CUDNN_DATA_INT8;
}
template <>
constexpr cudnnDataType_t GetCudnnDType<int32_t>() {
	return CUDNN_DATA_INT32;
}
template <>
constexpr cudnnDataType_t GetCudnnDType<int64_t>() {
	return CUDNN_DATA_INT64;
}
