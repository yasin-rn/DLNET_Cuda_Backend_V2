#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "Tensor_Kernels.cuh"
#include <cuda_fp16.h>    
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <chrono>
#include <string>
#include <sstream>     
#include <iomanip>     
#include <typeinfo>    

#include <cudnn.h>

template <typename T>
class Tensor
{
private:
	int N;
	int C;
	int H;
	int W;
	int Strides[4];
	int DimSize;

	bool IsChunkPart = false;
	bool IsOwnData = true;

	T* Data;
	T** BatchPtrs;

	size_t TotalSize;

	int Device = 0;
	cudnnTensorDescriptor_t CudnnDesc;


public:
	cudaDataType_t CudaDataType;
	cudnnDataType_t CudnnDataType;

	Tensor(int n, int c, int h, int w);
	Tensor(int n, int c, int h, int w, T* hostData);

	Tensor(int n, int h, int w);
	Tensor(int n, int h, int w, T* hostData);

	Tensor(int h, int w);
	Tensor(int h, int w, T* hostData);

	Tensor(int w);
	Tensor(int w, T* hostData);

	Tensor(int n, int c, int h, int w, T* view_data_ptr,
		int original_n_stride, int original_c_stride, int original_h_stride, int original_w_stride,
		bool is_view_flag, int dimSize);

	~Tensor();

	Tensor(const Tensor<T>& other);
	Tensor<T>& operator=(const Tensor<T>& other);

	Tensor(Tensor<T>&& other) noexcept;
	Tensor<T>& operator=(Tensor<T>&& other) noexcept;

	int GetN() const { return N; }
	int GetC() const { return C; }
	int GetH() const { return H; }
	int GetW() const { return W; }
	int GetDimsize() const { return DimSize; }
	T* GetData() const { return Data; }
	void** GetBatchPtrs() const { return reinterpret_cast<void**>(BatchPtrs); }

	int GetLen(int dim);
	int GetStride(int dim);

	cudnnTensorDescriptor_t GetDesc();

	void Fill(T value);
	void FillRandomUniform();
	void FillRandomUniform(unsigned long long seed);

	std::vector<Tensor<T>> Chunk(int dim, int numOfChunk);
	void Reshape(int n, int c, int h, int w);

	void SetValue(int n, int c, int h, int w, T value);
	void SetValue(int n, int h, int w, T value);
	void SetValue(int h, int w, T value);
	void SetValue(int w, T value);

	std::string ToString() const;




};



template <typename U>
constexpr cudaDataType_t GetCudaDataType();

template <>
constexpr cudaDataType_t GetCudaDataType<__half>() {
	return CUDA_C_16F;
}

template <>
constexpr cudaDataType_t GetCudaDataType<float>() {
	return CUDA_C_32F;
}

template <>
constexpr cudaDataType_t GetCudaDataType<double>() {
	return CUDA_C_64F;
}

template <>
constexpr cudaDataType_t GetCudaDataType<int8_t>() {
	return CUDA_R_8I;
}
template <>
constexpr cudaDataType_t GetCudaDataType<int32_t>() {
	return CUDA_R_32I;
}
template <>
constexpr cudaDataType_t GetCudaDataType<int64_t>() {
	return CUDA_R_64I;
}

template <typename U>
constexpr cudnnDataType_t GetCudnnDataType();

template <>
constexpr cudnnDataType_t GetCudnnDataType<__half>() {
	return CUDNN_DATA_HALF;
}

template <>
constexpr cudnnDataType_t GetCudnnDataType<float>() {
	return CUDNN_DATA_FLOAT;
}

template <>
constexpr cudnnDataType_t GetCudnnDataType<double>() {
	return CUDNN_DATA_DOUBLE;
}

template <>
constexpr cudnnDataType_t GetCudnnDataType<int8_t>() {
	return CUDNN_DATA_INT8;
}
template <>
constexpr cudnnDataType_t GetCudnnDataType<int32_t>() {
	return CUDNN_DATA_INT32;
}
template <>
constexpr cudnnDataType_t GetCudnnDataType<int64_t>() {
	return CUDNN_DATA_INT64;
}
