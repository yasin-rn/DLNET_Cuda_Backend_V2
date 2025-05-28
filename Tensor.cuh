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
#include <sstream>     // std::ostringstream (string oluşturmak için)
#include <iomanip>     // std::fixed, std::setprecision (float formatlama için)
#include <typeinfo>    // typeid (veri tipini yazdırmak için)

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

	bool IsChunkPart = false;
	bool IsOwnData = true;

	T* Data;
	T** BatchPtrs;

	size_t TotalSize;

	int Device = 0;
	cudnnTensorDescriptor_t CudnnDesc;

public:

	Tensor(int n, int c, int h, int w);
	Tensor(int n, int c, int h, int w, T* hostData);

	Tensor(int n, int h, int w);
	Tensor(int n, int h, int w, T* hostData);

	Tensor(int h, int w);
	Tensor(int h, int w, T* hostData);

	Tensor(int w);
	Tensor(int w, T* hostData);

	~Tensor();

	Tensor(const Tensor<T>& other);
	Tensor<T>& operator=(const Tensor<T>& other);

	Tensor(Tensor<T>&& other) noexcept;
	Tensor<T>& operator=(Tensor<T>&& other) noexcept;

	int GetN() const { return N; }
	int GetC() const { return C; }
	int GetH() const { return H; }
	int GetW() const { return W; }
	T* GetData() const { return Data; }
	T** GetBatchPtrs() const { return BatchPtrs; }

	cudnnTensorDescriptor_t GetDesc();

	void Fill(T value);
	void FillRandomUniform();
	void FillRandomUniform(unsigned long long seed);
	void Reshape(int n, int c, int h, int w);

	std::string ToString() const;

	template <typename T>
	constexpr cudnnDataType_t GetCudnnDataType();

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<float>() {
		return CUDNN_DATA_FLOAT();
	}

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<double>() {
		return CUDNN_DATA_DOUBLE;
	}

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<__half>() {
		return CUDNN_DATA_HALF;
	}

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<__nv_fp8_e5m2>() {
		return CUDNN_DATA_HALF;
	}

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<__nv_fp8_e4m3>() {
		return CUDNN_DATA_HALF;
	}

	template <>
	constexpr cudnnDataType_t GetCudnnDataType<__nv_fp8_e8m0>() {
		return CUDNN_DATA_HALF;
	}
	template <>
	constexpr cudnnDataType_t GetCudnnDataType<__nv_fp4_e2m1>() {
		return CUDNN_DATA_HALF;
	}

};



