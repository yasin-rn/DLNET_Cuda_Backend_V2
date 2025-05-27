#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "Tensor_Kernels.cuh"

#include <cuda_fp16.h>    
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

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

	int getN() const { return N; }
	int getC() const { return C; }
	int getH() const { return H; }
	int getW() const { return W; }
	T* getData() const { return Data; }
	T** getBatchPtrs() const { return BatchPtrs; }

	void Fill(T value);
	void FillRandomUniform();
	void FillRandomUniform(int seed);

};



