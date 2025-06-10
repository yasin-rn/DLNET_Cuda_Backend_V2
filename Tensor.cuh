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
#include "DataTypes.cuh"
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

	size_t MemSize;

	int Device = 0;

	cudnnTensorDescriptor_t TensorDesc;
	cudnnSeqDataDescriptor_t SeqDesc;

	cudaDataType_t CudaDataType;
	cudnnDataType_t CudnnDataType;

public:
	int* DevSeqPerBatch;

	Tensor();
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
	int GetTotalSize() const { return N * C * H * W; }
	int GetDimsize() const { return DimSize; }
	int GetMemSize() const { return MemSize; }
	std::vector<int> GetShape() const { return { N, C, H, W }; }

	T* GetData() const { return Data; }
	void** GetBatchPtrs() const { return reinterpret_cast<void**>(BatchPtrs); }

	int GetLen(int dim);
	int GetStride(int dim) const;

	cudnnTensorDescriptor_t GetTensorDesc();
	cudnnSeqDataDescriptor_t GetSeqDesc();

	cudaDataType_t GetCudaDataType();
	cudnnDataType_t GetCudnnDataType();

	void ToSeqData();

	void Fill(T value);
	void FillRandomUniform();
	void FillRandomUniform(unsigned long long seed);

	std::vector<Tensor<T>> Chunk(int dim, int numOfChunk);
	void Reshape(int n, int c, int h, int w);
	void Reshape(int n, int h, int w);
	void Reshape(int h, int w);
	void Reshape(int w);

	void SetValue(int n, int c, int h, int w, T value);
	void SetValue(int n, int h, int w, T value);
	void SetValue(int h, int w, T value);
	void SetValue(int w, T value);


	std::string ToString() const;

};


