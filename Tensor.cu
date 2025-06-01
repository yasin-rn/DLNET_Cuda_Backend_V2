#include "Tensor.cuh"
#include <cudnn.h>

template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w) :N(n), C(c), H(h), W(w)
{
	DimSize = 4;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);

	std::vector<T*> hostPtrs(N * C);
	for (size_t i = 0; i < N * C; ++i)
		hostPtrs[i] = Data + i * Strides[2];

	cudaMalloc(&BatchPtrs, N * C * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * C * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);


}
template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w, T* hostData) :N(n), C(c), H(h), W(w)
{

	DimSize = 4;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);
	cudaMemcpy(Data, hostData, TotalSize, cudaMemcpyHostToDevice);

	std::vector<T*> hostPtrs(N * C);
	for (size_t i = 0; i < N * C; ++i)
		hostPtrs[i] = Data + i * Strides[2];

	cudaMalloc(&BatchPtrs, N * C * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * C * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);

}

template <typename T>
Tensor<T>::Tensor(int n, int h, int w) :N(n), C(1), H(h), W(w)
{

	DimSize = 3;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[2];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);
}
template <typename T>
Tensor<T>::Tensor(int n, int h, int w, T* hostData) :N(n), C(1), H(h), W(w)
{

	DimSize = 3;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);
	cudaMemcpy(Data, hostData, TotalSize, cudaMemcpyHostToDevice);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[2];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);
}

template <typename T>
Tensor<T>::Tensor(int h, int w) :N(1), C(1), H(h), W(w)
{

	DimSize = 2;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[2];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);
}
template <typename T>
Tensor<T>::Tensor(int h, int w, T* hostData) :N(1), C(1), H(h), W(w)
{
	DimSize = 2;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);
	cudaMemcpy(Data, hostData, TotalSize, cudaMemcpyHostToDevice);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[3];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);
}

template <typename T>
Tensor<T>::Tensor(int w) :N(1), C(1), H(1), W(w)
{
	DimSize = 1;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[3];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);

}
template <typename T>
Tensor<T>::Tensor(int w, T* hostData) :N(1), C(1), H(1), W(w)
{
	DimSize = 1;
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;

	IsChunkPart = false;
	IsOwnData = true;

	TotalSize = sizeof(T) * N * C * H * W;

	cudaMalloc(&Data, TotalSize);
	cudaMemcpy(Data, hostData, TotalSize, cudaMemcpyHostToDevice);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[3];

	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptor(CudnnDesc, CUDNN_TENSOR_NCHW, GetCudnnDType<T>(), N, C, H, W);
}

template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w, T* view_data_ptr,
	int original_n_stride, int original_c_stride, int original_h_stride, int original_w_stride,
	bool is_view_flag, int dimSize)
	: N(n), C(c), H(h), W(w), Data(view_data_ptr), IsOwnData(!is_view_flag), IsChunkPart(is_view_flag), BatchPtrs(nullptr), CudnnDesc(nullptr), DimSize(dimSize)
{
	cudaGetDevice(&Device);
	CudaDataType = GetCudaDType<T>();
	CudnnDataType = GetCudnnDType<T>();

	if (IsOwnData) {
		TotalSize = sizeof(T) * N * C * H * W;
	}
	else {
		TotalSize = 0;
	}

	this->Strides[0] = original_w_stride;
	this->Strides[1] = original_h_stride;
	this->Strides[2] = original_c_stride;
	this->Strides[3] = original_n_stride;

	cudnnCreateTensorDescriptor(&CudnnDesc);
	cudnnSetTensor4dDescriptorEx(CudnnDesc, GetCudnnDType<T>(), N, C, H, W,
		original_n_stride, original_c_stride, original_h_stride, original_w_stride);

	if (N > 0) {
		std::vector<T*> host_batch_ptrs_vec(N);
		for (int i = 0; i < N; ++i) {
			host_batch_ptrs_vec[i] = this->Data + ((size_t)i * original_n_stride);
		}
		cudaMalloc(&BatchPtrs, (size_t)N * sizeof(T*));
		cudaMemcpy(BatchPtrs, host_batch_ptrs_vec.data(), (size_t)N * sizeof(T*), cudaMemcpyHostToDevice);
	}
	else {
		BatchPtrs = nullptr;
	}
}

template <typename T>
Tensor<T>::~Tensor()
{
	if (IsOwnData && Data != nullptr)
	{
		cudaFree(Data);
		Data = nullptr;
	}

	if (BatchPtrs != nullptr)
	{
		cudaFree(BatchPtrs);
		BatchPtrs = nullptr;
	}

	if (CudnnDesc != nullptr)
	{
		cudnnDestroyTensorDescriptor(CudnnDesc);
		CudnnDesc = nullptr;
	}
}



template <typename T>
Tensor<T>::Tensor(const Tensor<T>& other)
	: N(other.N), C(other.C), H(other.H), W(other.W),
	IsChunkPart(other.IsChunkPart), IsOwnData(true),
	TotalSize(other.TotalSize),
	CudaDataType(other.CudaDataType), CudnnDataType(other.CudnnDataType), CudnnDesc(other.CudnnDesc), DimSize(other.DimSize)
{


	memcpy(Strides, other.Strides, 4 * sizeof(int));

	cudaMalloc(&Data, TotalSize);
	cudaMemcpy(Data, other.Data, TotalSize, cudaMemcpyDeviceToDevice);

	std::vector<T*> hostPtrs(N);
	for (size_t i = 0; i < N; ++i)
		hostPtrs[i] = Data + i * Strides[3];
	cudaMalloc(&BatchPtrs, N * sizeof(T*));
	cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

}
template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
{
	if (this != &other)
	{
		if (IsOwnData && Data)
			cudaFree(Data);
		if (BatchPtrs)
			cudaFree(BatchPtrs);

		N = other.N;
		C = other.C;
		H = other.H;
		W = other.W;
		DimSize = other.DimSize;

		IsChunkPart = other.IsChunkPart;
		IsOwnData = true;
		TotalSize = other.TotalSize;
		memcpy(Strides, other.Strides, 4 * sizeof(int));

		cudaMalloc(&Data, TotalSize);
		cudaMemcpy(Data, other.Data, TotalSize, cudaMemcpyDeviceToDevice);
		std::vector<T*> hostPtrs(N);
		for (size_t i = 0; i < N; ++i)
			hostPtrs[i] = Data + i * Strides[3];
		cudaMalloc(&BatchPtrs, N * sizeof(T*));
		cudaMemcpy(BatchPtrs, hostPtrs.data(), N * sizeof(T*), cudaMemcpyHostToDevice);

		CudnnDesc = other.CudnnDesc;
	}
	return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
	: N(other.N), C(other.C), H(other.H), W(other.W),
	IsChunkPart(other.IsChunkPart), IsOwnData(other.IsOwnData),
	Data(other.Data), BatchPtrs(other.BatchPtrs), TotalSize(other.TotalSize),
	CudaDataType(other.CudaDataType), CudnnDataType(other.CudnnDataType), CudnnDesc(other.CudnnDesc), DimSize(other.DimSize)
{
	memcpy(Strides, other.Strides, 4 * sizeof(int));
	other.Data = nullptr;
	other.BatchPtrs = nullptr;
	other.IsOwnData = false;
	other.TotalSize = 0;
	other.CudnnDesc = nullptr;
}
template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept
{
	if (this != &other)
	{
		if (IsOwnData && Data)
			cudaFree(Data);
		if (BatchPtrs)
			cudaFree(BatchPtrs);

		N = other.N;
		C = other.C;
		H = other.H;
		W = other.W;
		DimSize = other.DimSize;

		IsChunkPart = other.IsChunkPart;
		IsOwnData = other.IsOwnData;
		TotalSize = other.TotalSize;
		memcpy(Strides, other.Strides, 4 * sizeof(int));
		Data = other.Data;
		BatchPtrs = other.BatchPtrs;

		CudnnDesc = other.CudnnDesc;

		other.Data = nullptr;
		other.BatchPtrs = nullptr;
		other.IsOwnData = false;
		other.TotalSize = 0;
		other.CudnnDesc = nullptr;

	}
	return *this;
}

template <typename T>
int Tensor<T>::GetLen(int dim)
{
	int selectedDim = 4 - DimSize + dim;
	switch (selectedDim)
	{
	case(0):return N;
	case(1):return DimSize == 3 ? N : C;
	case(2):return H;
	case(3):return W;
	default:
		return N * C * H * W;
	}
}

template <typename T>
int Tensor<T>::GetStride(int dim)
{
	switch (dim)
	{
	case(0):return Strides[0];
	case(1):return Strides[1];
	case(2):return Strides[2];
	case(3):return Strides[3];
	default:
		return N * C * H * W;
	}
}

template <typename T>
cudnnTensorDescriptor_t Tensor<T>::GetDesc()
{
	return CudnnDesc;
}

template <typename T>
cudaDataType_t Tensor<T>::GetCudaDataType()
{
	return CudaDataType;
}
template <typename T>
cudnnDataType_t Tensor<T>::GetCudnnDataType()
{
	return CudnnDataType;
}


template <typename T>
void Tensor<T>::Fill(T value)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	int minGridSize = 0;
	int blockSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, FillKernel<T>, 0, 0);

	int blocksPerGrid = (TotalSize + blockSize - 1) / blockSize;

	FillKernel<T> << <blocksPerGrid, blockSize >> > (Data, TotalSize, value);

	cudaDeviceSynchronize();
}

template <typename T>
void Tensor<T>::FillRandomUniform()
{
	unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
	FillRandomUniform(seed);

}
template <typename T>
void Tensor<T>::FillRandomUniform(unsigned long long seed)
{
	if (TotalSize == 0) return;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	int minGridSize = 0;
	int blockSize = 0;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, FillRandomUniformKernel<T>, 0, 0);

	size_t numElements = TotalSize / sizeof(T);
	if (numElements == 0) return;

	int blocksPerGrid = (numElements + blockSize - 1) / blockSize;
	if (blocksPerGrid == 0 && numElements > 0) blocksPerGrid = 1;

	FillRandomUniformKernel<T> << <blocksPerGrid, blockSize >> > (Data, numElements, seed);

	cudaGetLastError();

	cudaDeviceSynchronize();
}

template <typename T>
std::vector<Tensor<T>> Tensor<T>::Chunk(int dim, int numOfChunk) {

	int selectedDim = 4 - DimSize + dim;
	std::vector<Tensor<T>> chunks;

	int original_n_dim = this->N;
	int original_c_dim = this->C;
	int original_h_dim = this->H;
	int original_w_dim = this->W;

	int actual_wStride = 1;
	int actual_hStride = original_w_dim;
	int actual_cStride = original_h_dim * original_w_dim;
	int actual_nStride = original_c_dim * original_h_dim * original_w_dim;

	T* base_data_ptr = this->Data;
	long long current_element_offset = 0;

	int dim_to_chunk_original_size;
	long long stride_of_chunked_dim_in_elements;

	switch (selectedDim) {
	case 0: dim_to_chunk_original_size = original_n_dim; stride_of_chunked_dim_in_elements = actual_nStride; break;
	case 1: dim_to_chunk_original_size = original_c_dim; stride_of_chunked_dim_in_elements = actual_cStride; break;
	case 2: dim_to_chunk_original_size = original_h_dim; stride_of_chunked_dim_in_elements = actual_hStride; break;
	case 3: dim_to_chunk_original_size = original_w_dim; stride_of_chunked_dim_in_elements = actual_wStride; break;
	default: return chunks;
	}


	if (dim_to_chunk_original_size == 0 && numOfChunk > 0) {
		for (int i = 0; i < numOfChunk; ++i) {
			chunks.emplace_back((selectedDim == 0 ? 0 : original_n_dim),
				(selectedDim == 1 ? 0 : original_c_dim),
				(selectedDim == 2 ? 0 : original_h_dim),
				(selectedDim == 3 ? 0 : original_w_dim),
				base_data_ptr,
				actual_nStride, actual_cStride, actual_hStride, actual_wStride,
				true, DimSize);
		}
		return chunks;
	}



	if (numOfChunk > dim_to_chunk_original_size) {
		for (int i = 0; i < dim_to_chunk_original_size; ++i) {
			int chunk_n = (selectedDim == 0) ? 1 : original_n_dim;
			int chunk_c = (selectedDim == 1) ? 1 : original_c_dim;
			int chunk_h = (selectedDim == 2) ? 1 : original_h_dim;
			int chunk_w = (selectedDim == 3) ? 1 : original_w_dim;
			T* chunk_data_start_ptr = base_data_ptr + current_element_offset;
			chunks.emplace_back(chunk_n, chunk_c, chunk_h, chunk_w,
				chunk_data_start_ptr,
				actual_nStride, actual_cStride, actual_hStride, actual_wStride,
				true, DimSize);
			current_element_offset += stride_of_chunked_dim_in_elements;
		}

		T* zero_chunk_data_ptr = base_data_ptr + current_element_offset;
		for (int i = 0; i < numOfChunk - dim_to_chunk_original_size; ++i) {
			int chunk_n = (selectedDim == 0) ? 0 : original_n_dim;
			int chunk_c = (selectedDim == 1) ? 0 : original_c_dim;
			int chunk_h = (selectedDim == 2) ? 0 : original_h_dim;
			int chunk_w = (selectedDim == 3) ? 0 : original_w_dim;
			chunks.emplace_back(chunk_n, chunk_c, chunk_h, chunk_w,
				zero_chunk_data_ptr,
				actual_nStride, actual_cStride, actual_hStride, actual_wStride,
				true, DimSize);
		}
		return chunks;
	}


	int chunk_dim_base_val = dim_to_chunk_original_size / numOfChunk;
	int remainder_val = dim_to_chunk_original_size % numOfChunk;

	for (int i = 0; i < numOfChunk; ++i) {
		int current_chunk_dim_actual_val = chunk_dim_base_val + (i < remainder_val ? 1 : 0);

		int chunk_n = original_n_dim;
		int chunk_c = original_c_dim;
		int chunk_h = original_h_dim;
		int chunk_w = original_w_dim;


		switch (selectedDim) {
		case 0: chunk_n = current_chunk_dim_actual_val; break;
		case 1: chunk_c = current_chunk_dim_actual_val; break;
		case 2: chunk_h = current_chunk_dim_actual_val; break;
		case 3: chunk_w = current_chunk_dim_actual_val; break;
		}

		T* chunk_data_start_ptr = base_data_ptr + current_element_offset;

		chunks.emplace_back(chunk_n, chunk_c, chunk_h, chunk_w,
			chunk_data_start_ptr,
			actual_nStride, actual_cStride, actual_hStride, actual_wStride,
			true, DimSize);


		current_element_offset += (long long)current_chunk_dim_actual_val * stride_of_chunked_dim_in_elements;
	}

	return chunks;
}

template <typename T>
void Tensor<T>::Reshape(int n, int c, int h, int w)
{
	N = n;
	C = c;
	H = h;
	W = w;

	Strides[0] = 1;
	Strides[1] = W;
	Strides[2] = H * W;
	Strides[3] = C * H * W;
}


template <typename T>
void Tensor<T>::SetValue(int n, int c, int h, int w, T value)
{
	cudnnDataType_t dataType_desc;
	int n_from_desc, c_from_desc, h_from_desc, w_from_desc;
	int nStride_from_desc, cStride_from_desc, hStride_from_desc, wStride_from_desc;

	cudnnGetTensor4dDescriptor(this->CudnnDesc,
		&dataType_desc,
		&n_from_desc, &c_from_desc, &h_from_desc, &w_from_desc,
		&nStride_from_desc, &cStride_from_desc, &hStride_from_desc, &wStride_from_desc);

	size_t index = static_cast<size_t>(n) * nStride_from_desc +
		static_cast<size_t>(c) * cStride_from_desc +
		static_cast<size_t>(h) * hStride_from_desc +
		static_cast<size_t>(w) * wStride_from_desc;

	cudaMemcpy(this->Data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Tensor<T>::SetValue(int n, int h, int w, T value)
{
	cudnnDataType_t dataType_desc;
	int n_from_desc, c_from_desc, h_from_desc, w_from_desc;
	int nStride_from_desc, cStride_from_desc, hStride_from_desc, wStride_from_desc;

	cudnnGetTensor4dDescriptor(this->CudnnDesc,
		&dataType_desc,
		&n_from_desc, &c_from_desc, &h_from_desc, &w_from_desc,
		&nStride_from_desc, &cStride_from_desc, &hStride_from_desc, &wStride_from_desc);

	size_t c_index = 0;


	size_t index = static_cast<size_t>(n) * nStride_from_desc +
		static_cast<size_t>(c_index) * cStride_from_desc +
		static_cast<size_t>(h) * hStride_from_desc +
		static_cast<size_t>(w) * wStride_from_desc;

	cudaMemcpy(this->Data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Tensor<T>::SetValue(int h, int w, T value)
{
	cudnnDataType_t dataType_desc;
	int n_from_desc, c_from_desc, h_from_desc, w_from_desc;
	int nStride_from_desc, cStride_from_desc, hStride_from_desc, wStride_from_desc;

	cudnnGetTensor4dDescriptor(this->CudnnDesc,
		&dataType_desc,
		&n_from_desc, &c_from_desc, &h_from_desc, &w_from_desc,
		&nStride_from_desc, &cStride_from_desc, &hStride_from_desc, &wStride_from_desc);

	size_t n_index = 0;
	size_t c_index = 0;

	size_t index = static_cast<size_t>(n_index) * nStride_from_desc +
		static_cast<size_t>(c_index) * cStride_from_desc +
		static_cast<size_t>(h) * hStride_from_desc +
		static_cast<size_t>(w) * wStride_from_desc;

	cudaMemcpy(this->Data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void Tensor<T>::SetValue(int w, T value)
{
	cudnnDataType_t dataType_desc;
	int n_from_desc, c_from_desc, h_from_desc, w_from_desc;
	int nStride_from_desc, cStride_from_desc, hStride_from_desc, wStride_from_desc;

	cudnnGetTensor4dDescriptor(this->CudnnDesc,
		&dataType_desc,
		&n_from_desc, &c_from_desc, &h_from_desc, &w_from_desc,
		&nStride_from_desc, &cStride_from_desc, &hStride_from_desc, &wStride_from_desc);

	size_t n_index = 0;
	size_t c_index = 0;
	size_t h_index = 0;


	size_t index = static_cast<size_t>(n_index) * nStride_from_desc +
		static_cast<size_t>(c_index) * cStride_from_desc +
		static_cast<size_t>(h_index) * hStride_from_desc +
		static_cast<size_t>(w) * wStride_from_desc;

	cudaMemcpy(this->Data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
}

std::string getFriendlyTypeName(const std::type_info& ti) {
	if (ti == typeid(__half))    return "float16";
	if (ti == typeid(float))     return "float32";
	if (ti == typeid(double))    return "float64";
	if (ti == typeid(int8_t))    return "int8";
	if (ti == typeid(int32_t))   return "int32";
	if (ti == typeid(int64_t))   return "int64";

	return ti.name();
}

template <typename T>
std::string Tensor<T>::ToString() const {
	std::ostringstream oss;


	size_t logical_num_elements = static_cast<size_t>(this->N) * this->C * this->H * this->W;

	if (Data == nullptr || logical_num_elements == 0) {
		oss << "tensor([], dtype=" << getFriendlyTypeName(typeid(T)) << ", device=cuda:" << std::to_string(this->Device) << ")";
		return oss.str();
	}

	std::vector<T> host_data_elements(logical_num_elements);

	cudnnDataType_t dataType_desc;
	int n_from_desc, c_from_desc, h_from_desc, w_from_desc;
	int nStride_from_desc, cStride_from_desc, hStride_from_desc, wStride_from_desc;

	cudnnGetTensor4dDescriptor(this->CudnnDesc,
		&dataType_desc,
		&n_from_desc, &c_from_desc, &h_from_desc, &w_from_desc,
		&nStride_from_desc, &cStride_from_desc, &hStride_from_desc, &wStride_from_desc);

	for (int n_idx = 0; n_idx < this->N; ++n_idx) {
		for (int c_idx = 0; c_idx < this->C; ++c_idx) {
			for (int h_idx = 0; h_idx < this->H; ++h_idx) {
				for (int w_idx = 0; w_idx < this->W; ++w_idx) {
					size_t offset_in_original_data =
						static_cast<size_t>(n_idx) * nStride_from_desc +
						static_cast<size_t>(c_idx) * cStride_from_desc +
						static_cast<size_t>(h_idx) * hStride_from_desc +
						static_cast<size_t>(w_idx) * wStride_from_desc;

					size_t linear_idx_in_chunk_buffer =
						static_cast<size_t>(n_idx) * this->C * this->H * this->W +
						static_cast<size_t>(c_idx) * this->H * this->W +
						static_cast<size_t>(h_idx) * this->W +
						w_idx;

					cudaError_t err = cudaMemcpy(&host_data_elements[linear_idx_in_chunk_buffer],
						this->Data + offset_in_original_data,
						sizeof(T),
						cudaMemcpyDeviceToHost);
					if (err != cudaSuccess) {
						oss.str("");
						oss << "Error copying element at NCHW(" << n_idx << "," << c_idx << "," << h_idx << "," << w_idx
							<< ") : " << cudaGetErrorString(err);
						return oss.str();
					}
				}
			}
		}
	}

	oss << "tensor(";
	oss << std::fixed << std::setprecision(4);

	auto print_val = [&](T val) {
		if constexpr (std::is_same_v<T, __half>) {
			oss << std::fixed << std::setprecision(4) << static_cast<float>(val);
		}
		else if constexpr (std::is_same_v<T, float>) {
			oss << std::fixed << std::setprecision(4) << val;
		}
		else if constexpr (std::is_same_v<T, double>) {
			oss << std::fixed << std::setprecision(4) << val;
		}
		else if constexpr (std::is_same_v<T, int8_t>) {
			oss << static_cast<int>(val);
		}
		else if constexpr (std::is_same_v<T, int32_t>) {
			oss << val;
		}
		else if constexpr (std::is_same_v<T, int64_t>) {
			oss << static_cast<long long>(val);
		}
		else {
			oss << std::fixed << std::setprecision(4) << static_cast<float>(val);
		}
		};


	const int PRINT_THRESHOLD_W = 10;
	const int EDGE_ITEMS_W = 3;

	int N_ = this->N;
	int C_ = this->C;
	int H_ = this->H;
	int W_ = this->W;

	if (logical_num_elements == 0) {
		oss << "[]";
	}
	else {
		if (N_ == 1 && C_ == 1 && H_ == 1 && W_ == 1) {
			print_val(host_data_elements[0]);
		}
		else {
			std::string base_indent = "        ";

			oss << "[";
			for (int n = 0; n < N_; ++n) {
				if (n > 0) oss << "," << "\n" << base_indent;
				if (N_ > 1) oss << "[";

				for (int c = 0; c < C_; ++c) {
					if (c > 0) oss << "," << "\n" << base_indent << (N_ > 1 ? " " : "");
					if (C_ > 1) oss << "[";

					for (int h = 0; h < H_; ++h) {
						if (h > 0) oss << "," << "\n" << base_indent << (N_ > 1 ? "  " : "") << (C_ > 1 ? " " : "");
						oss << "[";

						if (W_ > PRINT_THRESHOLD_W) {
							for (int w = 0; w < EDGE_ITEMS_W; ++w) {
								size_t idx = static_cast<size_t>(n) * C_ * H_ * W_ +
									static_cast<size_t>(c) * H_ * W_ +
									static_cast<size_t>(h) * W_ + w;
								print_val(host_data_elements[idx]);
								if (w < EDGE_ITEMS_W - 1) oss << ", ";
							}
							oss << ", ..., ";
							for (int w = W_ - EDGE_ITEMS_W; w < W_; ++w) {
								size_t idx = static_cast<size_t>(n) * C_ * H_ * W_ +
									static_cast<size_t>(c) * H_ * W_ +
									static_cast<size_t>(h) * W_ + w;
								print_val(host_data_elements[idx]);
								if (w < W_ - 1) oss << ", ";
							}
						}
						else {
							for (int w = 0; w < W_; ++w) {
								size_t idx = static_cast<size_t>(n) * C_ * H_ * W_ +
									static_cast<size_t>(c) * H_ * W_ +
									static_cast<size_t>(h) * W_ + w;
								print_val(host_data_elements[idx]);
								if (w < W_ - 1) oss << ", ";
							}
						}
						oss << "]";
					}
					if (C_ > 1) oss << "]";
				}
				if (N_ > 1) oss << "]";
			}
			oss << "]";
		}
	}

	oss << ", dtype=" << getFriendlyTypeName(typeid(T));
	oss << ", device=cuda:" << this->Device << ")";

	return oss.str();
}



template class Tensor<__half>;
template class Tensor<float>;
template class Tensor<double>;

template class Tensor<int8_t>;
template class Tensor<int32_t>;
template class Tensor<int64_t>;
