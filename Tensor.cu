#include "Tensor.cuh"
#include <cudnn.h>


template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w) :N(n), C(c), H(h), W(w)
{

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


}
template <typename T>
Tensor<T>::Tensor(int n, int c, int h, int w, T* hostData) :N(n), C(c), H(h), W(w)
{

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

}

template <typename T>
Tensor<T>::Tensor(int n, int h, int w) :N(n), C(1), H(h), W(w)
{

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
}
template <typename T>
Tensor<T>::Tensor(int n, int h, int w, T* hostData) :N(n), C(1), H(h), W(w)
{

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
}

template <typename T>
Tensor<T>::Tensor(int h, int w) :N(1), C(1), H(h), W(w)
{
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
}
template <typename T>
Tensor<T>::Tensor(int h, int w, T* hostData) :N(1), C(1), H(h), W(w)
{
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
}

template <typename T>
Tensor<T>::Tensor(int h) :N(1), C(1), H(h), W(1)
{
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

}
template <typename T>
Tensor<T>::Tensor(int h, T* hostData) :N(1), C(1), H(h), W(1)
{
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
}



template <typename T>
Tensor<T>::Tensor(const Tensor<T>& other)
	: N(other.N), C(other.C), H(other.H), W(other.W), IsChunkPart(other.IsChunkPart), IsOwnData(true), TotalSize(other.TotalSize)
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
	}
	return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
	: N(other.N), C(other.C), H(other.H), W(other.W), IsChunkPart(other.IsChunkPart), IsOwnData(other.IsOwnData), Data(other.Data), BatchPtrs(other.BatchPtrs), TotalSize(other.TotalSize)
{
	memcpy(Strides, other.Strides, 4 * sizeof(int));
	other.Data = nullptr;
	other.BatchPtrs = nullptr;
	other.IsOwnData = false;
	other.TotalSize = 0;
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
		IsChunkPart = other.IsChunkPart;
		IsOwnData = other.IsOwnData;
		TotalSize = other.TotalSize;
		memcpy(Strides, other.Strides, 4 * sizeof(int));
		Data = other.Data;
		BatchPtrs = other.BatchPtrs;

		other.Data = nullptr;
		other.BatchPtrs = nullptr;
		other.IsOwnData = false;
		other.TotalSize = 0;
	}
	return *this;
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


std::string getFriendlyTypeName(const std::type_info& ti) {
	if (ti == typeid(float)) return "float32";
	if (ti == typeid(double)) return "float64";
	if (ti == typeid(int)) return "int32";
	if (ti == typeid(__half)) return "float16";
	if (ti == typeid(__nv_fp8_e5m2)) return "fp8_e5m2";
	if (ti == typeid(__nv_fp8_e4m3)) return "fp8_e4m3";

	return ti.name();
}
template <typename T>
std::string Tensor<T>::ToString() const {
	std::ostringstream oss;


	if (Data == nullptr || TotalSize == 0) {
		return "tensor([], dtype=" + getFriendlyTypeName(typeid(T)) + ", device=cuda:" + std::to_string(Device) + ")";
	}

	size_t numElements = TotalSize / sizeof(T);
	std::vector<T> host_data(numElements);
	cudaError_t err = cudaMemcpy(host_data.data(), Data, TotalSize, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		oss << "Error copying data to host: " << cudaGetErrorString(err);
		return oss.str();
	}

	oss << "tensor(";
	oss << std::fixed << std::setprecision(4);


	auto print_val = [&](T val) {
		if constexpr (std::is_same_v<T, __half>) {
			oss << __half2float(val);
		}
		else if constexpr (std::is_same_v<T, __nv_fp8_e5m2> ||
			std::is_same_v<T, __nv_fp8_e4m3>) {
			oss << static_cast<float>(val);
		}
		else {
			oss << std::fixed << std::setprecision(4) << static_cast<float>(val);

		}
		};


	const int PRINT_THRESHOLD_W = 10;
	const int EDGE_ITEMS_W = 3;

	int N_ = N, C_ = C, H_ = H, W_ = W;

	if (numElements == 0) {
		oss << "[]";
	}
	else {

		if (N_ == 1 && C_ == 1 && H_ == 1 && W_ == 1) {
			print_val(host_data[0]);

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
								print_val(host_data[idx]);
								if (w < EDGE_ITEMS_W - 1) oss << ", ";
							}
							oss << ", ..., ";
							for (int w = W_ - EDGE_ITEMS_W; w < W_; ++w) {
								size_t idx = static_cast<size_t>(n) * C_ * H_ * W_ +
									static_cast<size_t>(c) * H_ * W_ +
									static_cast<size_t>(h) * W_ + w;
								print_val(host_data[idx]);
								if (w < W_ - 1) oss << ", ";
							}
						}
						else {
							for (int w = 0; w < W_; ++w) {
								size_t idx = static_cast<size_t>(n) * C_ * H_ * W_ +
									static_cast<size_t>(c) * H_ * W_ +
									static_cast<size_t>(h) * W_ + w;
								print_val(host_data[idx]);
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
	oss << ", device=cuda:" << Device << ")";

	return oss.str();
}


template class Tensor<float>;
template class Tensor<double>;

template class Tensor<__half>;
template class Tensor<__nv_fp8_e5m2>;
template class Tensor<__nv_fp8_e4m3>;
template class Tensor<__nv_fp8_e8m0>;
template class Tensor<__nv_fp4_e2m1>;