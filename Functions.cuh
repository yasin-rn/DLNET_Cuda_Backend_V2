#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include "Tensor.cuh"
#include <iostream>
#include "Functions_Kernels.cuh"

class Functions
{
public:


	static cublasHandle_t CreateCublasHandle();
	static void DestroyCublasHandle(cublasHandle_t handle);

	static cudnnHandle_t CreateCudnnHandle();
	static void DestroyCudnnHandle(cudnnHandle_t handle);

	template <typename T, typename U>
	static void Matmul(cublasHandle_t handle, Tensor<T>& A, Tensor<T>& B, Tensor<U>& C, bool aTrans, bool bTrans, U alpha = 1, U beta = 0)
	{
		int m, n, k, lda, ldb, ldc;
		cublasStatus_t status;

		switch (A.GetDimsize()) {
		case(2):

			m = bTrans ? B.GetLen(0) : B.GetLen(1);
			n = C.GetLen(0);
			k = aTrans ? A.GetLen(0) : A.GetLen(1);

			lda = B.GetStride(1);
			ldb = A.GetStride(1);
			ldc = bTrans ? B.GetLen(0) : B.GetLen(1);
			 
			status = cublasGemmEx(
				handle,
				bTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				aTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				m,
				n,
				k,
				&alpha,
				B.GetData(), B.GetCudaDataType(), lda,
				A.GetData(), A.GetCudaDataType(), ldb,
				&beta,
				C.GetData(), C.GetCudaDataType(), ldc,
				C.GetCudaDataType(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
			break;

		case(3):

			m = bTrans ? B.GetLen(1) : B.GetLen(2);
			n = C.GetLen(1);
			k = aTrans ? A.GetLen(1) : A.GetLen(2);

			lda = B.GetStride(1);
			ldb = A.GetStride(1);
			ldc = bTrans ? B.GetLen(1) : B.GetLen(2);

			status = cublasGemmBatchedEx(
				handle,
				bTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				aTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				m,
				n,
				k,
				&alpha,
				B.GetBatchPtrs(), B.GetCudaDataType(), lda,
				A.GetBatchPtrs(), A.GetCudaDataType(), ldb,
				&beta,
				C.GetBatchPtrs(), C.GetCudaDataType(), ldc,
				A.GetLen(0), C.GetCudaDataType(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);

			break;

		case(4):

			m = bTrans ? B.GetLen(2) : B.GetLen(3);
			n = C.GetLen(2);
			k = aTrans ? A.GetLen(2) : A.GetLen(3);

			lda = B.GetStride(1);
			ldb = A.GetStride(1);
			ldc = bTrans ? B.GetLen(2) : B.GetLen(3);

			cublasGemmBatchedEx(
				handle,
				bTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				aTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
				m,
				n,
				k,
				&alpha,
				B.GetBatchPtrs(), B.GetCudaDataType(), lda,
				A.GetBatchPtrs(), A.GetCudaDataType(), ldb,
				&beta,
				C.GetBatchPtrs(), C.GetCudaDataType(), ldc,
				A.GetLen(0) + A.GetLen(1), C.GetCudaDataType(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);

			break;
		default:
			break;
		}

	}

	template <typename T>
	static void Add(cudnnHandle_t handle, Tensor<T>& A, Tensor<T>& Bias, T alpha = 1, T beta = 1)
	{
		cudnnAddTensor(handle, &alpha, Bias.GetTensorDesc(), Bias.GetData(), &beta, A.GetTensorDesc(), A.GetData());
	}


	template <typename T>
	static void ActivationForward(cudnnHandle_t handle, Tensor<T>& input, Tensor<T>& output, cudnnActivationMode_t activation, T alpha = 1, T beta = 0)
	{
		cudnnActivationDescriptor_t activationDesc;
		cudnnCreateActivationDescriptor(&activationDesc);
		cudnnSetActivationDescriptor(activationDesc, activation, CUDNN_PROPAGATE_NAN, 0.0);

		cudnnActivationForward(handle, activationDesc, &alpha, input.GetTensorDesc(), input.GetData(), &beta, output.GetTensorDesc(), output.GetData());
		cudnnDestroyActivationDescriptor(activationDesc);
	}

	template <typename T>
	static void SoftmaxForward(cudnnHandle_t handle, Tensor<T>& input, Tensor<T>& output, T alpha = 1, T beta = 0)
	{
		cudnnSoftmaxForward(
			handle,
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			input.GetTensorDesc(),
			input.GetData(),
			&beta,
			output.GetTensorDesc(),
			output.GetData());
	}

	template <typename T>
	static void ReduceTensor(cudnnHandle_t handle, Tensor<T>& input, Tensor<T>& output, cudnnReduceTensorOp_t operation, T alpha = 1, T beta = 0)
	{
		cudnnReduceTensorDescriptor_t reduceDesc;
		cudnnCreateReduceTensorDescriptor(&reduceDesc);
		cudnnSetReduceTensorDescriptor(
			reduceDesc,
			operation,
			input.GetCudnnDataType(),
			CUDNN_NOT_PROPAGATE_NAN,
			CUDNN_REDUCE_TENSOR_NO_INDICES,
			CUDNN_32BIT_INDICES
		);

		size_t workspaceSize;
		void* workspaceArea;

		cudnnGetReductionWorkspaceSize(handle, reduceDesc, input.GetTensorDesc(), output.GetTensorDesc(), &workspaceSize);
		cudaMalloc(&workspaceArea, workspaceSize);

		cudnnStatus_t status =
			cudnnReduceTensor(
				handle,
				reduceDesc,
				nullptr, 0,
				workspaceArea, workspaceSize,
				&alpha, input.GetTensorDesc(), input.GetData(),
				&beta, output.GetTensorDesc(), output.GetData()
			);

		cudnnDestroyReduceTensorDescriptor(reduceDesc);
		cudaFree(workspaceArea);
	}

	template <typename T>
	static void MeanVariance(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, Tensor<T>& input, Tensor<T>& mean, Tensor<T>& variance)
	{
		int N = input.GetH();
		int W = input.GetW();

		ReduceTensor(cudnnHandle, input, mean, CUDNN_REDUCE_TENSOR_AVG);

		Tensor<T> X_Sub_Mean = input;

		T val_one = static_cast<T>(1.0);
		T val_minus_one = static_cast<T>(-1.0);

		Functions::Add(cudnnHandle, X_Sub_Mean, mean, val_minus_one, val_one);

		X_Sub_Mean.Reshape(N, 1, W);

		T alpha_for_variance = static_cast<T>(1.0) / static_cast<T>(W);
		T beta_for_variance = static_cast<T>(0.0);

		variance.Reshape(N, 1, 1);

		Matmul(cublasHandle,
			X_Sub_Mean,
			X_Sub_Mean,
			variance,
			false,
			true,
			alpha_for_variance,
			beta_for_variance);


	}

	template <typename T>
	static void LayerNormForward(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, Tensor<T>& input, Tensor<T>& output, Tensor<T>& scale, Tensor<T>& bias, T alpha = 1, T beta = 0, double epsilon = 1e-6)
	{
		int Channel = input.GetLen(0);
		int Height = input.GetLen(1);

		Tensor<T> mean(Channel, 1);
		Tensor<T> variance(Channel, 1);

		MeanVariance(cublasHandle, cudnnHandle, input, mean, variance);


		mean.Reshape(1, Channel, 1, 1);
		variance.Reshape(1, Channel, 1, 1);

		scale.Reshape(1, Channel, 1, 1);
		bias.Reshape(1, Channel, 1, 1);

		input.Reshape(1, Channel, Height, 1);
		output.Reshape(1, Channel, Height, 1);

		cudnnNormalizationForwardInference(
			cudnnHandle,
			CUDNN_NORM_PER_CHANNEL,
			CUDNN_NORM_OPS_NORM,
			CUDNN_NORM_ALGO_STANDARD,
			&alpha,
			&beta,
			input.GetTensorDesc(), input.GetData(),
			scale.GetTensorDesc(), scale.GetData(), bias.GetData(),
			mean.GetTensorDesc(), mean.GetData(), variance.GetData(),
			nullptr, nullptr, nullptr,
			output.GetTensorDesc(), output.GetData(),
			epsilon, 1
		);
		input.Reshape(Channel, Height);
		output.Reshape(Channel, Height);
	}

	template <typename T>
	static Tensor<T> Concat(const std::vector<Tensor<T>>& tensors, int dim)
	{
		if (tensors.empty()) return Tensor<T>();

		const Tensor<T>& firstTensor = tensors[0];
		int selectedDim = 4 - firstTensor.GetDimsize() + dim;
		std::vector<int> newDim = firstTensor.GetShape();
		for (size_t i = 1; i < tensors.size(); i++) {
			newDim[selectedDim] += tensors[i].GetShape()[selectedDim];
		}
		int n = newDim[0], c = newDim[1], h = newDim[2], w = newDim[3];
		Tensor<T> concated;
		if (firstTensor.GetDimsize() == 1)      concated = Tensor<T>(w);
		else if (firstTensor.GetDimsize() == 2) concated = Tensor<T>(h, w);
		else if (firstTensor.GetDimsize() == 3) concated = Tensor<T>(n, h, w);
		else                                    concated = Tensor<T>(n, c, h, w);

		if (concated.GetTotalSize() == 0) return concated;

		int num_tensors = tensors.size();
		std::vector<const T*> h_input_data_ptrs(num_tensors);
		std::vector<int> h_input_strides(num_tensors * 4);
		std::vector<int> h_input_dims(num_tensors * 4);
		std::vector<int> h_cumulative_dims(num_tensors);
		int current_cumulative_dim = 0;

		for (int i = 0; i < num_tensors; ++i) {
			h_input_data_ptrs[i] = tensors[i].GetData();

			auto shape = tensors[i].GetShape();
			h_input_dims[i * 4 + 0] = shape[0];
			h_input_dims[i * 4 + 1] = shape[1];
			h_input_dims[i * 4 + 2] = shape[2];
			h_input_dims[i * 4 + 3] = shape[3];

			h_input_strides[i * 4 + 3] = 1; 
			h_input_strides[i * 4 + 2] = shape[3]; 
			h_input_strides[i * 4 + 1] = shape[2] * shape[3]; 
			h_input_strides[i * 4 + 0] = shape[1] * shape[2] * shape[3]; 

			current_cumulative_dim += shape[selectedDim];
			h_cumulative_dims[i] = current_cumulative_dim;
		}

		const T** d_input_data_ptrs;
		int* d_input_strides, * d_input_dims, * d_cumulative_dims;

		cudaMalloc(&d_input_data_ptrs, num_tensors * sizeof(T*));
		cudaMalloc(&d_input_strides, num_tensors * 4 * sizeof(int));
		cudaMalloc(&d_input_dims, num_tensors * 4 * sizeof(int));
		cudaMalloc(&d_cumulative_dims, num_tensors * sizeof(int));

		cudaMemcpy(d_input_data_ptrs, h_input_data_ptrs.data(), num_tensors * sizeof(T*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_input_strides, h_input_strides.data(), num_tensors * 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_input_dims, h_input_dims.data(), num_tensors * 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_cumulative_dims, h_cumulative_dims.data(), num_tensors * sizeof(int), cudaMemcpyHostToDevice);

		int output_total_size = concated.GetTotalSize();
		int block_size = 256;
		int grid_size = (output_total_size + block_size - 1) / block_size;

		ConcatKernel<T> << <grid_size, block_size >> > (
			concated.GetData(), output_total_size, d_input_data_ptrs,
			d_input_strides, d_input_dims, d_cumulative_dims,
			selectedDim, num_tensors
			);

		cudaDeviceSynchronize();

		cudaFree(d_input_data_ptrs);
		cudaFree(d_input_strides);
		cudaFree(d_input_dims);
		cudaFree(d_cumulative_dims);

		return concated;
	}

private:

};
