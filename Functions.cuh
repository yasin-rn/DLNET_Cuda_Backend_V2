#pragma once
#include <cublas_v2.h>
#include <cudnn.h>
#include "Tensor.cuh"
#include <iostream>

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
	static void Add(cudnnHandle_t handle, Tensor<T>& A, Tensor<T>& Bias, T alpha, T beta)
	{
		cudnnAddTensor(handle, &alpha, Bias.GetDesc(), Bias.GetData(), &beta, A.GetDesc(), A.GetData());
	}


	template <typename T>
	static void ActivationForward(cudnnHandle_t handle, Tensor<T>& input, Tensor<T>& output, cudnnActivationMode_t activation, T alpha, T beta)
	{
		cudnnActivationDescriptor_t activationDesc;
		cudnnCreateActivationDescriptor(&activationDesc);
		cudnnSetActivationDescriptor(activationDesc, activation, CUDNN_PROPAGATE_NAN, 0.0);

		cudnnActivationForward(handle, activationDesc, &alpha, input.GetDesc(), input.GetData(), &beta, output.GetDesc(), output.GetData());
		cudnnDestroyActivationDescriptor(activationDesc);
	}

	template <typename T>
	static void SoftmaxForward(cudnnHandle_t handle, Tensor<T>& input, Tensor<T>& output, T alpha, T beta)
	{

		cudnnStatus_t status = cudnnSoftmaxForward(
			handle,
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			input.GetDesc(),
			input.GetData(),
			&beta,
			output.GetDesc(),
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

		cudnnGetReductionWorkspaceSize(handle, reduceDesc, input.GetDesc(), output.GetDesc(), &workspaceSize);
		cudaMalloc(&workspaceArea, workspaceSize);

		cudnnStatus_t status =
			cudnnReduceTensor(
				handle,
				reduceDesc,
				nullptr, 0,
				workspaceArea, workspaceSize,
				&alpha, input.GetDesc(), input.GetData(),
				&beta, output.GetDesc(), output.GetData()
			);

		cudnnDestroyReduceTensorDescriptor(reduceDesc);
		cudaFree(workspaceArea);
	}

	template <typename T>
	static void MeanVariance(cublasHandle_t cublasHandle, cudnnHandle_t cudnnHandle, Tensor<T>& input, Tensor<T>& mean, Tensor<T>& variance)
	{
		int N = input.GetLen(0);
		int W = input.GetLen(1);

		ReduceTensor(cudnnHandle, input, mean, CUDNN_REDUCE_TENSOR_AVG);

		Tensor<T> X_Sub_Mean = input;

		T val_one = static_cast<T>(1.0);
		T val_minus_one = static_cast<T>(-1.0);

		Functions::Add(cudnnHandle, X_Sub_Mean, mean, val_minus_one, val_one);

		X_Sub_Mean.Reshape(N, 1, W);

		T alpha_for_variance = static_cast<T>(1.0) / static_cast<T>(W);
		T beta_for_variance = static_cast<T>(0.0);

		Matmul(cublasHandle,
			X_Sub_Mean,
			X_Sub_Mean,
			variance,
			false,
			true,
			alpha_for_variance,
			beta_for_variance);

		variance.Reshape(N, 1);
	}



private:

};
