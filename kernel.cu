#include <iostream>
#include "Tensor.cuh"
#include <vector>
#include <cublas_v2.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

template <typename T>
void Matmul(cublasHandle_t handle, Tensor<T>& A, Tensor<T>& B, Tensor<T>& C, bool aTrans, bool bTrans, T alpha = 1, T beta = 0)
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
			B.GetData(), CUDA_R_8I, lda,
			A.GetData(), CUDA_R_8I, ldb,
			&beta,
			C.GetData(), C.CudaDataType, ldc,
			CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		break;

	case(3):

		m = bTrans ? B.GetLen(1) : B.GetLen(2);
		n = C.GetLen(1);
		k = aTrans ? A.GetLen(1) : A.GetLen(2);

		lda = B.GetStride(1);
		ldb = A.GetStride(1);
		ldc = bTrans ? B.GetLen(1) : B.GetLen(2);

		cublasGemmBatchedEx(
			handle,
			bTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
			aTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
			m,
			n,
			k,
			&alpha,
			B.GetBatchPtrs(), B.CudaDataType, lda,
			A.GetBatchPtrs(), A.CudaDataType, ldb,
			&beta,
			C.GetBatchPtrs(), C.CudaDataType, ldc,
			A.GetLen(0), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

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
			B.GetBatchPtrs(), B.CudaDataType, lda,
			A.GetBatchPtrs(), A.CudaDataType, ldb,
			&beta,
			C.GetBatchPtrs(), C.CudaDataType, ldc,
			A.GetLen(0) + A.GetLen(1), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

		break;
	default:
		break;
	}

}


int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	Tensor<float> A(5, 4);
	Tensor<float> B(4, 3);
	Tensor<float> C(5, 3);

	float alpha = static_cast<float>(1.0f);
	float beta = static_cast<float>(0.0f);

	A.FillRandomUniform();
	B.FillRandomUniform();

	auto A_chunks = A.Chunk(1, 2);
	auto B_chunks = B.Chunk(1, 2);

	Matmul(handle, A, B, C, false, false, alpha, beta);

	std::cout << "A:" << A.ToString() << std::endl;
	std::cout << "B:" << B.ToString() << std::endl;
	std::cout << "C2:" << C.ToString() << std::endl;

	cublasDestroy(handle);


	return 0;
}