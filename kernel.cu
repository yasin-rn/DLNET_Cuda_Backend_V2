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
			B.GetData(), B.CudaDataType, lda,
			A.GetData(), A.CudaDataType, ldb,
			&beta,
			C.GetData(), C.CudaDataType, ldc,
			C.CudaDataType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
			A.GetLen(0), C.CudaDataType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

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
			A.GetLen(0) + A.GetLen(1), C.CudaDataType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		 
		break;
	default:
		break;
	}

}


int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	Tensor<__nv_fp8_e5m2> A(5, 8);
	Tensor<__nv_fp8_e5m2> B(4, 6);
	Tensor<__nv_fp8_e5m2> C1(5, 3);
	Tensor<__nv_fp8_e5m2> C2(5, 3);

	__nv_fp8_e5m2 alpha = static_cast<__nv_fp8_e5m2>(1.0f);
	__nv_fp8_e5m2 beta = static_cast<__nv_fp8_e5m2>(0.0f);

	A.FillRandomUniform();
	B.FillRandomUniform();

	auto A_chunks = A.Chunk(1, 2);
	auto B_chunks = B.Chunk(1, 2);

	Matmul(handle, A_chunks[0], B_chunks[0], C1, false, false, alpha, beta);
	Matmul(handle, A_chunks[1], B_chunks[1], C2, false, false, alpha, beta);

	std::cout << "A[0]:" << A_chunks[0].ToString() << std::endl;
	std::cout << "B[0]:" << B_chunks[0].ToString() << std::endl;
	std::cout << "C1:" << C1.ToString() << std::endl;

	std::cout << "A[1]:" << A_chunks[1].ToString() << std::endl;
	std::cout << "B[1]:" << B_chunks[1].ToString() << std::endl;
	std::cout << "C2:" << C2.ToString() << std::endl;

	cublasDestroy(handle);


	return 0;
}