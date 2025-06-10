#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

template <typename T>
__global__ void ConcatKernel(T* output_data, int output_total_size,
    const T** input_data_ptrs,
    const int* input_strides, // 2D gibi düşün: [t0_sN, t0_sC, t0_sH, t0_sW, t1_sN, ...]
    const int* input_dims,    // 2D gibi düşün: [t0_N, t0_C, t0_H, t0_W, t1_N, ...]
    const int* cumulative_dims,
    int concat_dim_index, int num_tensors)
{
    // 1. Her thread için global ve benzersiz bir indeks hesapla
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_total_size) {
        return; // Toplam boyutu aşan thread'ler hiçbir şey yapmasın
    }

    // 2. Bu thread'in sorumlu olduğu elemanın hedef tensördeki N,C,H,W koordinatlarını bul
    int out_dims[4] = { 0 };
    int temp_idx = idx;

    // Hedef tensörün stride'larını on-the-fly hesapla
    int out_c_dim = input_dims[1]; // Varsayılan olarak ilk tensörün boyutu
    int out_h_dim = input_dims[2];
    int out_w_dim = input_dims[3];
    if (concat_dim_index == 1) out_c_dim = cumulative_dims[num_tensors - 1];
    if (concat_dim_index == 2) out_h_dim = cumulative_dims[num_tensors - 1];
    if (concat_dim_index == 3) out_w_dim = cumulative_dims[num_tensors - 1];

    int out_stride_n = out_c_dim * out_h_dim * out_w_dim;
    int out_stride_c = out_h_dim * out_w_dim;
    int out_stride_h = out_w_dim;

    out_dims[0] = temp_idx / out_stride_n;
    temp_idx %= out_stride_n;
    out_dims[1] = temp_idx / out_stride_c;
    temp_idx %= out_stride_c;
    out_dims[2] = temp_idx / out_stride_h;
    out_dims[3] = temp_idx % out_stride_h;

    // 3. Hedef koordinatına bakarak hangi kaynak tensörden veri alınacağını bul
    int source_tensor_idx = 0;
    int coord_to_check = out_dims[concat_dim_index];
    for (int i = 0; i < num_tensors; ++i) {
        if (coord_to_check < cumulative_dims[i]) {
            source_tensor_idx = i;
            break;
        }
    }

    // 4. Kaynak tensör içindeki yerel koordinatları hesapla
    int in_dims[4];
    in_dims[0] = out_dims[0];
    in_dims[1] = out_dims[1];
    in_dims[2] = out_dims[2];
    in_dims[3] = out_dims[3];

    int prev_cumulative_dim = (source_tensor_idx > 0) ? cumulative_dims[source_tensor_idx - 1] : 0;
    in_dims[concat_dim_index] = out_dims[concat_dim_index] - prev_cumulative_dim;

    // 5. Kaynak tensörün stride'larını ve data pointer'ını al
    const int* src_strides = input_strides + source_tensor_idx * 4;
    const T* src_data = input_data_ptrs[source_tensor_idx];

    // 6. Kaynak tensördeki doğrusal adresi (offset) hesapla
    size_t source_offset = (size_t)in_dims[0] * src_strides[0] +
        (size_t)in_dims[1] * src_strides[1] +
        (size_t)in_dims[2] * src_strides[2] +
        (size_t)in_dims[3] * src_strides[3];

    // 7. Kopyalamayı gerçekleştir!
    output_data[idx] = src_data[source_offset];
}