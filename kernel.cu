#include <iostream>
#include "Tensor.cuh" // Tensor sınıfının bulunduğu başlık dosyası
#include <vector>

// getFriendlyTypeName ve GetCudnnDataType<T> şablonlarının
// global namespace'de veya erişilebilir bir yerde tanımlandığını varsayıyoruz.
// Eğer Tensor.cuh içinde değillerse, buraya da eklenebilirler veya
// Tensor.cuh'taki gibi inline bırakılabilirler.

// Önceki yanıttaki GetCudnnDataType<T> ve getFriendlyTypeName tanımlamalarının
// Tensor.cuh dosyasında doğru bir şekilde yapıldığını varsayıyorum.


#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")


int main() {
    // CUDA cihazını senkronize etmek ve hataları yakalamak için bir yardımcı
    auto checkCuda = []() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Device Sync Error: " << cudaGetErrorString(err) << std::endl;
        }
        };

    std::cout << "## 4D Tensor Chunk Testi ##" << std::endl;
    try {
        Tensor<float> TestTesor4D(2, 4, 6, 8);
        TestTesor4D.FillRandomUniform(123); // Sabit bir seed ile doldur
        checkCuda();
        std::cout << "Orijinal 4D Tensor: " << TestTesor4D.ToString() << std::endl << std::endl;

        auto ChunkA = TestTesor4D.Chunk(0, 2); // N boyutu boyunca 2 parça
        checkCuda();
        auto ChunkB = TestTesor4D.Chunk(1, 2); // C boyutu boyunca 2 parça
        checkCuda();
        auto ChunkC = TestTesor4D.Chunk(2, 2); // H boyutu boyunca 2 parça
        checkCuda();
        auto ChunkD = TestTesor4D.Chunk(3, 2); // W boyutu boyunca 2 parça
        checkCuda();

        for (size_t i = 0; i < ChunkA.size(); ++i) {
            std::cout << "Chunk A[" << i << "] (N=" << ChunkA[i].GetN() << ", C=" << ChunkA[i].GetC() << ", H=" << ChunkA[i].GetH() << ", W=" << ChunkA[i].GetW() << "): " << ChunkA[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < ChunkB.size(); ++i) {
            std::cout << "Chunk B[" << i << "] (N=" << ChunkB[i].GetN() << ", C=" << ChunkB[i].GetC() << ", H=" << ChunkB[i].GetH() << ", W=" << ChunkB[i].GetW() << "): " << ChunkB[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < ChunkC.size(); ++i) {
            std::cout << "Chunk C[" << i << "] (N=" << ChunkC[i].GetN() << ", C=" << ChunkC[i].GetC() << ", H=" << ChunkC[i].GetH() << ", W=" << ChunkC[i].GetW() << "): " << ChunkC[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < ChunkD.size(); ++i) {
            std::cout << "Chunk D[" << i << "] (N=" << ChunkD[i].GetN() << ", C=" << ChunkD[i].GetC() << ", H=" << ChunkD[i].GetH() << ", W=" << ChunkD[i].GetW() << "): " << ChunkD[i].ToString() << std::endl;
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "4D Test Hata: " << e.what() << std::endl;
    }
    checkCuda();

    std::cout << "------------------------------------" << std::endl;
    std::cout << "## 3D Tensor Chunk Testi (N, H, W) ##" << std::endl;
    // Tensor sınıfının 3D constructor'ı N, H, W alıyor ve C=1 varsayıyor.
    // Chunk(dim, numOfChunk) için dim: 0->N, (1->C anlamsız olur), 2->H, 3->W
    try {
        Tensor<float> TestTesor3D(2, 6, 8); // N=2, C=1, H=6, W=8
        TestTesor3D.FillRandomUniform(456);
        checkCuda();
        std::cout << "Orijinal 3D Tensor (N=2, C=1, H=6, W=8): " << TestTesor3D.ToString() << std::endl << std::endl;

        auto Chunk3D_N = TestTesor3D.Chunk(0, 2); // N boyutu boyunca 2 parça
        checkCuda();
        // dim=1 (C boyutu) C=1 olduğu için pek anlamlı değil ama test edilebilir. 1 parça veya 0 boyutlu parçalar üretir.
        auto Chunk3D_C = TestTesor3D.Chunk(1, 1); // C boyutu (C=1), 1 parça
        checkCuda();
        auto Chunk3D_H = TestTesor3D.Chunk(2, 2); // H boyutu boyunca 2 parça
        checkCuda();
        auto Chunk3D_W = TestTesor3D.Chunk(3, 2); // W boyutu boyunca 2 parça
        checkCuda();

        for (size_t i = 0; i < Chunk3D_N.size(); ++i) {
            std::cout << "Chunk3D_N[" << i << "] (N=" << Chunk3D_N[i].GetN() << ", C=" << Chunk3D_N[i].GetC() << ", H=" << Chunk3D_N[i].GetH() << ", W=" << Chunk3D_N[i].GetW() << "): " << Chunk3D_N[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < Chunk3D_C.size(); ++i) {
            std::cout << "Chunk3D_C[" << i << "] (N=" << Chunk3D_C[i].GetN() << ", C=" << Chunk3D_C[i].GetC() << ", H=" << Chunk3D_C[i].GetH() << ", W=" << Chunk3D_C[i].GetW() << "): " << Chunk3D_C[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < Chunk3D_H.size(); ++i) {
            std::cout << "Chunk3D_H[" << i << "] (N=" << Chunk3D_H[i].GetN() << ", C=" << Chunk3D_H[i].GetC() << ", H=" << Chunk3D_H[i].GetH() << ", W=" << Chunk3D_H[i].GetW() << "): " << Chunk3D_H[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < Chunk3D_W.size(); ++i) {
            std::cout << "Chunk3D_W[" << i << "] (N=" << Chunk3D_W[i].GetN() << ", C=" << Chunk3D_W[i].GetC() << ", H=" << Chunk3D_W[i].GetH() << ", W=" << Chunk3D_W[i].GetW() << "): " << Chunk3D_W[i].ToString() << std::endl;
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "3D Test Hata: " << e.what() << std::endl;
    }
    checkCuda();

    std::cout << "------------------------------------" << std::endl;
    std::cout << "## 2D Tensor Chunk Testi (H, W) ##" << std::endl;
    // Tensor sınıfının 2D constructor'ı H, W alıyor ve N=1, C=1 varsayıyor.
    // Chunk(dim, numOfChunk) için dim: (0->N anlamsız), (1->C anlamsız), 2->H, 3->W
    try {
        Tensor<float> TestTesor2D(6, 8); // N=1, C=1, H=6, W=8
        TestTesor2D.FillRandomUniform(789);
        checkCuda();
        std::cout << "Orijinal 2D Tensor (N=1, C=1, H=6, W=8): " << TestTesor2D.ToString() << std::endl << std::endl;

        auto Chunk2D_H = TestTesor2D.Chunk(2, 2); // H boyutu boyunca 2 parça
        checkCuda();
        auto Chunk2D_W = TestTesor2D.Chunk(3, 2); // W boyutu boyunca 2 parça
        checkCuda();

        for (size_t i = 0; i < Chunk2D_H.size(); ++i) {
            std::cout << "Chunk2D_H[" << i << "] (N=" << Chunk2D_H[i].GetN() << ", C=" << Chunk2D_H[i].GetC() << ", H=" << Chunk2D_H[i].GetH() << ", W=" << Chunk2D_H[i].GetW() << "): " << Chunk2D_H[i].ToString() << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < Chunk2D_W.size(); ++i) {
            std::cout << "Chunk2D_W[" << i << "] (N=" << Chunk2D_W[i].GetN() << ", C=" << Chunk2D_W[i].GetC() << ", H=" << Chunk2D_W[i].GetH() << ", W=" << Chunk2D_W[i].GetW() << "): " << Chunk2D_W[i].ToString() << std::endl;
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "2D Test Hata: " << e.what() << std::endl;
    }
    checkCuda();

    std::cout << "------------------------------------" << std::endl;
    std::cout << "## 1D Tensor Chunk Testi (W) ##" << std::endl;
    // Tensor sınıfının 1D constructor'ı W alıyor ve N=1, C=1, H=1 varsayıyor.
    // Chunk(dim, numOfChunk) için dim: (0,1,2 anlamsız), 3->W
    try {
        Tensor<float> TestTesor1D(8); // N=1, C=1, H=1, W=8
        TestTesor1D.FillRandomUniform(101112);
        checkCuda();
        std::cout << "Orijinal 1D Tensor (N=1, C=1, H=1, W=8): " << TestTesor1D.ToString() << std::endl << std::endl;

        auto Chunk1D_W = TestTesor1D.Chunk(3, 2); // W boyutu boyunca 2 parça
        checkCuda();

        for (size_t i = 0; i < Chunk1D_W.size(); ++i) {
            std::cout << "Chunk1D_W[" << i << "] (N=" << Chunk1D_W[i].GetN() << ", C=" << Chunk1D_W[i].GetC() << ", H=" << Chunk1D_W[i].GetH() << ", W=" << Chunk1D_W[i].GetW() << "): " << Chunk1D_W[i].ToString() << std::endl;
        }
        std::cout << std::endl;

        // Daha fazla parça testi (PyTorch davranışı: bazıları 0 boyutlu olabilir)
        std::cout << "1D Tensor Chunk Testi (W=8, numOfChunk=10)" << std::endl;
        auto Chunk1D_W_more = TestTesor1D.Chunk(3, 10);
        checkCuda();
        for (size_t i = 0; i < Chunk1D_W_more.size(); ++i) {
            std::cout << "Chunk1D_W_more[" << i << "] (N=" << Chunk1D_W_more[i].GetN() << ", C=" << Chunk1D_W_more[i].GetC() << ", H=" << Chunk1D_W_more[i].GetH() << ", W=" << Chunk1D_W_more[i].GetW() << "): " << Chunk1D_W_more[i].ToString() << std::endl;
        }
        std::cout << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "1D Test Hata: " << e.what() << std::endl;
    }
    checkCuda();


    std::cout << "------------------------------------" << std::endl;
    std::cout << "## Özel Parçalama Durumları Testi ##" << std::endl;
    try {
        Tensor<float> TestTesorEdge(1, 1, 1, 5); // N=1, C=1, H=1, W=5
        TestTesorEdge.FillRandomUniform(131415);
        checkCuda();
        std::cout << "Orijinal Kenar Durumu Tensörü (N=1, C=1, H=1, W=5): " << TestTesorEdge.ToString() << std::endl << std::endl;

        std::cout << "Parça Sayısı > Boyut (W=5, numOfChunk=7)" << std::endl;
        auto ChunkEdge_W_gt = TestTesorEdge.Chunk(3, 7);
        checkCuda();
        for (size_t i = 0; i < ChunkEdge_W_gt.size(); ++i) {
            std::cout << "ChunkEdge_W_gt[" << i << "] (N=" << ChunkEdge_W_gt[i].GetN() << ", C=" << ChunkEdge_W_gt[i].GetC() << ", H=" << ChunkEdge_W_gt[i].GetH() << ", W=" << ChunkEdge_W_gt[i].GetW() << "): " << ChunkEdge_W_gt[i].ToString() << std::endl;
        }
        std::cout << std::endl;

        std::cout << "numOfChunk = 1 (W=5, numOfChunk=1)" << std::endl;
        auto ChunkEdge_W_eq1 = TestTesorEdge.Chunk(3, 1);
        checkCuda();
        for (size_t i = 0; i < ChunkEdge_W_eq1.size(); ++i) {
            std::cout << "ChunkEdge_W_eq1[" << i << "] (N=" << ChunkEdge_W_eq1[i].GetN() << ", C=" << ChunkEdge_W_eq1[i].GetC() << ", H=" << ChunkEdge_W_eq1[i].GetH() << ", W=" << ChunkEdge_W_eq1[i].GetW() << "): " << ChunkEdge_W_eq1[i].ToString() << std::endl;
        }
        std::cout << std::endl;

        Tensor<float> TestTesorZeroDim(1, 0, 1, 1); // C=0
        std::cout << "Orijinal Sıfır Boyutlu Tensör (N=1, C=0, H=1, W=1): " << TestTesorZeroDim.ToString() << std::endl << std::endl;
        std::cout << "Sıfır Boyutlu Tensörü Parçalama (C=0, dim=1, numOfChunk=2)" << std::endl;
        auto ChunkZero_C = TestTesorZeroDim.Chunk(1, 2);
        checkCuda();
        for (size_t i = 0; i < ChunkZero_C.size(); ++i) {
            std::cout << "ChunkZero_C[" << i << "] (N=" << ChunkZero_C[i].GetN() << ", C=" << ChunkZero_C[i].GetC() << ", H=" << ChunkZero_C[i].GetH() << ", W=" << ChunkZero_C[i].GetW() << "): " << ChunkZero_C[i].ToString() << std::endl;
        }
        std::cout << std::endl;


    }
    catch (const std::exception& e) {
        std::cerr << "Özel Durum Test Hata: " << e.what() << std::endl;
    }
    checkCuda();


    std::cout << "Testler tamamlandı." << std::endl;

    return 0;
}