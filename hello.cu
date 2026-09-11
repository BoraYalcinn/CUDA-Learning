#include <stdio.h>

__global__ void helloFromGPU() {

    // print the blocks and thread IDs
    // warp = 32 threads (128 threads/block) ----> (128/32 = 4 warp blocks)
    int warp_ID_Value = 0;
    warp_ID_Value = threadIdx.x / 32;


    printf("Hello from GPU Thread ID : %d and Warp ID :  %d !\n",threadIdx.x, warp_ID_Value);
}

int main() {
    printf("Hello from CPU!\n");
    helloFromGPU<<<1, 128>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return 0;
}
