#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void test01(){
    // print the blocks and threadId's

    printf("\nThe block ID is  %d \nThe thread ID is %d\n",blockIdx.x ,threadIdx.x);

}

int main(){


    // kernel_name <<<num_of_blocks , num_of_threads_per_block>>>();
    // test01 <<<1,1>>> ();

    // add two vectors has 2048 elements
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024); // 10MB
    test01 <<<2,1024>>>();


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }



    cudaDeviceSynchronize();

    return 0;
}
