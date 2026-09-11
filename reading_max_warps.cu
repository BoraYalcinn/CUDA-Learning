#include <stdio.h>
#include <cuda_runtime.h>



int main(){

    int device;
    cudaGetDevice(&device); // get current CUDA device
    cudaDeviceProp prop;


    cudaGetDeviceProperties(&prop,device);
    
    printf("Max_threads_per_SM  0: %d \n", prop.maxThreadsPerMultiProcessor);
    printf("Max_warps_per_SM    0: %d \n", prop.maxThreadsPerMultiProcessor / 32);

    int maxThreadsPerMP = 0;
    
    cudaDeviceGetAttribute(&maxThreadsPerMP,cudaDevAttrMaxThreadsPerMultiProcessor,device);

    printf("Max_threads_per_SM  0: %d \n", maxThreadsPerMP);
    printf("Max_warps_per_SM    0: %d \n", maxThreadsPerMP/32);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)printf("Error has been occured: %s\n",cudaGetErrorString(err));

    return 0;
}