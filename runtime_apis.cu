#include <stdio.h>
#include <cuda_runtime.h>


// A CODE THAT WILL DISPLAY YOUR DEVICE PROPERTIES !

int main(){

    int nDevices;

    cudaGetDeviceCount(&nDevices);

    for( int i = 0;i < nDevices; i++){

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        
        printf("Device number: %d\n",i);
        printf("Device name: %s\n",prop.name);

        int clockRateKHz;
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrMemoryClockRate, i);
        printf("Memory Clock Rate (KHz): %d\n", clockRateKHz);

        printf("Memory Bus Width (bits): %d\n",prop.memoryBusWidth);
        printf("Peak Memory Bandwith (GB/s): %f\n",2.*clockRateKHz*(prop.memoryBusWidth/8)/1.0e6);
        printf("Total Global Memory: %lu\n",prop.totalGlobalMem);
        printf("Compute Capability: %d.%d\n",prop.major,prop.minor);
        printf("Number of SMs: %d\n",prop.multiProcessorCount);
        printf("Max Threads per Block: %d\n",prop.maxThreadsPerBlock);
        printf("Max Grid Dimension: x = %d, y = %d, z = %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
        

        cudaError_t err = cudaDeviceSynchronize();
        if(err != cudaSuccess){
            printf("Error occured: %s\n",cudaGetErrorString(err));
        }

        

    }

    return 0;

}