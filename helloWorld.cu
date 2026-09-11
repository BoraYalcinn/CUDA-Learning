#include <stdlib.h>
#include <iostream>


__global__ void hello(void ){

    printf("Hello from gpu !\n");

}


int main(){

    hello<<<1,1>>>();


    cudaError_t err;
    err = cudaDeviceSynchronize(); // are there any runtime errors ?
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }


    return 0;
}