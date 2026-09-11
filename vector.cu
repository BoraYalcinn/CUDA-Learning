#include <stdio.h>
#include "cuda_runtime.h"

#define SIZE 1024 // size of the vectors

__global__ void add(int* A,int* B,int* C,int n){

    int i = threadIdx.x;
    C[i] = A[i] + B[i];

}

int main(){

    int *A ,*B,*C; // host vectors
    int *d_A , *d_B , *d_C; // device vectors

    int size = SIZE * sizeof(int);


    // allocate memory for host vectors
    A = (int *)malloc(size);
    
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // allocate memory for device vectors
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    // initialize inputs
    for(int i = 0; i < SIZE ;i++){
        A[i] = i;
        B[i] = SIZE - i;
    }   

    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    // Launch the add CUDA kernel
    add <<<1,1024>>> (d_A,d_B,d_C,SIZE);


    cudaError_t err = cudaGetLastError(); // are there any launch errors?
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize(); // are there any runtime errors ?
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }


    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    printf("Execution Finished !\n");

    
    for(int i = 0; i< SIZE;i++){
        printf("%d + %d = %d ", A[i],B[i],C[i]);
        printf("\n");
    }


    // free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    

    return 0;
}