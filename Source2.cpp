#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


#define MAXR 16
#define MAXC 16
#define BINS 32
#define N (MAXR*MAXC)
__global__ void count(int* arrayONE_d, int* occurrences_d, int* occurrences_final_d) {

    //provide unique thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
#ifndef USE_ALT_KERNEL
    if (idx < N) {
        //for(k=0; k < MAXR*MAXC; k++) {
        for (int j = 0; j < 32; j++) {
            if (arrayONE_d[idx] == occurrences_d[j]) {
#ifndef USE_ATOMICS
                occurrences_final_d[j]++;
#else
                atomicAdd(occurrences_final_d + j, 1);
#endif

            }
            else {}


        }
    }
#else
    // use one thread per histo bin
    if (idx < BINS) {
        int count = 0;
        int myval = occurrences_d[idx];
        for (int i = 0; i < N; i++) if (arrayONE_d[i] == myval) count++;
        occurrences_final_d[idx] = count;
    }

#endif
}


int main(void) {



    //const int N = MAXR*MAXC;

    int arr1_h[MAXR][MAXC];
    //int *occurrences_h[0][32];
    //creating arrays for the device (GPU)
    //int *arr1_d;
    int occurrences_h[32]; // mod
    int* occurrences_d;

    int occurrences_final_h[32] = { 0 };  // mod
    int* occurrences_final_d;

    int arrayONE_h[256] = { 0 };  // mod
    int* arrayONE_d;

    int i, j;

    // allocating memory for the arrays on the device
    cudaMalloc((void**)&arrayONE_d, MAXR * MAXC * sizeof(int)); // change to 16384 when using 128x128
    cudaMalloc((void**)&occurrences_d, 32 * sizeof(int));
    cudaMalloc((void**)&occurrences_final_d, 32 * sizeof(int));

     // this loop takes the information from .txt file and puts it into arr1 matrix
    for (i = 0; i < MAXR; i++) {


        for (j = 0; j < MAXC; j++)
        {
            //            fscanf(fp,"%d\t", &arr1_h[i][j]);
            arr1_h[i][j] = j;  // mod
        }

    }

    for (i = 0; i < MAXR; i++) {

        for (j = 0; j < MAXC; j++) {
            //printf("d\t", arr1_h[i][j]);
        }

    }


    int x, y;
    int z = 0;
    // this loop flattens the 2d array and makes it a 1d array of length MAXR*MAXC
    for (x = 0; x < MAXR; x++)
    {
        for (y = 0; y < MAXC; y++)
        {
            //  printf("**%d   ",arr1_h[x][y]);

            arrayONE_h[z] = arr1_h[x][y];  // mod
            z++;

        }
    }

    int length = sizeof(arrayONE_h) / sizeof(arrayONE_h[0]);

    printf("**LENGTH = %d\n", length);

    // copying the arrays/memory from the host to the device (GPU)
    cudaMemcpy(arrayONE_d, arrayONE_h, MAXR * MAXC * sizeof(int), cudaMemcpyHostToDevice);  //mod
    cudaMemcpy(occurrences_d, occurrences_h, 32 * sizeof(int), cudaMemcpyHostToDevice);   // mod
    cudaMemcpy(occurrences_final_d, occurrences_final_h, 32 * sizeof(int), cudaMemcpyHostToDevice); // mod

    // how many blocks we will allocate
    //dim3 DimGrid();
    //how many threads per block we will allocate
#ifndef USE_ALT_KERNEL
    dim3 DimBlock(N);
#else
    dim3 DimBlock(BINS);
#endif
    //kernel launch against the GPU
    count << <1, DimBlock >> > (arrayONE_d, occurrences_d, occurrences_final_d);

    //copy the arrays post-computation from the device back to the host (CPU)
    cudaMemcpy(occurrences_final_h, occurrences_final_d, 32 * sizeof(int), cudaMemcpyDeviceToHost); // mod
    cudaMemcpy(occurrences_h, occurrences_d, 32 * sizeof(int), cudaMemcpyDeviceToHost);  // mod

    // some error checking - run this with cuda-memcheck when executing your code
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    //free up the memory of the device arrays
    cudaFree(arrayONE_d);
    cudaFree(occurrences_d);
    cudaFree(occurrences_final_d);

    //print out the number of occurrences of each 0-31 value
    for (i = 0; i < 32; i++) {
        printf("%d ", occurrences_final_h[i]);

    }
    printf("\n");
}