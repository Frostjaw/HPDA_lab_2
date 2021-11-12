
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__global__ void find_occurances(char* text, int* occurances, int n) 
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) 
    {
        char character = text[id];
        int index = character - 97;
        //atomicAdd(&occurances[character], 1);
        atomicAdd(&occurances[index], 1);
    }
}

int main()
{
    int n_char = 26;

    int n;
    cout << "Enter size of text: \n";
    cin >> n;

    char* text_gpu;

    size_t bytes = n * sizeof(char);
    cudaMallocManaged(&text_gpu, bytes);

    srand((unsigned)time(0));
    for (int i = 0; i < n; i++) {
        text_gpu[i] = 'a' + rand() % 26;
    }

    cout << "GPU implementation:" << endl;

    int* occurances_gpu;

    size_t bytes_occurances = n_char * sizeof(int);
    cudaMallocManaged(&occurances_gpu, bytes_occurances);

    for (int i = 0; i < n_char; i++) {
        occurances_gpu[i] = 0;
    }

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    auto gpu_t1 = high_resolution_clock::now();
    find_occurances<<<blocks, threads>>>(text_gpu, occurances_gpu, n);
    cudaDeviceSynchronize();
    auto gpu_t2 = high_resolution_clock::now();

    auto gpu_exe_time_ms = duration_cast<milliseconds>(gpu_t2 - gpu_t1);

    cout << "Execution time: " << gpu_exe_time_ms.count() << "ms" << endl;

    //cout << "input:" << endl;
    //for (int i = 0; i < n; i++) {
    //    cout << text[i] << endl;
    //}

    cout << "occurances:" << endl;
    for (int i = 0; i < n_char; i++) {
        cout << static_cast<char>(i + 97) << ": " << occurances_gpu[i] << endl;
    }

    int sum_gpu = 0;
    for (int i = 0; i < n_char; i++) {
        sum_gpu += occurances_gpu[i];
    }

    cout << "sum: " << sum_gpu << endl;

    // CPU
    cout << endl << "CPU implementation:" << endl;

    char* text_cpu;

    text_cpu = (char*)malloc(bytes);

    memcpy(text_cpu, text_gpu, bytes);

    int* occurances_cpu;
    occurances_cpu = (int*)malloc(bytes_occurances);

    for (int i = 0; i < n_char; i++) {
        occurances_cpu[i] = 0;
    }

    auto cpu_t1 = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        char character = text_cpu[i];
        int index = character - 97;
        occurances_cpu[index]++;
    }
    auto cpu_t2 = high_resolution_clock::now();

    auto cpu_exe_time_ms = duration_cast<milliseconds>(cpu_t2 - cpu_t1);

    cout << "Execution time: " << cpu_exe_time_ms.count() << "ms" << endl;

    //cout << "input:" << endl;
    //for (int i = 0; i < n; i++) {
    //    cout << text[i] << endl;
    //}

    cout << "occurances:" << endl;
    for (int i = 0; i < n_char; i++) {
        cout << static_cast<char>(i + 97) << ": " << occurances_cpu[i] << endl;
    }

    int sum_cpu = 0;
    for (int i = 0; i < n_char; i++) {
        sum_cpu += occurances_cpu[i];
    }

    cout << "sum: " << sum_cpu;

    free(text_cpu);
    free(occurances_cpu);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}