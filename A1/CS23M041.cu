/**
*   CS6023: GPU Programming 
*   Assignment 1
*   
*   Please don't change any existing code in this file.
*
*   You can add your code whereever needed. Please add necessary memory APIs
*   for your implementation. Use cudaFree() to free up memory as soon as you're
*   done with an allocation. This will ensure that you don't run out of memory 
*   while running large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>

using std::cin;
using std::cout;


__global__
void CalculateHadamardProduct(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    unsigned i = (bid * threadsPerBlock + tid)/ N;
    unsigned j = (bid * threadsPerBlock + tid) % N;
    unsigned cur = i * N + j;
    if ( cur < N*N ){
        A[cur] = A[cur] * B[ j*N + i];
    }
}


__global__
void FindWeightMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x * blockDim.x + threadIdx.y;
    unsigned threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    unsigned i = (bid * threadsPerBlock + tid)/ N;
    unsigned j = (bid * threadsPerBlock + tid) % N;
    unsigned cur = i * N + j;
    if ( cur < N*N && B[cur] > A[cur]){
          A[cur] = B[cur];
    }
}

__global__
void CalculateFinalMatrix(long int* A, long int* B, int N) {

    // TODO: Write your kernel here
    unsigned bid = blockIdx.x * gridDim.y + blockIdx.y;
    unsigned tid = threadIdx.x * blockDim.x + threadIdx.y;
    unsigned threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    unsigned i = (bid * threadsPerBlock + tid)/ ( 2* N);
    unsigned j = (bid * threadsPerBlock + tid) % ( 2* N);
    unsigned cur = i * 2 * N + j;
    if ( cur < 4*N*N){
        B[cur] *= A[ (i%N)*N + j%N];
    }
}

int main(int argc, char** argv) {


    int N;
    cin >> N;
    long int* A = new long int[N * N];
    long int* B = new long int[N * N];
    long int* C = new long int[N * N];
    long int* D = new long int[2 * N * 2 * N];


    for (long int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> C[i];
    }

    for (long int i = 0; i < 2 * N * 2 * N; i++) {
        cin >> D[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    */

    long int* d_A;
    long int* d_B;
    long int* d_C;
    long int* d_D;

    cudaMalloc(&d_A, N * N * sizeof(long int));
    cudaMemcpy(d_A, A, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, N * N * sizeof(long int));
    cudaMemcpy(d_B, B, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, N * N * sizeof(long int));
    cudaMemcpy(d_C, C, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_D, 4 * N * N * sizeof(long int));
    cudaMemcpy(d_D, D, 4 * N * N * sizeof(long int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(N * N / 1024.0), 1, 1);


    auto start = std::chrono::high_resolution_clock::now();
    CalculateHadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();    

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    cudaFree(d_B); 

    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(N * N / 1024.0), 1, 1);


    start = std::chrono::high_resolution_clock::now();
    FindWeightMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    cudaDeviceSynchronize();    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    cudaFree(d_C); 

    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);


    start = std::chrono::high_resolution_clock::now();
    CalculateFinalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_D, N);
    cudaDeviceSynchronize();    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end - start;
    cudaFree(d_A); 
    // Make sure your final output from the device is stored in d_D.

    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */

    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(d_D); 
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < 2 * N; i++) {
            for (long int j = 0; j < 2 * N; j++) {
                file << D[i * 2 * N + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2 << elapsed2.count() << "\n";
        file2 << elapsed3.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}