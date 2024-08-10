
/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define ll long long 
using namespace std;

using std::cin;
using std::cout;


__global__ void solve(long long int *ans, long long int *gmat, long long int *gfilter, int m, int n, int k){

  extern __shared__ long long int filter[]; 
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x; 

  if ( tid < m * n){
    int row = tid/n; 
    int col = tid % n;
    int workperthread = ceil( (k*k)/(float)n); 

    for(int i = threadIdx.x; i < k*k; i += workperthread){
        filter[i] = gfilter[i]; 
    }

    __syncthreads(); 

    long long int sum = 0; 
    for(int i = row - k/2, r = 0; i < min(row + k/2 + 1, m); i++, r++){
        if ( i < 0 ) continue; 
        for(int j = col - k/2, c = 0; j < min (col + k/2 + 1, n); j++, c++){
            if ( j < 0) continue; 
            sum += filter[r*k + c] * gmat[i*n + j]; 
        }
    }
    ans[ row*n + col] = sum; 
  }

}

int main(int argc, char** argv) {

    int m, n, k; 
    cin >> m >> n >> k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];

    long long int *gmat, *gfilter, *ans; 

    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    cudaMalloc(&gmat, (m*n)*sizeof(long long int));     
    cudaMemcpy(gmat, h_mat, (m*n)*sizeof(long long int), cudaMemcpyHostToDevice);     
    cudaMalloc(&gfilter, (k*k)*sizeof(long long int)); 
    cudaMemcpy(gfilter, h_filter, (k*k)*sizeof(long long int), cudaMemcpyHostToDevice); 
    cudaMalloc(&ans, (m*n)*sizeof(long long int));                 
    
    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    
    int nblocks = ceil( (m*n)/ (float)1024); 
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    solve <<< nblocks, 1024, (k*k)*sizeof(long long int) >>> (ans, gmat, gfilter, m, n, k); 
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    
    cudaFree(gmat); 
    cudaFree(gfilter); 
    cudaMemcpy(h_ans, ans, (m*n)*sizeof(long long int), cudaMemcpyDeviceToHost);     
    cudaFree(ans); 
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;    
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
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
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}