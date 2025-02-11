#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define ll long long
using namespace std;

__device__ int gT;
//*******************************************

// Write down the kernels here

__global__ void initialize(int T, int *health, int H){
    unsigned tid = threadIdx.x; 
    if ( tid == 0){
        gT = T;    
    }
    health[tid] = H; 
}

__global__ void K2(int *health, int *updateHealth, int *alive, int round){
    if ( round % gT == 0) return ; 

    int srcTank = threadIdx.x; 

    if ( health[srcTank] <= 0) return;    

    health[srcTank] += updateHealth[srcTank];    

    if ( health[srcTank] <= 0) {
        atomicAdd(alive, -1);        
    }
}

__global__ void K1(int *health, int *updateHealth, ll *dist, int *finaldestTank,
    int *dev_score, int *dev_xcoord, int *dev_ycoord, int round){    

    if ( round % gT == 0) return ; 
    int srcTank = blockIdx.x;
    
    if ( health[srcTank] <= 0) return;    
  
    if ( threadIdx.x == 0){
      dist[srcTank] = 1e18;             
    }      
    __syncthreads(); 
    
    int destTank = ( srcTank + round) % gT;
    int j = threadIdx.x; 
    
    ll slope_num1 = dev_ycoord[destTank] - dev_ycoord[srcTank];
    ll slope_den1 = dev_xcoord[destTank] - dev_xcoord[srcTank];

    ll slope_num2 = dev_ycoord[j] - dev_ycoord[srcTank];
    ll slope_den2 = dev_xcoord[j] - dev_xcoord[srcTank];    

    ll curDist = 1e18;         

    bool flag = true; 

    if ( slope_num1 * slope_den2 != slope_den1 *  slope_num2 || j == srcTank || health[j] <= 0){
        flag = false; 
    }
    else{
        if (health[destTank] <= 0){                                            
            // slopes are same                            
            if ( slope_den1 == 0){ // vertical line

                if ( dev_ycoord[srcTank] < dev_ycoord[destTank] && dev_ycoord[j] > dev_ycoord[srcTank]) // +ve line
                    destTank = j;
                
                else if ( dev_ycoord[srcTank] > dev_ycoord[destTank] && dev_ycoord[j] < dev_ycoord[srcTank]) // -ve line
                    destTank = j;

                else flag = false; 
            }
            else{
                if (dev_xcoord[destTank] > dev_xcoord[srcTank] && dev_xcoord[j] > dev_xcoord[srcTank]){ // +ve line
                    destTank = j; 
                }

                else if (dev_xcoord[destTank] < dev_xcoord[srcTank] && dev_xcoord[j] < dev_xcoord[srcTank]){ // -ve line
                    destTank = j; 
                }                                                                

                else flag = false; 
            }                                                            
        }
        else{
            if ( dev_xcoord[j] < min (dev_xcoord[srcTank], dev_xcoord[destTank] ) || dev_xcoord[j] > max (dev_xcoord[srcTank], dev_xcoord[destTank] ))
                flag = false;
            else if ( dev_ycoord[j] < min (dev_ycoord[srcTank], dev_ycoord[destTank] ) || dev_ycoord[j] > max (dev_ycoord[srcTank], dev_ycoord[destTank] ))
                flag = false;
            else
                destTank = j;             
        }   
    }    
    if ( flag ){
        curDist = slope_num2*slope_num2 + slope_den2*slope_den2; 
        atomicMin(&dist[srcTank], curDist); 
    }

    __syncthreads(); 

    if (flag && dist[srcTank] == curDist){
        dev_score[srcTank] += 1;
        atomicAdd(&updateHealth[destTank], -1);
    }   
}
//***********************************************



int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int *alive;
    cudaHostAlloc(&alive, sizeof(int), 0);
    *alive = T;

    int round = 1;    
    
    
    thrust::device_vector<int> health(T, H);
    int *health_ptr = thrust::raw_pointer_cast(health.data());

    thrust::device_vector<int> updateHealth(T, 0);
    int *updateHealth_ptr = thrust::raw_pointer_cast(updateHealth.data());

    thrust::device_vector<int> dev_score(T, 0);
    int *dev_score_ptr = thrust::raw_pointer_cast(dev_score.data());

    thrust::device_vector<int> dev_xcoord(xcoord, xcoord + T);
    int *dev_xcoord_ptr = thrust::raw_pointer_cast(dev_xcoord.data());

    thrust::device_vector<int> dev_ycoord(ycoord, ycoord + T);
    int *dev_ycoord_ptr = thrust::raw_pointer_cast(dev_ycoord.data());

    thrust::device_vector<int> finaldestTank(T);
    int *finaldestTank_ptr = thrust::raw_pointer_cast(finaldestTank.data());

    thrust::device_vector<ll> dist(T);
    ll *dist_ptr = thrust::raw_pointer_cast(dist.data());

    initialize<<< 1 , T >>> (T, health_ptr, H);
    cudaDeviceSynchronize();

    while( *alive > 1 ){        

        K1 <<< T , T>>> (health_ptr, updateHealth_ptr, dist_ptr, finaldestTank_ptr, dev_score_ptr, dev_xcoord_ptr, dev_ycoord_ptr, round);
        cudaDeviceSynchronize();

        K2 <<< 1 , T>>> (health_ptr, updateHealth_ptr, alive, round);
        cudaDeviceSynchronize();

        thrust::fill(updateHealth.begin(), updateHealth.end(), 0);                                 
        ++round;        
    }    
    cudaMemcpy(score, dev_score_ptr, T * sizeof(int), cudaMemcpyDeviceToHost); 

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************
    
    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}