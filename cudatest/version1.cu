#include <stdio.h>

#define WIDTH 2048 // WIDTH x WIDTH matrix
#define TILE_WIDTH 32 //tile width, should be TILE_WIDTH * TILE_WIDTH < 1024 (maxmum thread for 0ne block)

__global__ void MatrixMulKernel (float *Md, float *Nd, float *Pd, int Width) {
        int Row = blockIdx.y * TILE_WIDTH + threadIdx.y ;
        int Col = blockIdx.x * TILE_WIDTH + threadIdx.x ;

        float Pvalue = 0 ;

        for (int k = 0; k < Width; ++k) {
                Pvalue += Md[Row*Width + k] * Nd[k*Width + Col];
        }

        Pd[Row * Width + Col] = Pvalue ;
}

int main(){
        float *M, *N, *P ;
        int k ;
        int size = WIDTH*WIDTH*sizeof(float) ;

        cudaEvent_t start, stop ;
        float time, gpu_time ;

        cudaEventCreate (&start) ;
        cudaEventCreate (&stop) ;
        M = (float *) malloc (size) ;
        N = (float *) malloc (size) ;
        P = (float *) malloc (size) ;

        for (k=0; k<WIDTH; ++k) {
                for (int m=0; m<WIDTH; ++m) {
                        M[k*WIDTH + m] = 1. ;
                        N[k*WIDTH + m] = 1. ;
                }
        }



        // Fig 3.10, 3.11, 3.14
        dim3 dimGrid (WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH) ;
        dim3 dimBlock (TILE_WIDTH, TILE_WIDTH) ;
        float *Md, *Nd, *Pd ;
        cudaEventRecord (start,0) ;

        cudaMalloc ((void**) &Md, size) ;
        cudaMemcpy (Md, M, size, cudaMemcpyHostToDevice) ;
        cudaMalloc ((void**) &Nd, size) ;
        cudaMemcpy (Nd, N, size, cudaMemcpyHostToDevice) ;
        cudaMalloc ((void**) &Pd, size) ;

        // 2. GPU version
        MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, WIDTH) ;

        cudaMemcpy (P, Pd, size, cudaMemcpyDeviceToHost) ;

        cudaEventRecord (stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime (&time, start, stop) ;
        gpu_time = time ;
        printf ("GPU time=%f \n",gpu_time) ;
        printf ("Fig 5.7: P[0]=%f\n", P[0]) ;

        cudaFree (Md) ;
        cudaFree (Nd) ;
        cudaFree (Pd) ;

        free (M) ;
        free (N) ;
        free (P) ;
}
