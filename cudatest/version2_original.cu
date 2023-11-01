#include <stdio.h>

#define WIDTH 2048 // WIDTH x WIDTH matrix
#define TILE_WIDTH 32 //tile width, should be TILE_WIDTH * TILE_WIDTH < 1024 (maxmum thread for 0ne block)

__global__ void MatrixMulKernel (float *Md, float *Nd, float *Pd, int Width) {
        __shared__ float Mds[TILE_WIDTH][TILE_WIDTH] ;
        __shared__ float Nds[TILE_WIDTH][TILE_WIDTH] ;

        int bx = blockIdx.x ;
        int by = blockIdx.y ;
        int tx = threadIdx.x ;
        int ty = threadIdx.y ;

        int Row = by * TILE_WIDTH + ty ;
        int Col = bx * TILE_WIDTH + tx ;

        float Pvalue = 0 ;

        for (int m = 0; m < Width/TILE_WIDTH; ++m) {
                Mds[ty][tx] = Md[Row*Width + m*TILE_WIDTH + tx] ;
                Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty) * Width + Col]  ;
                __syncthreads() ;

                for (int k = 0; k < TILE_WIDTH; ++k)
                        Pvalue += Mds[ty][k] * Nds[k][tx] ;
                __syncthreads() ;
        }

        Pd[Row*Width + Col] = Pvalue ;
}

int main () {
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
                M[k] = 1. ;
                N[k*WIDTH] = 1. ;
        }

        float *Md, *Nd, *Pd ;
        dim3 dimGrid (WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH) ;
        dim3 dimBlock (TILE_WIDTH, TILE_WIDTH) ;
        cudaEventRecord (start,0) ;

        cudaMalloc ((void**) &Md, size) ;
        cudaMemcpy (Md, M, size, cudaMemcpyHostToDevice) ;
        cudaMalloc ((void**) &Nd, size) ;
        cudaMemcpy (Nd, N, size, cudaMemcpyHostToDevice) ;
        cudaMalloc ((void**) &Pd, size) ;

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
