//version0

#include <stdio.h>

#define WIDTH 2048 // WIDTH x WIDTH matrix
#define TILE_WIDTH 32 //tile width, should be TILE_WIDTH * TILE_WIDTH < 1024 (maxmum thread for 0ne block)

void MatrixMultiplication (float *M, float *N, float *P, int Width) {
        for (int i = 0; i < Width; ++i)
                for (int j= 0; j < Width; ++j) {
                        float sum = 0;
                        for (int k = 0; k < Width; ++k) {
                                float a = M[i * Width + k] ;
                                float b = N[k * Width + j] ;
                                sum += a * b ;
                        }
                        P[i * Width + j] = sum ;
                }
}

int main () {
        float *M, *N, *P ;
        int k ;
        int size = WIDTH*WIDTH*sizeof(float) ;

        cudaEvent_t start, stop ;
        float time, cpu_time;

        cudaEventCreate (&start) ;
        cudaEventCreate (&stop) ;
        M = (float *) malloc (size) ;
        N = (float *) malloc (size) ;
        P = (float *) malloc (size) ;

        for (k=0; k<WIDTH; ++k) {
                M[k] = 1. ;
                N[k*WIDTH] = 1. ;
        }

        cudaEventRecord (start,0) ;
        MatrixMultiplication (M, N, P, WIDTH) ;

        cudaEventRecord (stop, 0) ;
        cudaEventSynchronize(stop) ;
        cudaEventElapsedTime (&time, start, stop) ;
        cpu_time = time ;
        printf ("CPU time=%f msec\n",cpu_time) ;
        printf ("Fig 3.4: P[0]=%f\n", P[0]) ; P[0] = -1 ;

        cudaMalloc ((void**) &M, size) ;
        cudaMalloc ((void**) &N, size) ;
        cudaMalloc ((void**) &P, size) ;

        cudaFree (M) ;
        cudaFree (N) ;
        cudaFree (P) ;


}
