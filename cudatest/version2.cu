//version2 루프풀기

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
                float value=0;
                Mds[ty][tx] = Md[Row*Width + m*TILE_WIDTH + tx] ;
                Nds[ty][tx] = Nd[(m*TILE_WIDTH + ty) * Width + Col]  ;
                __syncthreads() ;

value = Mds[ty][0] * Nds[0][tx]+Mds[ty][1] * Nds[1][tx]+Mds[ty][2] * Nds[2][tx]+Mds[ty][3] * Nds[3][tx]+Mds[ty][4] * Nds[4][tx]
        +Mds[ty][5] * Nds[5][tx]+Mds[ty][6] * Nds[6][tx]+Mds[ty][7] * Nds[7][tx]+Mds[ty][8] * Nds[8][tx]+Mds[ty][9] * Nds[9][tx]
        +Mds[ty][10] * Nds[10][tx]+Mds[ty][11] * Nds[11][tx]+Mds[ty][12] * Nds[12][tx]+Mds[ty][13] * Nds[13][tx]+Mds[ty][14] * Nds[14][tx]
        +Mds[ty][15] * Nds[15][tx]+Mds[ty][16] * Nds[16][tx]+Mds[ty][17] * Nds[17][tx]+Mds[ty][18] * Nds[18][tx]+Mds[ty][19] * Nds[19][tx]
        +Mds[ty][20] * Nds[20][tx]+Mds[ty][21] * Nds[21][tx]+Mds[ty][22] * Nds[22][tx]+Mds[ty][23] * Nds[23][tx]+Mds[ty][24] * Nds[24][tx]
        +Mds[ty][25] * Nds[25][tx]+Mds[ty][26] * Nds[26][tx]+Mds[ty][27] * Nds[27][tx]+Mds[ty][28] * Nds[28][tx]+Mds[ty][29] * Nds[29][tx]
        +Mds[ty][30] * Nds[30][tx]+Mds[ty][31] * Nds[31][tx]+Mds[ty][32] * Nds[32][tx];

__syncthreads();
               Pvalue += value;
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
        printf ("32x32: P[0]=%f\n", P[0]) ;

        cudaFree (Md) ;
        cudaFree (Nd) ;
        cudaFree (Pd) ;

        free (M) ;
        free (N) ;
        free (P) ;
}
