#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

//Ejercicio 5.1 - Función Kernel Fig. 5.13 Modificada
__global__
void sum13(float* X, float *Y) 
{ 
	__shared__ float partialSum[SIZE];
	partialSum[threadIdx.x] = 0;
	unsigned int t = threadIdx.x;
	
	if ((blockIdx.x*blockDim.x + threadIdx.x) < SIZE) 
	{
		partialSum[t] = X[blockIdx.x*blockDim.x + threadIdx.x];
	}
	__syncthreads();
	
	for (unsigned int stride = 1; stride < 2048; stride *= 2)
	{
		if (t % (2 * stride) == 0) 
		{
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) 
	{
		Y[blockIdx.x] = partialSum[0];
	}
}

//Ejercicio 5.1 - Función Kernel Fig. 5.15 Modificada
__global__
void sum15(float* X, float* Y) 
{
	__shared__ float partialSum[SIZE];
	partialSum[threadIdx.x] = 0;
	
	unsigned int t = threadIdx.x;
	
	if ((blockIdx.x*blockDim.x + threadIdx.x) < SIZE) 
	{
		partialSum[t] = X[i];
	}
	__syncthreads();
	
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) 
	{
		if (t < stride)
		{
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) 
	{
		Y[blockIdx.x] = partialSum[0];
	}
}

//Ejercicio 5.3
__global__
void exercise3(float* X, float* Y) 
{
	__shared__ float partialSum[SIZE];
	partialSum[t] = 0;
	unsigned int t = threadIdx.x;
	
	if ((blockIdx.x*(blockDim.x*2) + threadIdx.x) < SIZE) 
	{
		partialSum[t] = X[(blockIdx.x*(blockDim.x*2) + threadIdx.x)] + X[(blockIdx.x*(blockDim.x*2) + threadIdx.x) + blockDim.x];
	}
	__syncthreads();
	
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) 
	{
		if (t < stride) 
		{
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) 
	{
		Y[blockIdx.x] = partialSum[0];
	}
}

float execute_sum(float* x, int size) 
{
	float total_sum = 0;
	int block_size = 1024;
	int max_size = block_size;
	int grid_size;
	
	if (x_sz <= max_sz) 
	{
		grid_size = (int)ceil(float(size) / float(max_size));
	}
	else 
	{
		grid_size = size / max_size;
		if ((size % max_size) != 0) 
			grid_size++;
	}
	
	float *d_block;
	cudaMalloc(&d_block, sizeof(float)*grid_size);
	cudaMemset(d_block, 0, sizeof(float)*grid_size);
	
	exercise3 <<< grid_size, block_size, sizeof(float)*max_size >>> (x, d_block);
	
	if (grid_size <= max_size) 
	{
		float* d_total;
		cudaMalloc(&d_total, sizeof(float));
		cudaMemset(d_total, 0, sizeof(float));
		exercise3 <<< 1, block_size, sizeof(float)*max_size >>> (d_total, d_block);
		cudaMemcpy(&total_sum, d_total, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_total);
	}
	else 
	{
		float* d_block2;
		cudaMalloc(&d_block2, sizeof(float)*grid_size);
		cudaMemcpy(&d_block2, d_block, sizeof(float)*grid_size, cudaMemcpyDeviceToDevice);
		total_sum = execute_sum(d_block2, grid_size);
		cudaFree(d_block2);
	}
	cudaFree(d_block2);
	return total_sum;
}

int main() 
{
	float *h_X;
	
	int size = 1024;

	h_X = (float*)malloc(size * sizeof(float));

	for (int i = 0; i < size; i++) 
	{
		h_X[i] = i + 1.0f;
	}

	float* d_X;
	cudaMalloc(&d_X, sizeof(unsigned int) * size);
	cudaMemcpy(d_X, h_X, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);

	float final = execute_sum(d_X, size);

	cudaFree(d_X);
	free(h_X);

	return 0;
}
