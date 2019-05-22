#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;

// Compute vector sum C = A + B
void vecAddCPU(float *h_A, float* h_B, float* h_C, int n) {
	for (int i = 0; i < n; i++) {
		h_C[i] = h_A[i] + h_B[i];
	}
}

// Compute vector sum C = A + B
// Each thread performs onne pair-wise addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

void vecAddGPU(float* A, float* B, float* C, int n) {
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc(&d_C, size);

	chrono::time_point<chrono::system_clock> GPU_Start, GPU_End;
	GPU_Start = chrono::system_clock::now();
	vecAddKernel <<< ceil(n/256.0), 256 >>> (d_A, d_B, d_C, n);
	GPU_End = chrono::system_clock::now();

	cout << "GPU: " << chrono::duration_cast<chrono::nanoseconds>(GPU_End - GPU_Start).count() << "ns." << endl;

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	//Free device memory for A, B, C
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	// Memory allocation for h_A, h_B and h_C
	float *h_A, *h_B, *h_C;
	int n = 0;
	// I/O to read h_A and h_B, N elements each
	//cout << "Insert N:" << endl;
	//cin >> n;
	n = 100000000;
	h_A = (float*)malloc(n * sizeof(float));
	h_B = (float*)malloc(n * sizeof(float));
	h_C = (float*)malloc(n * sizeof(float));
	for (int i = 0; i < n; i++) {
		h_A[i] = i;
		h_B[i] = i;
		h_C[i] = 0;
	}
	
	chrono::time_point<chrono::system_clock> CPU_Start, CPU_End;

	CPU_Start = chrono::system_clock::now();
	vecAddCPU(h_A, h_B, h_C, n);
	CPU_End = chrono::system_clock::now();

	cout << "CPU: " << chrono::duration_cast<chrono::nanoseconds>(CPU_End - CPU_Start).count() << "ns." << endl;
	
	vecAddGPU(h_A, h_B, h_C, n);

	cout<<"Primera respuesta: "<<h_C[0]<<endl;	
	
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
