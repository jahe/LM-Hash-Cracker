
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t desWithCuda(unsigned long*, const unsigned long*, const unsigned long*, unsigned int);
__device__ unsigned long des(const unsigned long, const unsigned long);
__device__ unsigned long ip(const unsigned long);

__global__ void desKernel(unsigned long *ciphertext, const unsigned long *plaintext, const unsigned long *k)
{
    int i = threadIdx.x;
	ciphertext[i] = des(plaintext[i], k[i]);
}
/* --------------------------------
   DES-Verschlüsselungsfunktion
   m ist der Klartext (64 Bit)
   k ist der DES-Schlüssel (64 Bit)
  -------------------------------- */
__device__ unsigned long des(const unsigned long m, const unsigned long k)
{
	unsigned long M;	// 64 Bit Klartext
	unsigned int L;		// 32 Bit Linker Block von M
	unsigned int R;		// 32 Bit Rechter Block von M
	unsigned int C;		// 28 Bit Teilschlüssel
	unsigned int D;		// 28 Bit Teilschlüssel

	// 64 Bit DES-Schlüssel auf zwei Bitfolgen (28 Bit) C und D abbilden
	C = PC1_C(k);
	D = PC1_D(k);

	// 1. Schritt: Initiale Permutation auf m
	M = ip(m);

	// 2. Schritt: M in Linken und Rechten Block splitten
	L = (int) (M >> 32);		// logisch shiften
	R = (int) (M << 32 >> 32);

	// 3. Schritt: 16 DES-Runden
	for (int i = 1; i < 17; i++)
	{
		unsigned long K = makeRoundKey(C, D, i);
		unsigned int newL = R;
		unsigned int newR = f(R,K) ^ L;
		R = newR;
		L = newL;
	}

}

/* --------------------------------------------------------------------------
   Abbildung f
   Bildet den rechten Block R (32 Bit) und den Rundenschlüssel K (48 Bit)
   auf Bitfolge f(R,K) (32 Bit) ab.
   -------------------------------------------------------------------------- */
__device__ unsigned int f(unsigned int R, unsigned long K)
{
	unsigned long exp_R;

	// Expansionsfunktion E(R), 32 Bit -> 48 Bit
	exp_R = (unsigned long) R;

	// HIER WEITER MACHEN!!!
}

/* --------------------------------
   Abbildung PC1_C
   Bildet 64 Bit DES-Schlüssel auf 
   Bitfolge C (28 Bit) ab.
   -------------------------------- */
__device__ unsigned int PC1_C(const unsigned long k)
{
	// HIER PC1 PERMUTATION VON 64 Bit k auf 28 Bit C
}

/* --------------------------------
   Abbildung PC1_D
   Bildet 64 Bit DES-Schlüssel auf 
   Bitfolge D (28 Bit) ab.
   -------------------------------- */
__device__ unsigned int PC1_D(const unsigned long k)
{
	// HIER PC1 PERMUTATION VON 64 Bit k auf 28 Bit D
}

// makeKey erzeugt einen 48 Bit langen Rundenschlüssel
__device__ unsigned long makeRoundKey(unsigned int C, unsigned int D, int i)
{
	// v(i) ist die Anzahl der zirkulären Linksshifts
	unsigned short v;

	if (i == 1 || i == 2 || i == 9 || i == 16)
		v = 1;
	else
		v = 2;

	// C und D um v bitweise rotieren 
	C = ((C << v) | (C >> 28-v)) & 0xFFFFFFF;
	D = ((D << v) | (D >> 28-v)) & 0xFFFFFFF;

	return PC2(C, D);
}

__device__ unsigned long PC2(unsigned int C, unsigned int D)
{
	// HIER DIE PERMUTATION "PC2" auf C
}

__device__ unsigned long ip(const unsigned long m)
{
	// HIER NOCH DIE INITIALE PERMUTATION AUF m DURCHFÜHREN!
	return m;
}

int main()
{
    const int arraySize = 3;
    const unsigned long klartext[arraySize] = { 1, 2, 3};
    const unsigned long key[arraySize] = { 3, 2, 1 };
    unsigned long cipher[arraySize] = { 0 };

    // Verschlüsselt die Klartexte parallel.
    cudaError_t cudaStatus = desWithCuda(cipher, klartext, key, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("{%d,%d,%d} + {%d,%d,%d} = {%d,%d,%d}\n",
        klartext[0], klartext[1], klartext[2], key[0], key[1], key[0], cipher[0], cipher[1], cipher[2]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t desWithCuda(unsigned long *c, const unsigned long *p, const unsigned long *k, unsigned int size)
{
    unsigned long *klartext = 0;
    unsigned long *key = 0;
    unsigned long *cipher = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&cipher, size * sizeof(unsigned long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&klartext, size * sizeof(unsigned long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&key, size * sizeof(unsigned long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(klartext, p, size * sizeof(unsigned long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(key, k, size * sizeof(unsigned long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	// <<<dimGrid, dimBlock>>>
	desKernel<<<1, size>>>(cipher, klartext, key);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, cipher, size * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(cipher);
    cudaFree(klartext);
    cudaFree(key);
    
    return cudaStatus;
}
