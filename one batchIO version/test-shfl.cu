#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <helper_functions.h>

int main(int argc, char* argv[])
{
	int* test[4] = {3, 8, 9, 1};
	int sum = 0;
	
	CUdeviceptr vec, res;
	
	cuMemAlloc(&vec, 4 * sizeof(int));
	cuMemAlloc(&res, sizeof(int));
	cuMemcpyHtoD(vec, test, sum * sizeof(int));
	
	vec_sum<<<1, 4>>>(vec, res);
	
	cuMemcpyDtoH(sum, res, number * sizeof(double));
	
	printf("Expexted 21, calculated %d", sum);
	
	return 0;
}

__global__ void vec_sum(unsigned int vec, unsigned int res)
{
	vec[threadIdx.x] = vec[threadIdx.x] + __shfl_xor(vec, 2);
	vec[threadIdx.x] = vec[threadIdx.x] + __shfl_xor(vec, 1);
	
	if (threadIdx.x == 0) res = vec[0];
}
