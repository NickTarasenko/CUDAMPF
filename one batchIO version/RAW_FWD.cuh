/*
 * Local memory version
 * MMX/IMX/DMX are stored in local memory
*/

#include "header_def.h"
#include "simd_def.h"
#include "simd_function.h"


#ifndef _SHAREDMEMORY_KERNEL_FWD_
#define _SHAREDMEMORY_KERNEL_FWD_

extern "C" __global__ 
void KERNEL(unsigned int* seq, unsigned int total, unsigned int* offset,
double* sc, int* L,  unsigned int* L_6r, float* mat, float* tran,
int e_lm, int QV, double mu, double lambda)									
{
	volatile __shared__ unsigned int cache[blockDim.y][blockDim.x]; //Temp var for element of sequence
	float MMX[Q]; //must check max size (can be overflow on GPU)
	float IMX[Q];
	float DMX[Q];
	//const int operThread = blockDim.y * threadIdx.x + threadIdx.y; //Column of vectors
	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y; //Index of seq
	float mmx, imx, dmx;
	float sv, dcv;
	int NCJ_MOVE;
	unsigned int LEN, OFF, res, res_s; //Mb use __shared__ ??
	float xE, xJ, xB, xN, xC;
	int q, i, j, z ; //indexes
	float totscale;
	
	xE = 0.0f;
	xJ = 0.0f;
	xB = 0.0f; 
	xN = 1.0f;
	xC = 0.0f;
	mmx = 0.0f;
	imx = 0.0f;
	dmx = 0.0f;
	sv = 0.0f;
	
	totscale = 0.0f;
	
	NCJ_MOVE = 0.0f;
	
	LEN = 0;
	OFF = 0;
	res = 0;
	res_s = 0;
	
	init_flogsum();
	
	for (q = 0; q < Q; q++)
	{
		
		MMX[q] = 0.0f; 
		IMX[q] = 0.0f;
		DMX[q] = 0.0f;
	}
	
	//Later here must be while for total > 768
	LEN = L_6r[seqIdx];
	OFF = offset[seqIdx];
	
	NCJ_MOVE = rintf(logf(3.0f / (float)(L[seqIdx] + 3.0f)));
	
	for (i = 0; i < LEN; i += 32)
	{
		dcv = 0.0f;
		xE = 0.0f;
		
		
		cache[threadIdx.y][threadIdx.x] = seq[OFF + i + threadIdx.x];
		
		for (j = 0; j < 32; j++)
		{
			res = cache[threadIdx.y][j];	
			if ((res & 0x000000ff) == 31) break; //Else we can use goto statment
			
			for (z = 0; z < 4; z++)
			{
				res_s = ((res >> (8 * z)) & 0x000000ff);
				if (res_s == 31) break;
				res_s *= Q * 32;
				
				mmx = MMX[Q - 1];
				reorder_float32(mmx);
				imx = IMX[Q - 1];
				reorder_float32(imx);
				dmx = DMX[Q - 1];
				reorder_float32(dmx);
				
				for (q = 0; q < Q; q++)
				{
					//match state
					sv = xB + __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
					sv = flogsum(sv, (mmx + __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]))); //M_M
					sv = flogsum(sv, (imx + __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]))); //I_M
					sv = flogsum(sv, (dmx + __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]))); //D_M
					sv = sv + __ldg(&mat[res_s + q * 32 + threadIdx.x]);
					xE = flogsum(sv, xE);
					
					mmx = MMX[q];
					imx = IMX[q];
					dmx = DMX[q];
					
					MMX[q] = sv;
					DMX[q] = dcv;
					//delete state
					dcv = flogsum(sv, __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x])); //M_D
					//insert state
					sv = flogsum(mmx, (imx + __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]))); //M_I
					sv = flogsum(sv, (imx + __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]))); //I_I
					IMX[q] = sv;
				}
				
				//For D_D path
				reorder_float32(dcv);
				DMX[0] = 0.0f;
				for (q = 0; q < Q; 1++)
				{
					DMX[q] = flogsum(dcv, DMX[q]);
					dcv = DMX[q] + __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]); //D_D
				}
				
				// Serialization (?)
				
				for (q = 0; q < Q; q++) xE = flogsum(DMX[q], xE);
				
				//Check this realisation
				xC = flogsum(xE + e_lm, xC);
				xJ = flogsum(xE + e_lm, xJ);
				xB = flogsum(xJ + NCJ_MOVE, xN + NCJ_MOVE);
				
				if (xE > 1.0e4)
				{
					xN = xN / xE;
					xC = xC / xE;
					xJ = xJ / xE;
					xB = xB / xE;
					xE = 1.0 / xE;
					
					for (q = 0; q < Q; q++)
					{
						MMX[q] += xE;
						DMX[q] += xE;
						IMX[q] += xE;
					}
					
					totscale = flogsum(totscale, log(xE));
					xE = 1.0;
				}
			}
		}
	}
	
	//check over\underflow and NaN(?)
	
	sc[seqIdx] = flogsum(totscale, xC + NCJ_MOVE);
}


#endif /* _SHAREDMEMORY_KERNEL_FWD_ */
