/*
 * Local memory version
 * MMX/IMX/DMX are stored in local memory
*/


#ifndef _SHAREDMEMORY_KERNEL_FWD_
#define _SHAREDMEMORY_KERNEL_FWD_

extern "C" __global__
void KERNEL(unsigned int* seq, unsigned int total, unsigned int* offset,
double* sc, int* L, unsigned int* L_6r, float* mat, float* ins, float* tran,
int QV, double mu, double lambda)									
{
	volatile __shared__ unsigned int cache[blockDim.y][blockDim.x]; //Temp var for element of sequence
	volatile __shared__ float MMX[blockSize.y][Q * 32];
	volatile __shared__ float IMX[blockSize.y][Q * 32];
	volatile __shared__ float DMX[blockSize.y][Q * 32];
	const int operThread = blockDim.y * threadIdx.x + threadIdx.y; //Column of vectors
	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y; //Index of seq
	float mmx, imx, dmx;
	float sv, dcv;
	int NCJ_MOVE;
	unsigned int LEN, OFF, res, res_s; //Mb use __shared__ ??
	int xE, xJ, xB, xN, xC;
	int q, i, j, z ; //indexes
	
	// 0xff800000 = -infinity
	xE = 0xff800000;
	xJ = 0xff800000;
	xB = 0xff800000; //!!!!!!!!!!!!!!!!
	xN = 0xff800000;
	xC = 0xff800000;
	mmx = 0xff800000;
	imx = 0xff800000;
	dmx = 0xff800000;
	sv = 0xff800000;
	
	
	NCJ_MOVE = 0xff800000;
	
	LEN = 0;
	OFF = 0;
	res = 0;
	res_s = 0;
	
	for (q = 0; q < Q; q++)
	{
		
		MMX[threadIdx.y][q * 32 + threadIdx.x] = 0xff800000; 
		IMX[threadIdx.y][q * 32 + threadIdx.x] = 0xff800000;
		DMX[threadIdx.y][q * 32 + threadIdx.x] = 0xff800000;
	}
	
	//Later here must be while for total > 768
	LEN = L_6r[seqIdx];
	OFF = offset[seqIdx];
	
	//NCJ_MOVE = rintf(logf(3.0f / (float)(L[seqIdx] + 3.0f)));
	
	for (i = 0; i < LEN; i += 32)
	{
		dcv = 0xff800000;
		xE = 0xff800000;
		
		
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
				
				mmx = MMX[threadIdx.y][(Q - 1) * 32 + threadIdx.x];
				imx = IMX[threadIdx.y][(Q - 1) * 32 + threadIdx.x];
				dmx = DMX[threadIdx.y][(Q - 1) * 32 + threadIdx.x];
				
				for (q = 0; q < Q; q++)
				{
					//match state
					sv = fadd4(xB, __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x])); //B_M
					sv = flogsum(sv, (mmx + __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]))); //M_M
					sv = flogsum(sv, (imx + __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]))); //I_M
					sv = flogsum(sv, (dmx + __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]))); //D_M
					sv = fadd4(sv, __ldg(&mat[res_s + q * 32 + threadIdx.x]));
					xE = flogsum(sv, xE);
					
					mmx = MMX[threadIdx.y][q * 32 + threadIdx.x];
					imx = IMX[threadIdx.y][q * 32 + threadIdx.x];
					dmx = DMX[threadIdx.y][q * 32 + threadIdx.x];
					
					MMX[threadIdx.y][q * 32 + threadIdx.x] = sv;
					DMX[threadIdx.y][q * 32 + threadIdx.x] = dcv;
					//delete state
					dcv = flogsum(sv, __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]); //M_D
					//insert state
					sv = flogsum(mmx, (imx + __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]))); //M_I
					sv = flogsum(sv, (imx + __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]))); //I_I
					IMX[threadIdx.y][q * 32 + threadIdx.x] = sv;
				}
			}
		}
	}
	
}


#endif /* _SHAREDMEMORY_KERNEL_FWD_ */
