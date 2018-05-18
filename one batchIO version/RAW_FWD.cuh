/*
 * Local memory version
 * MMX/IMX/DMX are stored in local memory
*/

#ifndef _SHAREDMEMORY_KERNEL_FWD_
#define _SHAREDMEMORY_KERNEL_FWD_

extern "C" __global__ 
void KERNEL(unsigned int* seq, unsigned int total, unsigned int* offset,
double* sc, int* L,  unsigned int* L_6r, float* mat, float* tran,
int e_lm, int QV, double mu, double lambda)									
{
	volatile __shared__ unsigned int cache[RIB][32]; //Temp var for element of sequence
	float MMX[Q]; //must check max size (can be overflow on GPU)
	float IMX[Q];
	float DMX[Q];
	//const int operThread = blockDim.y * threadIdx.x + threadIdx.y; //Column of vectors
	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y; //Index of seq
	float mmx, imx, dmx;
	float sv, sv1, dcv;
	int NCJ_MOVE;
	unsigned int LEN, OFF, res, res_s; //Mb use __shared__ ??
	float xE, xJ, xN, xC, xBt, nullsv;
	__shared__ float xB;
	int q, i, j, z, h ; //indexes
	float totscale;

	if (threadIdx.y != 0 && blockIdx.y != 0) return;
	//printf("FWD ##### warp: %d ## thread: %d #####Init...\n", threadIdx.y, threadIdx.x);

	NCJ_MOVE = rintf(logf(3.0f / (float)(L[seqIdx] + 3.0f)));

	xE = 0.0f;
	xJ = 0.0f;
	xB = NCJ_MOVE; 
	xN = 1.0f;
	xC = 0.0f;
	mmx = 0.0f;
	imx = 0.0f;
	dmx = 0.0f;
	sv = 0.0f;
	sv1 = 0.0f;
	dcv = 0.0f;
	
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
	
	nullsv = (float) LEN * logf((float) LEN / (LEN + 1)) + log(1.0- (float) LEN / (LEN + 1));
	
	for (i = 0; i < LEN; i += 32)
	{
		dcv = 0.0f;
		xE = 0.0f;
		
		
		cache[threadIdx.y][threadIdx.x] = seq[OFF + i + threadIdx.x];
		
		for (j = 0; j < 32; j++)
		{
			//if (threadIdx.x == 0) printf("FWD # warp: %d ## thread: %d # Get new elem of seq...\n", threadIdx.y, threadIdx.x);

			res = cache[threadIdx.y][j];	
			if ((res & 0x000000ff) == 31) break; //Else we can use goto statment
			
			for (z = 0; z < 4; z++)
			{
				res_s = ((res >> (8 * z)) & 0x000000ff);
				if (res_s == 31) break;

				res_s *= Q * 32;

				xE = 0.0f;
				dcv = 0.0f;

				//  
				xBt = xB;

				mmx = MMX[Q - 1];
				reorder_float32(mmx);
				imx = IMX[Q - 1];
				reorder_float32(imx);
				dmx = DMX[Q - 1];
				reorder_float32(dmx);
				
				for (q = 0; q < Q; q++)
				{
					// sv - accumulator 
					//match state
					sv = xBt + __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
					sv = flogsum(sv, (mmx + __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]))); //M_M
					sv1 = imx + __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]); //I_M
					sv1 = flogsum(sv1, (dmx + __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]))); //D_M
					sv = flogsum(sv, sv1);
					sv = sv + __ldg(&mat[res_s + q * 32 + threadIdx.x]);
					
					xE = sv + xE;
					

					mmx = MMX[q];
					imx = IMX[q];
					dmx = DMX[q]; 
					
					MMX[q] = sv;
					DMX[q] = dcv;

					//insert state
					sv1 = mmx + __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]); //M_I
					sv1 = flogsum(sv1, (imx + __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]))); //I_I
					IMX[q] = sv1; //INS[]?

					//delete state
					dcv = sv + __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]); //M_D
				}
				
				//For D_D path
				reorder_float32(dcv); // ~vec_sld() .. Криво сделанный?
				DMX[0] = 0.0f;
				for (q = 0; q < Q; q++)
				{
					DMX[q] = flogsum(dcv, DMX[q]);
					dcv = DMX[q] + __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]); //D_D
				}
				
				// Serialization (?)
				for (h = 0; h < 4; h++)
				{
					int cv = 0;

					reorder_float32(dcv);
					for (q = 0; q < Q; q++)
					{
						sv = flogsum(dcv, DMX[q]);
						cv = cv | ((sv > DMX[q]) ? 1 : 0);
						DMX[q] = sv;
						dcv = dcv + __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
					} 

					if (cv == 0) break;
				}

				__syncthreads();

				
				for (q = 0; q < Q; q++) xE = DMX[q] + xE;

				xE = xE + __shfl_xor(xE, 16);
				xE = xE + __shfl_xor(xE, 8);
				xE = xE + __shfl_xor(xE, 4);
				xE = xE + __shfl_xor(xE, 2);
				xE = xE + __shfl_xor(xE, 1);
				
				__syncthreads();

				if (threadIdx.x == 0)
				{
					xN = xN + NCJ_MOVE; //?
					xC = flogsum(xE + e_lm, xC + NCJ_MOVE);
					xJ = flogsum(xE + e_lm, xJ + NCJ_MOVE);
					xB = flogsum(xJ + NCJ_MOVE, xN + NCJ_MOVE);

					/*xC = xE + e_lm;
					xJ = xE + e_lm;
					xB = flogsum(xJ + NCJ_MOVE, xN + NCJ_MOVE);*/
					
					if (xE > 1.0e4)
					{
						xN = xN / xE;
						xC = xC / xE;
						xJ = xJ / xE;
						xB = xB / xE;
						xE = 1.0 / xE;
						
						for (q = 0; q < Q; q++)
						{
							MMX[q] = MMX[q] + xE;
							DMX[q] = DMX[q] + xE;
							IMX[q] = IMX[q] + xE;
						}
						
						totscale = totscale + xE;
						xE = 1.0;
					}
				}

				__syncthreads();
			}
		}
	}
	
	//check over\underflow and NaN(?)
	if (threadIdx.x == 0) sc[seqIdx] = (flogsum(totscale, xC + NCJ_MOVE)); //- nullsv) / logf(2);
}


#endif /* _SHAREDMEMORY_KERNEL_FWD_ */
