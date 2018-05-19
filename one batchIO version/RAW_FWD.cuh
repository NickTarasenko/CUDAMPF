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
	float MMX[Q];
	float IMX[Q];
	float DMX[Q];

	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y;
	unsigned int LEN, OFF, res, res_s;

	float mmx, imx, dmx, dcv;
	float NCJ_MOVE, sv;

	float    xN, xE, xB, xC, xJ;
	volatile __shared__ float xEv, xBv, totscale;

	int i, q, j, z, h; //indexes

	if (threadIdx.y != 0 && blockIdx.y != 0) return;
	//printf("FWD ##### warp: %d ## thread: %d #####Init...\n", threadIdx.y, threadIdx.x);

	NCJ_MOVE = 3.0f / (float)(L[seqIdx] + 3.0f);

	xE = 0.0f;
	xJ = 0.0f;
	if (threadIdx.x == 0) xB = NCJ_MOVE; 
	xN = 1.0f;
	xC = 0.0f;

	mmx = 1.0f;
	imx = 1.0f;
	dmx = 1.0f;
	
	totscale = 0.0f;
	
	//NCJ_MOVE = 0.0f;
	
	LEN = 0;
	OFF = 0;
	res = 0;
	res_s = 0;
	
	//init_flogsum();
	
	for (q = 0; q < Q; q++)
	{
		
		MMX[q] = 0.0f; 
		IMX[q] = 0.0f;
		DMX[q] = 0.0f;
	}

	//Later here must be while for total > 768
	LEN = L_6r[seqIdx];
	OFF = offset[seqIdx];
	
	NCJ_MOVE = (float) 3.0f / (float)(L[seqIdx] + 3.0f);
	//printf("%f      %f\n", NCJ_MOVE, 3.0f);

	//nullsv = (float) LEN * logf((float) LEN / (LEN + 1)) + log(1.0- (float) LEN / (LEN + 1));
	
	for (i = 0; i < LEN; i += 32)
	{		
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
				//xBt = xB;

				mmx = MMX[Q - 1];
				//if (z == 1 && (threadIdx.x == 0 || threadIdx.x == 1)) printf("thread %d : %f\n", threadIdx.x, mmx); 
				reorder_float32(mmx);
				//if (z == 1 && (threadIdx.x == 1 || threadIdx.x == 2)) printf("thread %d : %f\n", threadIdx.x, mmx); 
				imx = IMX[Q - 1];
				reorder_float32(imx);
				dmx = DMX[Q - 1];
				reorder_float32(dmx);
				
				for (q = 0; q < Q; q++)
				{
					// sv - accumulator 
					//match state
					sv = xB * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
					//printf("%f    %f\n", xBt, xB);
					//if (threadIdx.x == 0) printf("FWD # warp: %d ## thread: %d # xB(%f) * tran(%f) = sv(%f\n)", threadIdx.y, threadIdx.x, xB, __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]), sv);
					sv = sv + mmx * __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]); //M_M
					sv = sv + imx * __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]); //I_M
					sv = sv + dmx * __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]); //D_M
					//sv = sv + sv1;
					sv = sv * __ldg(&mat[res_s + q * 32 + threadIdx.x]);
					//printf("t = %d q = %d: %f\n", threadIdx.x, q, xBt * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]));

					xE = sv + xE;
					//if (threadIdx.x == 0) printf("FWD # warp: %d ## thread: %d # xE = %f\n", threadIdx.y, threadIdx.x,xE);

					mmx = MMX[q];
					imx = IMX[q];
					dmx = DMX[q]; 
					
					MMX[q] = sv;
					DMX[q] = dcv;

					//delete state
					dcv = sv * __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]); //M_D

					//insert state
					sv = mmx * __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]); //M_I
					sv = sv + imx * __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]); //I_I
					IMX[q] = sv; //INS[]?
				}
				__syncthreads();
				//For D_D path
				dcv = __shfl_up(dcv, (threadIdx.x + 31) % 32); 
				DMX[0] = 0.0f;
				for (q = 0; q < Q; q++)
				{
					DMX[q] = dcv + DMX[q];
					dcv = DMX[q] * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]); //D_D
				}
				
				// Serialization (?)
				for (h = 0; h < 4; h++)
				{
					int cv = 0;

					reorder_float32(dcv);
					for (q = 0; q < Q; q++)
					{
						sv = dcv + DMX[q];
						cv = cv | ((sv >= DMX[q]) ? 1 : 0);
						DMX[q] = sv;
						dcv = dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
					} 

					if (cv == 0) break;
				}

				__syncthreads();

				//xEv = 0.0f;
				for (q = 0; q < Q; q++) xE += DMX[q];

				xE = xE + __shfl_down(xE, 16);
				xE = xE + __shfl_down(xE, 8);
				xE = xE + __shfl_down(xE, 4);
				xE = xE + __shfl_down(xE, 2);
				xE = xE + __shfl_down(xE, 1);

				if (threadIdx.x == 0)
				{
					xN = xN * NCJ_MOVE;
					xC = xC * NCJ_MOVE + xE * e_lm;
					xJ = xJ * NCJ_MOVE + xE * e_lm;
					xBv = xJ * NCJ_MOVE + xN * NCJ_MOVE;
					xEv = xE;
				}

				__syncthreads();

				if (xEv > 1.0e4)
				{
					if (threadIdx.x == 0)
					{
						xN = xN / xE;
						xC = xC / xE;
						xJ = xJ / xE;
						xBv = xBv / xE;
					}

					__syncthreads();

					xE = 1.0f / xEv;
					for (q = 0; q < Q; q++)
					{
						MMX[q] = MMX[q] * xE;
						IMX[q] = IMX[q] * xE;
						DMX[q] = DMX[q] * xE;
					}

					totscale += log(xEv);
					xEv = 1.0f;
				}
				
				__syncthreads();
			}
		}
	}
	
	//check over\underflow and NaN(?)
	if (threadIdx.x == 0) sc[seqIdx] = totscale + xC * NCJ_MOVE; //- nullsv) / logf(2);
}


#endif /* _SHAREDMEMORY_KERNEL_FWD_ */
