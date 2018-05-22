#ifndef _LOCALMEMORY_KERNEL_FWD_
#define _LOCALMEMORY_KERNEL_FWD_

extern "C" __global__ 
void KERNEL(unsigned int* seq, unsigned int total, unsigned int* offset,
double* sc, int* L,  unsigned int* L_6r, float* mat, float* tran, float* scales,
int e_lm, int QV, double mu, double lambda)									
{
	volatile __shared__ unsigned int cache[RIB][32]; //Temp var for element of sequence
	float MMX[Q];
	float IMX[Q];
	float DMX[Q];

	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y;
	unsigned int LEN, OFF, res, res_s;

	float mmx, imx, dmx, dcv;
	float NJC_MOVE, NJC_LOOP, sv;

	float    xN, xE, xB, xC, xJ;
	volatile __shared__ float xEv[RIB], xBv[RIB], totscale[RIB];

	int i, q, j, z, h; //indexes




	//while (seqIdx + idx) < total)

	for (q = 0; q < Q; q++)
	{
		MMX[q] = 0.0f;
		IMX[q] = 0.0f;
		DMX[q] = 0.0f;
	}
	NJC_MOVE = (float)3.0f / (float)(L[seqIdx] + 3.0f);
	NJC_LOOP = 1.0f - NJC_MOVE;

	xE = 0.0f;
	xN = 1.0f;
	xJ = 0.0f;
	xBv[threadIdx.y] = NJC_MOVE;
	xC = 0.0f;

	totscale[threadIdx.y] = 0.0f;

	LEN = L_6r[seqIdx];
	OFF = offset[seqIdx];

	for (i = 0; i < LEN; i += 32)
	{
		cache[threadIdx.y][threadIdx.x] = seq[OFF + i + threadIdx.x];

		for (j = 0; j < 32; j++)
		{
			res = cache[threadIdx.y][j];	
			if ((res & 0x000000ff) == 31) break;

			for (z = 0; z < 4; z++)
			{
				res_s = ((res >> (8 * z)) & 0x000000ff);
				if (res_s == 31) break;

				res_s *= 32 * Q;

				dcv = 0.0f;
				xB = xBv[threadIdx.y];
				xE = 0.0f;

				mmx = MMX[Q - 1];
				mmx = __shfl_sync(0x1F, mmx, threadIdx.x + 1);
				imx = IMX[Q - 1];
				imx = __shfl_sync(0x1F, imx, threadIdx.x + 1);
				dmx = DMX[Q - 1];
				dmx = __shfl_sync(0x1F, dmx, threadIdx.x + 1);

				if (threadIdx.x == 0) {mmx = 0; imx = 0; dmx = 0;}

				for (q = 0; q < Q; q++)
				{
					//Match state
					sv = xB * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
					/*if (i+j+z == 1 && threadIdx.x == 25) printf("threadIdx = %d # i = %d ## xB(%f) * tran(%f) = sv(%f)\n", 
						threadIdx.x, i+j+z, xB, __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]), sv);*/
					sv += mmx * __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]); //M_M
					sv += imx * __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]); //I_M
					sv += dmx * __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]); //D_M
					sv = sv * __ldg(&mat[res_s + q * 32 + threadIdx.x]); //match
					xE = xE + sv;

					mmx = MMX[q];
					imx = IMX[q];
					dmx = DMX[q];

					MMX[q] = sv;
					DMX[q] = dcv;

					dcv = sv * __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]); //M_D

					sv = mmx * __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]); //M_I
					sv += imx * __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]); //I_I
					IMX[q] = sv;

					//if (threadIdx.x == 25 && i+j+z == 1)printf("threadIdx = %d # i = %d # MMX[%d] = %f \n", threadIdx.x, i+j,q, MMX[q]);
				}



				dcv = __shfl(dcv, threadIdx.x + 1);
				if (threadIdx.x == 0) dcv = 0;
				DMX[0] = 0.0f;
				for (q = 0; q < Q; q++)
				{
					DMX[q] += dcv;
					dcv = DMX[q] * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]); //D_D
				}

				//Serialization
				for (h = 0; h < 4; h++)
				{
					int cv = 0;

					dcv = __shfl(dcv, threadIdx.x + 1);
					if (threadIdx.x == 0) dcv = 0;
					for (q = 0; q < Q; q++)
					{
						sv = dcv + DMX[q];
						cv = cv | ((sv >= DMX[q]) ? 1 : 0);
						DMX[q] = sv;
						dcv = dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
					} 

					if (cv == 0) break;
				}


				for (q = 0; q < Q; q++) xE += DMX[q];

				//if (threadIdx.x < 32 && i+j+z == 1) printf("i = %d # threadIdx = %d ## xE = %f\n", i+j+z, threadIdx.x, xE);
				xE = xE + __shfl_down_sync(0x1F, xE, 16);
				//if (threadIdx.x < 16 && i+j+z == 1) printf("i = %d # threadIdx = %d ## xE = %f\n", i+j+z, threadIdx.x, xE);
				xE = xE + __shfl_down_sync(0x1F, xE, 8);
				//if (threadIdx.x < 8 && i+j+z == 1) printf("i = %d # threadIdx = %d ## xE = %f\n", i+j+z, threadIdx.x, xE);
				xE = xE + __shfl_down_sync(0x1F, xE, 4);
				//if (threadIdx.x < 4 && i+j+z == 1) printf("i = %d # threadIdx = %d ## xE = %f\n", i+j+z, threadIdx.x, xE);
				xE = xE + __shfl_down_sync(0x1F, xE, 2);
				//if (threadIdx.x < 2 && i+j+z == 1) printf("i = %d # threadIdx = %d ## xE = %f\n", i+j+z, threadIdx.x, xE);
				xE = xE + __shfl_down_sync(0x1F, xE, 1);

				if (threadIdx.x == 0)
				{
					xN = xN * NJC_LOOP;
					xC = xC * NJC_LOOP + xE * 0.5;
					xJ = xJ * NJC_LOOP + xE * 0.5;
					xBv[threadIdx.y] = xJ * NJC_MOVE + xN * NJC_MOVE;

					/*xC = xE * e_lm;
					xJ = xE * e_lm;
					xB = xJ * NJC_MOVE + xN * NJC_MOVE;*/

					xEv[threadIdx.y] = xE;
				}

				__syncthreads();

				//if (threadIdx.x == 0 && i+j+z<10)printf("threadIdx = %d # i = %d # xE = %f xN = %f xC = %f xJ = %f xBv = %f\n", threadIdx.x, i+j+z, xE, xN, xC, xJ, xBv);
				//if (threadIdx.x == 0 && i+j+z<10)printf("threadIdx = %d # i = %d # xBv = %f \n", threadIdx.x, i+j+z, xBv);

				if (xEv[threadIdx.y] > 1.0e4)
				{
					if (threadIdx.x == 0)
					{
						xN = xN / xE;
						xC = xC / xE;
						xJ = xJ / xE;
						xBv[threadIdx.y] = xBv[threadIdx.y] / xE;
					}

					__syncthreads();

					xE = 1.0f / xEv[threadIdx.y];
					for (q = 0; q < Q; q++)
					{
						MMX[q] = MMX[q] * xE;
						IMX[q] = IMX[q] * xE;
						DMX[q] = DMX[q] * xE;
					}

					scales[i] = xEv[threadIdx.y];
					totscale[threadIdx.y] += logf(xEv[threadIdx.y]);
					xEv[threadIdx.y] = 1.0f;
				}
				else scales[i] = 1.0;

				__syncthreads();
			}
		}
	}

	if (threadIdx.x == 0) sc[threadIdx.y] = totscale[threadIdx.y]+ logf(xC) + logf(NJC_MOVE);

	//end while
}

#endif /* _LOCALMEMORY_KERNEL_FWD_ */