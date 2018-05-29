
#ifndef _LOCALMEMORY_KERNEL_BWD_
#define _LOCALMEMORY_KERNEL_BWD_

extern "C" __global__ 
void KERNEL(unsigned int* seq, unsigned int total, unsigned int* offset,
double* sc, int* L,  unsigned int* L_6r, float* mat, float* tran, float* fscale,
int e_lm, int QV, double mu, double lambda)	
{
	volatile __shared__ unsigned int cache[RIB][32]; //Temp var for element of sequence
	float MMX[Q];
	float IMX[Q];
	float DMX[Q];
	//float bscale;

	const int seqIdx = gridDim.y * blockIdx.y + threadIdx.y;

	unsigned int LEN, OFF, res, res_s, res_p;

	float mmx, imx, dmx, dcv;
	float tmmx, timx, tdmx;
	float NJC_MOVE, NJC_LOOP, sv;

	float    xN, xE, xB, xC, xJ;
	volatile __shared__ float xEv[RIB], xBv[RIB], totscale[RIB];

	int i, q, j, z, h; //indexes



	//while (seqIdx + idx) < total)

	LEN = L_6r[seqIdx];
	OFF = offset[seqIdx];

	NJC_MOVE = (float)3.0f / (float)(L[seqIdx] + 3.0f);
	NJC_LOOP = 1.0f - NJC_MOVE;

	xJ = 0.0f;
	xBv[threadIdx.y] = 0.0f;
	xN = 0.0f;
	xC = NJC_MOVE;
	xEv[threadIdx.y] = xC * 0.5f; //e_lm must be 0.5
	xE = xEv[threadIdx.y];
	dcv = 0;

	for (q = 0; q < Q; q++) 
	{
		MMX[q] = xE;
		DMX[q] = xE;
		IMX[q] = 0;
	}
	//D_D
	dcv = __shfl_sync(0x1f, DMX[Q - 1], threadIdx.x - 1);
	if (threadIdx.x == 31) dcv = 0.0f;

	for (q = Q - 1; q > 0; q--)
	{
		DMX[q] += dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
		dcv = DMX[q];
	}
	dcv = dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
	DMX[q] = DMX[q] + dcv;

	for (j = 1; j < 4; j++)
	{
		dcv = __shfl_sync(0x1f, DMX[Q - 1], threadIdx.x - 1);
		if (threadIdx.x == 31) dcv = 0.0f;
		for (q = Q - 1; q >= 0; q--)
		{
			DMX[q] += dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
			dcv = DMX[q];
		}
	}

	//M_D
	dcv = __shfl_sync(0x1f, DMX[0], threadIdx.x - 1);
	if (threadIdx.x == 31) dcv = 0.0f;
	for (q = Q - 1; q >= 0; q--)
	{
		MMX[q] += dcv * __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]);
		dcv = DMX[q];
	}

	if (fscale[seqIdx] > 1.0)
	{
		xEv[threadIdx.y] = xEv[threadIdx.y] / fscale[seqIdx];
		xN = xN / fscale[seqIdx];
		xC = xC / fscale[seqIdx];
		xJ = xJ / fscale[seqIdx];
		xBv[threadIdx.y] = xBv[threadIdx.y] / fscale[seqIdx];
		xE = 1.0 / fscale[seqIdx];

		for (q = 0; q < Q; q++)
		{
			MMX[q] = MMX[q] * xE;
			DMX[q] = DMX[q] * xE;
			IMX[q] = IMX[q] * xE;
		}
	}
	//bscale[L[seqIdx] - 1] = fscale[seqIdx];
	totscale[seqIdx] = logf(fscale[seqIdx]);

	res_p = 31; //flag

	for (i = LEN - 33; i >= 0; i -= 32)
	{
		cache[threadIdx.y][threadIdx.x] = seq[OFF + i + threadIdx.x];

		for (j = 31; j <= 0; j--)
		{
			res = cache[threadIdx.y][j];	
			if ((res & 0x000000ff) == 31) break;

			for (z = 3; z <= 0; z--)
			{
				res_s = ((res >> (8 * z)) & 0x000000ff);
				if (res_s == 31) break;

				res_s *= Q * 32;

				if (z + j + i == 0) break;

				if (res_p == 31) {res_p = res_s; break;}

				tmmx = __ldg(&tran[1 * 32 + threadIdx.x]); // M_M
				tmmx = __shfl_sync(0x1f, tmmx, threadIdx.x - 1);
				if (threadIdx.x == 31) tmmx = 0.0f;

				timx = __ldg(&tran[2 * 32 + threadIdx.x]); //I_M
				timx = __shfl_sync(0x1f, timx, threadIdx.x - 1);
				if (threadIdx.x == 31) timx = 0.0f;

				tdmx = __ldg(&tran[3 * 32 + threadIdx.x]); //D_M
				tdmx = __shfl_sync(0x1f, tdmx, threadIdx.x - 1);
				if (threadIdx.x == 31) tdmx = 0.0f;

				mmx = MMX[0] * __ldg(&mat[res_p + threadIdx.x]);
				mmx = __shfl_sync(0x1f, mmx, threadIdx.x - 1);
				if (threadIdx.x == 31) mmx = 0.0f;

				xB = 0;

				for (q = Q - 1; q >= 0; q--)
				{
					imx = IMX[q];
					sv = mmx  * timx; //Shifted I_M
					sv += imx * __ldg(&tran[q * 224 + 6 * 32 + threadIdx.x]); //I_I
					IMX[q] = sv;

					DMX[q] = mmx * tdmx; //Shifted D_M

					sv = mmx * tmmx; //Shifted M_M
					sv += imx * __ldg(&tran[q * 224 + 5 * 32 + threadIdx.x]); //M_I

					mmx = MMX[q] * __ldg(&mat[res_p + q * 32 + threadIdx.x]); //match state
					MMX[q] = sv;

					tdmx = __ldg(&tran[q * 224 + 3 * 32 + threadIdx.x]); //D_M
					timx = __ldg(&tran[q * 224 + 2 * 32 + threadIdx.x]); //I_M
					tmmx = __ldg(&tran[q * 224 + 1 * 32 + threadIdx.x]); //M_M

					xB += mmx * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
				}

				xB = xB + __shfl_down_sync(0x1F, xB, 16);
				xB = xB + __shfl_down_sync(0x1F, xB, 8);
				xB = xB + __shfl_down_sync(0x1F, xB, 4);
				xB = xB + __shfl_down_sync(0x1F, xB, 2);
				xB = xB + __shfl_down_sync(0x1F, xB, 1);

				if (threadIdx.x == 0)
				{
					xC = xC * NJC_LOOP;
					xJ = xJ * NJC_LOOP + xB * NJC_MOVE;
					xN = xN * NJC_LOOP + xB * NJC_MOVE;
					xEv[threadIdx.y] = xC * 0.5f + xJ * 0.5f;

					xBv[threadIdx.y] = xB;
				}
				__syncthreads();
				xE = xEv[threadIdx.y];

				dcv = DMX[0] + xE;
				dcv = __shfl_sync(0x1f, dcv, threadIdx.x - 1);
				if (threadIdx.x == 31) dcv = 0.0f;
				for (q = Q - 1; q > 0; q--)
				{
					dcv = dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]) + xE;
					DMX[q] += dcv; 
					dcv = DMX[q];
					MMX[q] += xE;
				}
				dcv = dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
				dcv += xE;
				DMX[q] = dcv;
				MMX[q] += xE;

				for (j = 1; j < 4; j++)	
				{
				  	dcv = __shfl_sync(0x1f, dcv, threadIdx.x - 1);
					if (threadIdx.x == 31) dcv = 0.0f;
				  	for (q = Q-1; q >= 0; q--)
				    {
				      dcv        = dcv * __ldg(&tran[q * 224 + 7 * 32 + threadIdx.x]);
				      DMX[q] += dcv;
				    }
				}

				dcv = __shfl_sync(0x1f, DMX[0], threadIdx.x - 1);
				if (threadIdx.x == 31) dcv = 0.0f;
				for (q = Q - 1; q >= 0; q--)
				{
					MMX[q] += dcv * __ldg(&tran[q * 224 + 4 * 32 + threadIdx.x]);
					dcv = DMX[q];
				}

				if (xBv[threadIdx.y] > 1.0e4)
				{
					if (threadIdx.x == 0)
					{
						xEv[threadIdx.y] /= xBv[threadIdx.y];
						xN /= xBv[threadIdx.y];
						xJ /= xBv[threadIdx.y];
						xC /= xBv[threadIdx.y];
					}

					__syncthreads();

					xB = 1.0f / xBv[threadIdx.y];

					for (q = 0; q < Q; q++)
					{
						MMX[q] *= xB;
						IMX[q] *= xB;
						DMX[q] *= xB;
					}

					totscale[threadIdx.y] += logf(xBv[threadIdx.y]);
					xBv[threadIdx.y] = 1.0f;
				}

				res_p = res_s;
			}
		}

	}

	xB = 0;
	for (q = 0; q < Q; q++)
	{
		mmx = MMX[q] * __ldg(&mat[res_s + threadIdx.x]);
		xB += mmx * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]);
	}

	xB = xB + __shfl_down_sync(0x1F, xB, 16);
	xB = xB + __shfl_down_sync(0x1F, xB, 8);
	xB = xB + __shfl_down_sync(0x1F, xB, 4);
	xB = xB + __shfl_down_sync(0x1F, xB, 2);
	xB = xB + __shfl_down_sync(0x1F, xB, 1);

	if (threadIdx.x == 0) 
	{
		xN = xB * NJC_MOVE + xN * NJC_LOOP;

		sc[seqIdx] = totscale[threadIdx.y] + log(xN);
	}

	__syncthreads();

	//end while
}

#endif /* _LOCALMEMORY_KERNEL_BWD_ */