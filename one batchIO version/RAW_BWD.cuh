
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

	int seqIdx = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int LEN, OFF, res, res_s, res_p;

	float mmx, imx, dmx, dcv;
	float tmmx, timx, tdmx;
	float NJC_MOVE, NJC_LOOP, sv;

	float    xN, xEv, xBv, xC, xJ;
	volatile __shared__ float xE[RIB], xB[RIB], totscale[RIB];

	int i, q, j, z, h; //indexes


	while (seqIdx < total)
	{
		//if (threadIdx.x == 0) printf("seqIdx: %d\n", seqIdx);

		//if (threadIdx.x == 0) printf("%d ## init for new seq...\n", seqIdx);

		LEN = L_6r[seqIdx];
		OFF = offset[seqIdx];

		NJC_MOVE = (float)3.0f / (float)(L[seqIdx] + 3.0f);
		NJC_LOOP = 1.0f - NJC_MOVE;

		xJ = 0.0f;
		xB[threadIdx.y] = 0.0f;
		xN = 0.0f;
		xC = NJC_MOVE;
		xE[threadIdx.y] = xC * 0.5f; //e_lm must be 0.5
		xEv = xE[threadIdx.y];
		dcv = 0;

		for (q = 0; q < Q; q++) 
		{
			MMX[q] = xEv;
			DMX[q] = xEv;
			IMX[q] = 0;
		}

		//if (threadIdx.x == 0) printf("%d ## init done...\n", seqIdx);

		//D_D
		dcv = __shfl_sync(0x1f, DMX[Q - 1], threadIdx.x - 1);
		if (threadIdx.x == 31) dcv = 0.0f;

		for (q = Q - 1; q > 0; q--)
		{
			DMX[q] += dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
			dcv = DMX[q];
		}
		dcv = dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
		DMX[q] = DMX[q] + dcv;

		for (j = 1; j < 4; j++)
		{
			dcv = __shfl_sync(0x1f, DMX[Q - 1], threadIdx.x - 1);
			if (threadIdx.x == 31) dcv = 0.0f;
			for (q = Q - 1; q >= 0; q--)
			{
				DMX[q] += dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
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
			xE[threadIdx.y] = xE[threadIdx.y] / fscale[seqIdx];
			xN = xN / fscale[seqIdx];
			xC = xC / fscale[seqIdx];
			xJ = xJ / fscale[seqIdx];
			xB[threadIdx.y] = xB[threadIdx.y] / fscale[seqIdx];
			xEv = 1.0 / fscale[seqIdx];

			for (q = 0; q < Q; q++)
			{
				MMX[q] = MMX[q] * xEv;
				DMX[q] = DMX[q] * xEv;
				IMX[q] = IMX[q] * xEv;
			}
		}
		//bscale[L[seqIdx] - 1] = fscale[seqIdx];
		totscale[threadIdx.y] = logf(fscale[seqIdx]);

		res_p = 31; //flag

		//if (threadIdx.x == 0) printf("%d ## Main recoursion.. ## %d \n", seqIdx, LEN);

		for (i = (int) LEN - 32; i >= 0; i -= 32)
		{
			cache[threadIdx.y][threadIdx.x] = seq[OFF + i + threadIdx.x];
			//if (threadIdx.x == 0) printf("i=%d\n", i);

			for (j = 31; j >= 0; j--)
			{	
				//if (threadIdx.x == 0) printf("i = %d, j=%d\n", i,  j);
				res = cache[threadIdx.y][j];	
				if ((res & 0x000000ff) == 31) break;

				for (z = 3; z >= 0; z--)
				{
					res_s = ((res >> (8 * z)) & 0x000000ff);
					if (res_s == 31) break;

					res_s *= Q * 32;

					//if (threadIdx.x == 0) printf("i = %d, j = %d, z = %d \n", i, j, z);
					if (z + j + i == 0){ res_p == 31;  break;}

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

					xBv = 0;

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

						xBv += mmx * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]); //B_M
					}

					xBv = xBv + __shfl_down_sync(0x1F, xBv, 16);
					xBv = xBv + __shfl_down_sync(0x1F, xBv, 8);
					xBv = xBv + __shfl_down_sync(0x1F, xBv, 4);
					xBv = xBv + __shfl_down_sync(0x1F, xBv, 2);
					xBv = xBv + __shfl_down_sync(0x1F, xBv, 1);

					if (threadIdx.x == 0)
					{
						xC = xC * NJC_LOOP;
						xJ = xJ * NJC_LOOP + xBv * NJC_MOVE;
						xN = xN * NJC_LOOP + xBv * NJC_MOVE;
						xE[threadIdx.y] = xC * 0.5f + xJ * 0.5f;

						xB[threadIdx.y] = xBv;
					}
					__syncthreads();
					xEv = xE[threadIdx.y];

					dcv = DMX[0] + xEv;
					dcv = __shfl_sync(0x1f, dcv, threadIdx.x - 1);
					if (threadIdx.x == 31) dcv = 0.0f;
					for (q = Q - 1; q > 1; q--)
					{
						dcv = dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]) + xEv;
						DMX[q] += dcv; 
						dcv = DMX[q];
						MMX[q] += xEv;
					}
					dcv = dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
					dcv += xEv;
					DMX[q] = dcv;
					MMX[q] += xEv;

					for (h = 1; h < 4; h++)	
					{
					  	dcv = __shfl_sync(0x1f, dcv, threadIdx.x - 1);
						if (threadIdx.x == 31) dcv = 0.0f;
					  	for (q = Q-1; q >= 0; q--)
					    {
					      dcv        = dcv * __ldg(&tran[Q * 224 + q * 32 + threadIdx.x]);
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

					if (xB[threadIdx.y] > 1.0e4)
					{
						if (threadIdx.x == 0)
						{
							xE[threadIdx.y] /= xB[threadIdx.y];
							xN /= xB[threadIdx.y];
							xJ /= xB[threadIdx.y];
							xC /= xB[threadIdx.y];
						}

						__syncthreads();

						xBv = 1.0f / xB[threadIdx.y];

						for (q = 0; q < Q; q++)
						{
							MMX[q] *= xBv;
							IMX[q] *= xBv;
							DMX[q] *= xBv;
						}

						totscale[threadIdx.y] += logf(xB[threadIdx.y]);
						xB[threadIdx.y] = 1.0f;
					}

					res_p = res_s;

					
				}
				if (z + j + i == 0) break;
			}

			//if (threadIdx.x == 0) printf("z+j+i = %d\n", z + j + i);
			if (z + j + i == 0) break;
		}

		//if (threadIdx.x == 0) printf("%d ## Main recoursion done...\n", seqIdx);

		xBv = 0;
		for (q = 0; q < Q; q++)
		{
			mmx = MMX[q] * __ldg(&mat[res_s + q * 32 + threadIdx.x]);
			xBv += mmx * __ldg(&tran[q * 224 + 0 * 32 + threadIdx.x]);
		}

		xBv = xBv + __shfl_down_sync(0x1F, xBv, 16);
		xBv = xBv + __shfl_down_sync(0x1F, xBv, 8);
		xBv = xBv + __shfl_down_sync(0x1F, xBv, 4);
		xBv = xBv + __shfl_down_sync(0x1F, xBv, 2);
		xBv = xBv + __shfl_down_sync(0x1F, xBv, 1);

		if (threadIdx.x == 0) 
		{
			xN = xBv * NJC_MOVE + xN * NJC_LOOP;

			sc[seqIdx] = totscale[threadIdx.y] + log(xN);
		}

		__syncthreads();

		seqIdx += blockDim.y * gridDim.y;
	}
}

#endif /* _LOCALMEMORY_KERNEL_BWD_ */
