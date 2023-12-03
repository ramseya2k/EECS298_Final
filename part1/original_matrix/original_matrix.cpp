#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void partTwoPartA(int A[16][16], int B[16][16], int AB[16][16])
{

	int M = 16;
	int N = 16;
	int P = 16;
	#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
	#pragma HLS INTERFACE s_axilite register port=A bundle=ctrl
	#pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
	#pragma HLS INTERFACE s_axilite register port=B bundle=ctrl
	#pragma HLS INTERFACE m_axi port=AB offset=slave bundle=data2
	#pragma HLS INTERFACE s_axilite register port=AB bundle=ctrl
	#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
	for(int i=0; i<N; ++i)
	{
		for(int j=0; j<P; ++j)
		{
			int ABij = 0;
			for(int k=0; k < M; ++k)
			{
				ABij += A[i][k] * B[k][j];
			}
			AB[i][j] = ABij;
		}
	}
 }
