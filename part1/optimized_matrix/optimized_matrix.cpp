#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void partTwoPartB(int A[16][16], int B[16][16], int AB[16][16])
{

	int M = 16;
	int N = 16;
	int P = 16;
	// local buffer array
	int A_buff[16][16];
	int B_buff[16][16];

	// assign the values of A & B into buffer variable
	for(int i=0; i < 16; ++i)
	{
		for(int j=0; j < 16; ++j)
		{
			A_buff[i][j] = A[i][j];
		}
	}

	for(int i=0; i < 16; ++i)
		{
			for(int j=0; j < 16; ++j)
			{
				B_buff[i][j] = A[i][j];
			}
		}


	#pragma HLS INTERFACE m_axi port=A offset=slave bundle=data0
	#pragma HLS INTERFACE s_axilite register port=A bundle=ctrl
	#pragma HLS INTERFACE m_axi port=B offset=slave bundle=data1
	#pragma HLS INTERFACE s_axilite register port=B bundle=ctrl
	#pragma HLS INTERFACE m_axi port=AB offset=slave bundle=data2
	#pragma HLS INTERFACE s_axilite register port=AB bundle=ctrl
	#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
	#pragma HLS ARRAY_PARTITION variable=A_buff complete dim=2
	#pragma HLS ARRAY_PARTITION variable=B_buff complete dim=1
	for(int i=0; i<N; ++i)
	{
		for(int j=0; j<P; ++j)
		{
			#pragma HLS PIPELINE II=1
			int ABij = 0;
			for(int k=0; k < M; ++k)
			{
				ABij += A_buff[i][k] * B_buff[k][j];
			}
			AB[i][j] = ABij;
		}
