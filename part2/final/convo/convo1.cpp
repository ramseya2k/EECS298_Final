#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void convolution1(float input[1][32][32], float weights[6][1][5][5], 
                  float bias[6], float output[6][28][28]) 
{
	#pragma HLS INTERFACE m_axi port=input offset=slave bundle=data0
    #pragma HLS INTERFACE s_axilite register port=input  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data1
    #pragma HLS INTERFACE s_axilite register port=weights  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data2
    #pragma HLS INTERFACE s_axilite register port=bias  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=data3
    #pragma HLS INTERFACE s_axilite register port=output  bundle=ctrl
    #pragma HLS INTERFACE s_axilite register port=return  bundle=ctrl
	
	
	
	for(int co = 0; co < 6; co++)
        for(int h = 0; h < 28; h++)
            for(int w = 0; w < 28; w++)
            {
                #pragma HLS PIPELINE II=1
				float sum = 0;
                for(int i = h, m = 0; i < (h + 5); i++, m++)
                {
                    #pragma HLS UNROLL
					for(int j = w, n = 0; j < (w + 5); j++, n++)
                        sum += weights[co][0][m][n] * input[0][i][j];
                }
                output[co][h][w] = sum + bias[co];
            }
}
