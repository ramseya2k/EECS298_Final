#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void convolution3(float input[6][14][14], float weights[16][6][5][5], 
                  float bias[16], float output[16][10][10]) {
    
	#pragma HLS INTERFACE m_axi port=input offset=slave bundle=data0
    #pragma HLS INTERFACE s_axilite register port=input  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data1
    #pragma HLS INTERFACE s_axilite register port=weights  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data2
    #pragma HLS INTERFACE s_axilite register port=bias  bundle=ctrl
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=data3
    #pragma HLS INTERFACE s_axilite register port=output  bundle=ctrl
    #pragma HLS INTERFACE s_axilite register port=return  bundle=ctrl
	
	#pragma HLS PIPELINE II=1
	
	for(int co = 0; co < 16; co++) {
        #pragma HLS PIPELINE II=1
        for(int h = 0; h < 10; h++) {
            for(int w = 0; w < 10; w++) {
                float sum = 0;
                #pragma HLS UNROLL factor=5
                for(int i = h, m = 0; i < (h + 5); i++, m++) {
                    #pragma HLS UNROLL factor=5
                    for(int j = w, n = 0; j < (w + 5); j++, n++) {
                        for (int ci = 0; ci < 6; ci++) {
                            sum += weights[co][ci][m][n] * input[ci][i][j];
                        }
                    }
                }
                output[co][h][w] = sum + bias[co];
            }
        }
    }
}
