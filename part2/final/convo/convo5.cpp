#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void convolution5(float input[16][5][5], float weights[120][16][5][5], 
                  float bias[120], float output[120][1][1]) {
    
  #pragma HLS INTERFACE m_axi port=input offset=slave bundle=data0
  #pragma HLS INTERFACE s_axilite register port=input bundle=ctrl
  #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data1
  #pragma HLS INTERFACE s_axilite register port=weights bundle=ctrl
  #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data2
  #pragma HLS INTERFACE s_axilite register port=bias bundle=ctrl
  #pragma HLS INTERFACE m_axi port=output offset=slave bundle=data3
  #pragma HLS INTERFACE s_axilite register port=output bundle=ctrl
  #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

	for (int co = 0; co < 120; co++) {
	        float sum = bias[co];

	        for (int ci = 0; ci < 16; ci++) {
				#pragma HLS PIPELINE II = 1
	        	for (int i = 0; i < 5; i++) {
					#pragma HLS unroll factor=5
	        		for (int j = 0; j < 5; j++) {
	                    sum += weights[co][ci][i][j] * input[ci][i][j];
	                }
	            }
	        }

	        output[co][0][0] = sum;
	    }

}
