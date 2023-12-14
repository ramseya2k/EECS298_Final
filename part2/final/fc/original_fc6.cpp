#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"



// Fully connected Layer 6
void fc6(const float input[120][1][1], const float weights[10][120][1][1], 
         const float bias[10], float output[10]) {
 
  #pragma HLS INTERFACE m_axi port=input offset=slave bundle=data0
  #pragma HLS INTERFACE s_axilite register port=input bundle=ctrl
  #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data1
  #pragma HLS INTERFACE s_axilite register port=weights bundle=ctrl
  #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data2
  #pragma HLS INTERFACE s_axilite register port=bias bundle=ctrl
  #pragma HLS INTERFACE m_axi port=output offset=slave bundle=data3
  #pragma HLS INTERFACE s_axilite register port=output bundle=ctrl   
  #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
 
    for(int n = 0; n < 10; n++) {
        output[n] = 0;
        for(int c = 0; c < 120; c++) {
            output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        output[n]+=bias[n];