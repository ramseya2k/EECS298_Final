#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"


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

  // Loop unrolling and pipeline
  LOOP_UNROLL: for(int n = 0; n < 10; n++) {
    #pragma HLS UNROLL factor=2
    float temp_output = 0;
    
    // Array partitioning for input
    PARTITION_INPUT: for(int c = 0; c < 120; c++) {
      #pragma HLS UNROLL
      float input_val = input[c][0][0];
      temp_output += weights[n][c][0][0] * input_val;
    }
    
    // Accumulate bias
    temp_output += bias[n];

    // Assign to output
    output[n] = temp_output;
  }
}
