#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

// Convolution Layer 3
void convolution3(float input[6][14][14], float weights[16][6][5][5], 
                  float bias[16], float output[16][10][10],
                  bool enable) 
{
  #pragma HLS INTERFACE m_axi port=input offset=slave bundle=axi_port0
  #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=axi_port1
  #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=axi_port2
  #pragma HLS INTERFACE m_axi port=output offset=slave bundle=axi_port3
  #pragma HLS INTERFACE s_axilite port=enable register bundle=ctrl
  #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl  
  if (enable) {
    for (int co = 0; co < 16; co++)
      for (int h = 0; h < 10; h++)
        for (int w = 0; w < 10; w++) {
          float sum = 0;
          for (int i = h, m = 0; i < (h + 5); i++, m++)
            for (int j = w, n = 0; j < (w + 5); j++, n++)
              for (int ci = 0; ci < 6; ci++)
                sum += weights[co][ci][m][n] * input[ci][i][j];
          output[co][h][w] = sum + bias[co];
        }
  }
}

// Fully connected Layer 6
void fc6(const float input[120][1][1], const float weights[10][120][1][1], 
         const float bias[10], float output[10],
         bool enable) 
{
  #pragma HLS INTERFACE m_axi port=input offset=slave bundle=axi_port0
  #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=axi_port1
  #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=axi_port2
  #pragma HLS INTERFACE m_axi port=output offset=slave bundle=axi_port3
  #pragma HLS INTERFACE s_axilite port=enable register bundle=ctrl
  #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl
  
  if (enable) {
    for (int n = 0; n < 10; n++) {
      output[n] = 0;
      for (int c = 0; c < 120; c++) {
        output[n] += weights[n][c][0][0] * input[c][0][0];
      }
      output[n] += bias[n];
    }
  }
}

// Top-level function combining convolution3 and fc6
void my_top_function(float input[6][14][14], float weights_conv3[16][6][5][5],
                     float bias_conv3[16], float output_conv3[16][10][10],
                     float input_fc6[120][1][1], float weights_fc6[10][120][1][1],
                     float bias_fc6[10], float output_fc6[10],
                     bool enable_convolution3, bool enable_fc6) 
{
  convolution3(input, weights_conv3, bias_conv3, output_conv3, enable_convolution3);
  fc6(input_fc6, weights_fc6, bias_fc6, output_fc6, enable_fc6);
}
