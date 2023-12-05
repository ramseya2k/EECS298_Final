// fc6_hls_optimized.cpp

#include <hls_stream.h>

const int IFM_DIM = 5;
const int OFM_DIM = 1;
const int IFM_CH = 16;
const int OFM_CH = 120;

typedef float data_t;

void fc6_hls_optimized(data_t input[IFM_CH][IFM_DIM][IFM_DIM],
                       data_t weights[OFM_CH][IFM_CH][IFM_DIM][IFM_DIM],
                       data_t bias[OFM_CH],
                       data_t output[OFM_CH]) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=input
#pragma HLS INTERFACE axis register both port=weights
#pragma HLS INTERFACE axis register both port=bias
#pragma HLS INTERFACE axis register both port=output

    // Loop unrolling and pipeline optimization
    for (int n = 0; n < OFM_CH; n++) {
#pragma HLS PIPELINE II=1
        data_t acc = bias[n];

        // Unrolling and pipelining the fully connected operation
        FCLoop:
        for (int c = 0; c < IFM_CH; c++) {
#pragma HLS UNROLL
            for (int h = 0; h < IFM_DIM; h++) {
#pragma HLS UNROLL
                for (int w = 0; w < IFM_DIM; w++) {
#pragma HLS UNROLL
                    acc += weights[n][c][h][w] * input[c][h][w];
                }
            }
        }

        output[n] = acc;
    }
}
