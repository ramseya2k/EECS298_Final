#include "ap_int.h"
#include "hls_stream.h"

const int IFM_DIM = 5;
const int OFM_DIM = 1;
const int IFM_CH = 16;
const int OFM_CH = 120;

typedef float data_t;

void conv6_axi(data_t input[IFM_CH][IFM_DIM][IFM_DIM],
               data_t weights[OFM_CH][IFM_CH][IFM_DIM][IFM_DIM],
               data_t bias[OFM_CH],
               data_t output[OFM_CH]) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=data_input
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=data_weights
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=data_bias
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=data_output
#pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

    // Loop unrolling and pipeline optimization
    for (int n = 0; n < OFM_CH; n++) {
#pragma HLS PIPELINE II=1
        data_t acc = bias[n];

        // Unrolling and pipelining the convolution operation
        ConvLoop:
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
