// convolution3_hls_optimized.cpp

#include <hls_stream.h>

const int IFM_DIM = 14;
const int OFM_DIM = 10;
const int IFM_CH = 6;
const int OFM_CH = 16;
const int KERNEL_SIZE = 5;

typedef float data_t;

void convolution3_hls_optimized(data_t input[IFM_CH][IFM_DIM][IFM_DIM],
                                 data_t weights[OFM_CH][IFM_CH][KERNEL_SIZE][KERNEL_SIZE],
                                 data_t bias[OFM_CH],
                                 data_t output[OFM_CH][OFM_DIM][OFM_DIM]) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=input
#pragma HLS INTERFACE axis register both port=weights
#pragma HLS INTERFACE axis register both port=bias
#pragma HLS INTERFACE axis register both port=output

    data_t line_buffer[KERNEL_SIZE - 1][IFM_DIM][(IFM_DIM + KERNEL_SIZE - 1)];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    // Loop unrolling and pipeline optimization
    for (int co = 0; co < OFM_CH; co++) {
        for (int h = 0; h < OFM_DIM; h++) {
#pragma HLS PIPELINE II=1
            for (int w = 0; w < OFM_DIM; w++) {
                data_t acc = 0;

                // Unrolling and pipelining the convolution operation
                ConvolutionLoop:
                for (int ci = 0; ci < IFM_CH; ci++) {
#pragma HLS UNROLL
                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
#pragma HLS UNROLL
                        for (int kw = 0; kw < KERNEL_SIZE; kw++) {
#pragma HLS UNROLL
                            acc += weights[co][ci][kh][kw] *
                                   input[ci][h + kh][w + kw];
                        }
                    }
                }
                output[co][h][w] = acc + bias[co];
            }
        }
    }
}
