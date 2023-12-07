#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void pl_lenet5_fpga(float image[32][32],
                    float conv3_weights[16][6][5][5], float conv3_bias[16], float conv3_output[16][10][10],
                    float fc6_weights[10][120], float fc6_bias[10], float fc6_output[10])
{
    // Define AXI4 Master interfaces for input and output arrays
    #pragma HLS INTERFACE m_axi port=image offset=slave bundle=data0
    #pragma HLS INTERFACE s_axilite register port=image bundle=ctrl

    #pragma HLS INTERFACE m_axi port=conv3_weights offset=slave bundle=data1
    #pragma HLS INTERFACE s_axilite register port=conv3_weights bundle=ctrl

    #pragma HLS INTERFACE m_axi port=conv3_bias offset=slave bundle=data2
    #pragma HLS INTERFACE s_axilite register port=conv3_bias bundle=ctrl

    #pragma HLS INTERFACE m_axi port=conv3_output offset=slave bundle=data3
    #pragma HLS INTERFACE s_axilite register port=conv3_output bundle=ctrl

    #pragma HLS INTERFACE m_axi port=fc6_weights offset=slave bundle=data4
    #pragma HLS INTERFACE s_axilite register port=fc6_weights bundle=ctrl

    #pragma HLS INTERFACE m_axi port=fc6_bias offset=slave bundle=data5
    #pragma HLS INTERFACE s_axilite register port=fc6_bias bundle=ctrl

    #pragma HLS INTERFACE m_axi port=fc6_output offset=slave bundle=data6
    #pragma HLS INTERFACE s_axilite register port=fc6_output bundle=ctrl

    // Define a control interface for the function
    #pragma HLS INTERFACE s_axilite register port=return bundle=ctrl

    // Convolution Layer 3
    Convolution3(image, conv3_weights, conv3_bias, conv3_output);

    // Fully Connected Layer 6
    FC6(conv3_output, fc6_weights, fc6_bias, fc6_output);
}

// Convolution Layer 3 operation
void Convolution3(float input[16][10][10], float weights[16][6][5][5], float bias[16], float output[16][10][10])
{
    for (int co = 0; co < 16; co++) {
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {
                // Placeholder for convolution logic
                // You need to add your convolution logic here
                // The following is a simple example; replace it with your implementation
                float conv_sum = bias[co];
                for (int ci = 0; ci < 6; ci++) {
                    for (int i = 0; i < 5; i++) {
                        for (int j = 0; j < 5; j++) {
                            conv_sum += weights[co][ci][i][j] * input[ci][h + i][w + j];
                        }
                    }
                }
                output[co][h][w] = hls::relu(conv_sum); // Apply ReLU activation
            }
        }
    }
}

// Fully Connected Layer 6 operation
void FC6(float input[16][10][10], float weights[10][120], float bias[10], float output[10])
{
    float fc6_input[120];

    // Flatten the input tensor
    for (int i = 0; i < 120; i++) {
        fc6_input[i] = input[i / 10][i % 10 / 10][i % 10 % 10];
    }

    // Matrix-vector multiplication for fully connected layer
    for (int co = 0; co < 10; co++) {
        float fc_sum = bias[co];
        for (int ci = 0; ci < 120; ci++) {
            fc_sum += weights[co][ci] * fc6_input[ci];
        }
        output[co] = hls::relu(fc_sum); // Apply ReLU activation
    }
}
