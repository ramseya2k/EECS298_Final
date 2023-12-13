#include <hls_stream.h>

// Fully connected Layer 6
void fc6(const float input[120][1][1], const float weights[10][120][1][1], 
         const float bias[10], float output[10]) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS ARRAY_PARTITION variable=weights complete dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=16 dim=1

    float temp_output[10];
    #pragma HLS ARRAY_PARTITION variable=temp_output complete

    // Computation
    for (int n = 0; n < 10; n++) {
        #pragma HLS UNROLL factor=2
        temp_output[n] = 0;
        for (int c = 0; c < 120; c++) {
            #pragma HLS PIPELINE II=1
            temp_output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        temp_output[n] += bias[n];
    }

    // Write back the results
    for (int n = 0; n < 10; n++) {
        #pragma HLS UNROLL factor=2
        output[n] = temp_output[n];
    }
}
