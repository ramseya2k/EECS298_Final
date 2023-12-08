// Convolution Layer 3 with HLS pragmas
void convolution3_hls(float input[6][14][14], float weights[16][6][5][5], 
                       float bias[16], float output[16][10][10]) {
    #pragma omp parallel for collapse(3)
    for (int co = 0; co < 16; co++) {
        for (int h = 0; h < 10; h++) {
            for (int w = 0; w < 10; w++) {
                float sum = 0;

                // Unrolling the inner loop
                for (int m = 0; m < 5; m++) {
                    for (int n = 0; n < 5; n++) {
                        for (int ci = 0; ci < 6; ci++) {
                            sum += weights[co][ci][m][n] * input[ci][h + m][w + n];
                        }
                    }
                }

                output[co][h][w] = sum + bias[co];
            }
        }
    }
}

// Pragmas for HLS
#pragma omp declare target
#pragma omp declare simd
#pragma HLS interface ap_memory port=input
#pragma HLS interface ap_memory port=weights
#pragma HLS interface ap_memory port=bias
#pragma HLS interface ap_memory port=output
#pragma HLS array_partition variable=weights complete dim=1
#pragma HLS array_partition variable=weights complete dim=2
#pragma HLS array_partition variable=weights complete dim=3
#pragma HLS array_partition variable=input complete dim=1
#pragma HLS array_partition variable=output complete dim=1
#pragma omp end declare target
