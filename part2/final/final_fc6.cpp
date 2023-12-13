#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

void fc6(const float input[120][1][1], const float weights[10][120][1][1], 
         const float bias[10], float output[10]) {
    for(int n = 0; n < 10; n++) {
        output[n] = 0;
        for(int c = 0; c < 120; c++) {
            output[n] += weights[n][c][0][0] * input[c][0][0];
        }
        output[n]+=bias[n];
    }
}
