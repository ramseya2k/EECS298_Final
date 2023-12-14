void convolution5(float input[16][5][5], float weights[120][16][5][5], 
                  float bias[120], float output[120][1][1]) {
    for(int co = 0; co < 120; co++) {
        float sum = 0;
        for(int i = 0, m = 0; i < 5; i++, m++) {
            for(int j = 0, n = 0; j < 5; j++, n++) {
                for (int ci = 0; ci < 16; ci++)
                    sum += weights[co][ci][m][n] * input[ci][i][j];
            }
        }
        output[co][0][0] = sum + bias[co];
    }
}
