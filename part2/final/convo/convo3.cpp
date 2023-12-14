void convolution3(float input[6][14][14], float weights[16][6][5][5], 
                  float bias[16], float output[16][10][10]) {
    for(int co = 0; co < 16; co++)
        for(int h = 0; h < 10; h++)
            for(int w = 0; w < 10; w++) {
                    float sum = 0;
                    for(int i = h, m = 0; i < (h+5); i++, m++) {
                        for(int j = w, n = 0; j < (w+5); j++, n++)
                            for (int ci = 0; ci < 6; ci++)
                                sum += weights[co][ci][m][n] * input[ci][i][j];
                    }
                    output[co][h][w] = sum + bias[co];
            }
}
