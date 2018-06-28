typedef float nn_t; //main type used

//#define DEBUG 1
#define BLOCK_SIZE 128
#define BLOCK_MEM (BLOCK_SIZE * BLOCK_SIZE)

#ifdef AOCL_BOARD_p510t_sch_ax115
#define FMA_LATENCY 5
#else
#define FMA_LATENCY 8
#endif
#define II_CYCLES FMA_LATENCY

__kernel void forward(global nn_t *restrict activations_in, global nn_t *restrict weights_in, global nn_t *restrict bias_in,
                              global nn_t *restrict activations_out, int minibatch_size, int rows_in, int rows_out, int enable_relu)
{
    int N = minibatch_size;
    int K = rows_in;
    int M = rows_out;
    // i1 = i
    // A = activations_in;
    // B = weights_in;
    // C = activations_out;
    //Create shift register with BLOCK_SIZE elements

    //printf("A[0] = %f; b[0] = %f; W[0] = %f\n",activations_in[0],bias_in[0],weights_in[0]);
    nn_t shift_reg[BLOCK_SIZE];

    //Define local block
    nn_t blockA[BLOCK_MEM];

    nn_t blockW[BLOCK_MEM];

    nn_t blockO[BLOCK_MEM];

    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        const int imax = i0 + BLOCK_SIZE > N ? N - i0 : BLOCK_SIZE;

        for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
            const int jmax = j0 + BLOCK_SIZE > M ? M - j0 : BLOCK_SIZE;

            for (int outer_count = 0; outer_count < BLOCK_SIZE; outer_count++) {
                int mi_block = BLOCK_SIZE * outer_count;

                #pragma unroll
                for (int j_block = 0; j_block < BLOCK_SIZE; j_block++) {
                    if (j_block < jmax && outer_count < imax) {
                        int j_global = j_block + j0;
#ifdef DEBUG
                        printf("Setting blockO[%d] = bias_in[%d] (%f)\n", mi_block + j_block, j_global, bias_in[j_global]);
#endif
                        blockO[mi_block + j_block] = bias_in[j_global];
                    }
                }
            }
            //barrier(CLK_GLOBAL_MEM_FENCE);

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                const int kmax = k0 + BLOCK_SIZE > K ? K - k0 : BLOCK_SIZE;
#ifdef DEBUG
                printf("Fetching blocks A(%d,%d) and W(%d,%d)\n", k0, i0, k0, j0);
#endif
//int i_fetch = 0;
//int j_fetch = 0;
                for (int outer_count = 0; outer_count < BLOCK_SIZE; outer_count++) {
                    int ki = K * (outer_count + i0);
                    int ki_block = BLOCK_SIZE * outer_count;
                    int sj = K * (outer_count + j0);
                    int sj_block = BLOCK_SIZE * outer_count;

#pragma unroll
                    for (int k_fetch = 0; k_fetch < BLOCK_SIZE; k_fetch++) {
                        int k_global = k_fetch + k0;
                        if (k_fetch < kmax && outer_count < imax) {
#ifdef DEBUG
                            printf("Loading blockA[%d] = activations_in[%d] (%f)\n", ki_block + k_fetch, ki + k_global,
                                                               activations_in[ki + k_global]);
#endif
                            blockA[ki_block + k_fetch] = activations_in[ki + k_global];
                        } else {
#ifdef DEBUG
                            printf("Zeroing blockA[%d] = 0\n", ki_block + k_fetch);
#endif
                            blockA[ki_block + k_fetch] = 0;
                        }
                        if (k_fetch < kmax && outer_count < jmax) {
#ifdef DEBUG
                            printf("Loading blockW[%d] = weights_in[%d] (%f)\n", sj_block + k_fetch, sj + k_global,
                                                               weights_in[sj + k_global]);
#endif
                            blockW[sj_block + k_fetch] = weights_in[sj + k_global];
                        } else {
#ifdef DEBUG
                            printf("Zeroing blockW[%d] = 0\n", sj_block + k_fetch);
#endif
                            blockW[sj_block + k_fetch] = 0;
                        }
                    }
                }
                //barrier(CLK_GLOBAL_MEM_FENCE);
//printf("blockA:\n");
//render_matrix(blockA,BLOCK_SIZE,BLOCK_SIZE);
//printf("blockW:\n");
//render_matrix(blockW,BLOCK_SIZE,BLOCK_SIZE);
#pragma ivdep
                for (int i1 = 0; i1 < BLOCK_SIZE; i1++) {
//printf("i1: %d\n",i1);
                    if (i1 < imax) {
                        int mi_block = BLOCK_SIZE * i1; // M * i1;
#pragma ivdep
                        for (int j1 = 0; j1 < BLOCK_SIZE; j1++) {
//printf("j1: %d\n",j1);
                            if (j1 < jmax) {
                                int sj_block = BLOCK_SIZE * j1; // M * j1;

                                int mij_block = mi_block + j1;
#pragma unroll
                                for (int k1 = 0; k1 < BLOCK_SIZE; ++k1) {
                                    if (k1 < kmax) {
//int k_new = k1 + k0;
//shift_reg[k1] = activations_in[mi + k_new] * weights_in[sj + k_new];
#ifdef DEBUG
                                        printf("shift_reg[%d] = blockA[%d] * blockW[%d]\n", k1, mi_block + k1,
                                                                                       sj_block + k1);
#endif
                                        shift_reg[k1] = blockA[mi_block + k1] * blockW[sj_block + k1];
                                    } else {
                                        shift_reg[k1] = 0;
                                    }
                                }

//Sum every element of shift register
                                nn_t temp_sum = 0;

#pragma unroll
                                for (int l = 0; l < BLOCK_SIZE; ++l) {
                                    temp_sum += shift_reg[l];
                                }
#ifdef DEBUG
                                printf("blockO[%d] += %f\n", mij_block, temp_sum);
#endif
//write back result
                                blockO[mij_block] += temp_sum;
                            }
                        }
                    }
                }
            }
            //barrier(CLK_GLOBAL_MEM_FENCE);
#ifdef DEBUG
            printf("Writing back block O(%d,%d)\n", j0, i0);
#endif
            for (int outer_count = 0; outer_count < BLOCK_SIZE; outer_count++) {
                int mi_block = BLOCK_SIZE * outer_count;
                int mi = M * (outer_count + i0);

#pragma unroll
                for (int j_block = 0; j_block < BLOCK_SIZE; j_block++) {

                    if (j_block < jmax && outer_count < imax) {
                        int j_global = j_block + j0;
#ifdef DEBUG
                        if (enable_relu == 1) {
                            printf("Writing back relu activations_out[%d] = blockO[%d] (%f)\n", mi + j_global,
                                   mi_block + j_block, blockO[mi_block + j_block] > 0 ? blockO[mi_block + j_block] : 0);
                        } else {
                            printf("Writing back plain activations_out[%d] = blockO[%d] (%f)\n", mi + j_global,
                                   mi_block + j_block, blockO[mi_block + j_block]);
                        }
#endif
                        if (enable_relu == 1) {
                            activations_out[mi + j_global] =
                                    blockO[mi_block + j_block] > 0 ? blockO[mi_block + j_block] : 0;
                        } else {
                            activations_out[mi + j_global] = blockO[mi_block + j_block];
                        }
                    }
                }
            }
        }
    }
}

__kernel void forward_softmax(global nn_t *restrict activations_out, int minibatch_size, int rows_out) {
    for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_out * i;
        float max = -INFINITY;
        // find the max
        for (int j = 0; j < rows_out; j++) {
            if(activations_out[mi + j]>max){
                //printf("Max: %f is larger then %f\n",activations_out[mi + j],max);
                max = activations_out[mi + j];
            } else {
                //printf("Max: %f is NOT larger then %f\n",activations_out[mi + j],max);
            }
        }
        //printf("Max: %f\n",max);
        // sum the vector
        float sum = 0;
        for (int j = 0; j < rows_out; j++) {
            float temp = exp(activations_out[mi + j] - max);
            sum+=temp;
            activations_out[mi + j] = temp;
        }
        // divide the vector
        if(sum != 0) {
            sum = 1 / sum;
            for (int j = 0; j < rows_out; j++) {
                activations_out[mi + j] = activations_out[mi + j] * sum;
            }
        } else {
            printf("Sum was zero.\n");
        }
    }
}

__kernel void backward_first_delta(global nn_t *restrict activations_out, global nn_t *restrict ground_truth, global nn_t *restrict delta, int minibatch_size, int rows_out){
    for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_out * i;
        for (int j = 0; j < rows_out; j++) {
            delta[mi + j] = activations_out[mi+j] - ground_truth[mi+j];
        }
    }
}

__kernel void backward(global nn_t *restrict activations_prev, global nn_t *restrict weights, global nn_t *restrict bias, global nn_t *restrict delta_next, nn_t learn_rate, nn_t regulation_strength, int minibatch_size, int rows_in, int rows_out){
    //nn_t dW; //Transposed result, because weights is transposed.
    //nn_t db;
    /*nn_t activations_prev_sq[minibatch_size*rows_in]; //rows_in is the length of the previous layer

    for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_in * i;
        for (int j = 0; j < rows_in; j++) {
            activations_prev_sq[mi + j] = 1 - activations_prev[mi+j]*activations_prev[mi+j];
        }
    }*/

    /*for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_out * i;
        for (int j = 0; j < rows_out; j++) {
            delta[mi + j] = activations_out[mi+j] - ground_truth[mi+j];
        }
    }*/
}