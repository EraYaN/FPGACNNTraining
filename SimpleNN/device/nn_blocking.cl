typedef float nn_t; //main type used

//#define DEBUG 1
#define BLOCK_SIZE 16
#define BLOCK_MEM (BLOCK_SIZE * BLOCK_SIZE)

#ifdef AOCL_BOARD_p510t_sch_ax115
#define FMA_LATENCY 5
#else
#define FMA_LATENCY 8
#endif
#define II_CYCLES FMA_LATENCY

void gemm_nn(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc);
void gemm_nt(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc);
void gemm_tn(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc);
void gemm_tt(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc);
void gemm(int TA, int TB, int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        nn_t BETA,
        __global nn_t *restrict C, int ldc);
        

__kernel void forward(global nn_t *restrict activations_in, global nn_t *restrict weights_in, global nn_t *restrict bias_in,
                              global nn_t *restrict activations_out, int minibatch_size, int rows_in, int rows_out, int enable_relu)
{
    const int N = minibatch_size;
    const int K = rows_in;
    const int M = rows_out;
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
                    // } else {
                    //     blockO[mi_block + j_block] = 0;
                    }
                }
            }

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                const int kmax = k0 + BLOCK_SIZE > K ? K - k0 : BLOCK_SIZE;
#ifdef DEBUG
                printf("Fetching blocks A(%d,%d) and W(%d,%d)\n", k0, i0, k0, j0);
#endif
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
                #pragma ivdep
                for (int i1 = 0; i1 < BLOCK_SIZE; i1++) {
                    
                    
                        int mi_block = BLOCK_SIZE * i1; // M * i1;                      
                        #pragma ivdep
                        for (int j1 = 0; j1 < BLOCK_SIZE; j1++) {                              
                            int sj_block = BLOCK_SIZE * j1; // M * j1;
                            int mij_block = mi_block + j1;

                            #pragma unroll
                            for (int k1 = 0; k1 < BLOCK_SIZE; ++k1) {
                                if (i1 < imax && j1 < jmax && k1 < kmax) {
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
            //barrier(CLK_GLOBAL_MEM_FENCE);
#ifdef DEBUG
            printf("Writing back block O(%d,%d)\n", j0, i0);
#endif
            for (int outer_count = 0; outer_count < BLOCK_SIZE; outer_count++) {
                int mi_block = BLOCK_SIZE * outer_count;
                int mi = M * (outer_count + i0);

#pragma unroll
                for (int j_block = 0; j_block < BLOCK_SIZE; j_block++) {
                    int j_global = j_block + j0;
                    if (j_block < jmax && outer_count < imax) {                        
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
                    //} else {
                    //    activations_out[mi + j0] = blockO[mi_block];
                    }
                }
            }
        }
    }
}

__kernel void forward_softmax(global nn_t *restrict activations_out, int minibatch_size, int rows_out) {
    for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_out * i;
        
        nn_t max = -INFINITY;
        // find the max
        for (int j = 0; j < rows_out; j++) {
            //printf("act old %d: %f\n",mi+j,activations_out[mi + j]);
            if(activations_out[mi + j]>max){
                //printf("Max: %f is larger then %f\n",activations_out[mi + j],max);
                max = activations_out[mi + j];
            } else {
                max = max;
            }
        }
        //printf("Max: %f\n",max);
        // sum the vector
        nn_t sum = 0;
        for (int j = 0; j < rows_out; j++) {
            nn_t temp = exp(activations_out[mi + j] - max);
            sum+=temp;
            activations_out[mi + j] = temp;
        }
        // divide the vector       
        sum = 1 / sum;
        for (int j = 0; j < rows_out; j++) {
            
            activations_out[mi + j] = activations_out[mi + j] * sum;
            //printf("act new %d: %f\n",mi+j,activations_out[mi + j]);
        }
        
    }
}

__kernel void backward_first_delta(global nn_t *restrict activations_out, global nn_t *restrict ground_truth, global nn_t *restrict delta, int minibatch_size, int rows_out){
    int N = rows_out * minibatch_size;
    nn_t reciprocal = 1 / (nn_t)minibatch_size;
    for (int i = 0; i < N; i++) {
        delta[i] = (activations_out[i] - ground_truth[i]) * reciprocal;
    }
}

__kernel void backward(global nn_t *restrict activations, global nn_t *restrict weights, global nn_t * restrict dW, global nn_t *restrict bias, global nn_t *restrict delta, global nn_t *restrict delta_next, nn_t learn_rate, int minibatch_size, int rows_in, int rows_out, int layer){
    
    //Process Bias
    for (int i = 0; i < minibatch_size; i++) {
        int mi = rows_out * i;
        for (int j = 0; j < rows_out; j++) {
            //printf("#%d prob: %f; gt: %f; deriv: %d\n",mi+j,activations_out[mi + j],ground_truth[mi + j],(activations_out[mi + j] < 0 ? 0 : 1));
            //delta[mi + j] = (activations_out[mi + j] - ground_truth[mi + j]) * (activations_out[mi + j] < 0 ? 0 : 1);
            //printf("bias %d dn %d: %f\n",j,mi+j, delta_next[mi+j]);
            bias[j] -= learn_rate * delta_next[mi+j];
        }
    }
    
    if(layer > 0){
        //Process delta
        gemm(0,0, minibatch_size, rows_in, rows_out, 1, 
            delta_next, rows_out, 
            weights, rows_in,
            0,
            delta, rows_in);

        //Apply relu derivative
        //printf("Delta (%d,%d): \n------\n",minibatch_size,rows_in);
        for (int i = 0; i < minibatch_size; i++) {
            int mi = rows_in * i;
            for (int j = 0; j < rows_in; j++) {
                //printf("%f\t",delta[mi+j]);
                if(activations[mi+j] <= 0)
                    delta[mi+j] = 0;
            }
            //printf("\n");
        }
    }
    //printf("------\n");

    //Process dW
    gemm(1,0, rows_out, rows_in, minibatch_size, 1, 
        delta_next, rows_out, 
        activations, rows_in,
        0,
        dW, rows_in);

    //Apply stuff to weights (delta = dW)
    for (int i = 0; i < rows_in; i++) {
        int mi = rows_out * i;
        for (int j = 0; j < rows_out; j++) {
#ifdef DEBUG
            printf("dW[%d,%d] = %f;\n",i,j,dW[mi+j]);
#endif
            //printf("weights[%d,%d] = %f;\n",i,j,weights[mi+j]);
            weights[mi+j] -= learn_rate * dW[mi+j]; //delta is dW here
        }
    }   

    

}

// void gemm_nn_sr(int M, int N, int K, nn_t ALPHA, 
//         __global nn_t *restrict A, int lda, 
//         __global nn_t *restrict B, int ldb,
//         __global nn_t *restrict C, int ldc)
// {
//     int i,j,k;
//     double shift_reg[II_CYCLES];
    
//     for(k = 0; k < K; ++k){
//         for(i = 0; i < M; ++i){
//             nn_t A_PART;
//             A_PART = ALPHA*A[i*lda+k];

//             #pragma unroll
//             for (int l = 0; l < II_CYCLES; l++)
//             {
//                 shift_reg[l] = 0;
//             }

//             //Iterate through every element of input array
//             for(int j = 0; j < K; ++k)
//             {
//                 //Load ith element into end of shift register
//                 //if N > II_CYCLE, add to shift_reg[0] to preserve values
//                 shift_reg[II_CYCLES-1] = shift_reg[0] + A_PART * B[k*ldb+j];
//                 #pragma unroll
//                 //Shift every element of shift register
//                 for(int j = 0; j < II_CYCLES; ++j)
//                 {
//                     shift_reg[j] = shift_reg[j + 1];
//                     //printf("shift_reg[%d] = %f\n",j, shift_reg[j]);
//                 }
//             }
//             float temp_sum = 0;

//             #pragma unroll
//             for(int l = 0; l < II_CYCLES; ++l)
//             {
//                 temp_sum += shift_reg[l];
//             }

//             //write back result
//             C[i*ldc+j] = temp_sum;
//         }
//     }
// }

void gemm_nn(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            nn_t A_PART;
            A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// void gemm_nt(int M, int N, int K, nn_t ALPHA, 
//         __global nn_t *restrict A, int lda, 
//         __global nn_t *restrict B, int ldb,
//         __global nn_t *restrict C, int ldc)
// {
//     int i,j,k;
//     for(i = 0; i < M; ++i){
//         for(j = 0; j < N; ++j){
//             nn_t sum;
//             sum = 0;
//             for(k = 0; k < K; ++k){
//                 sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
//             }
//             C[i*ldc+j] += sum;
//         }
//     }
// }

void gemm_tn(int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        __global nn_t *restrict C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            nn_t A_PART;
            A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// void gemm_tt(int M, int N, int K, nn_t ALPHA, 
//         __global nn_t *restrict A, int lda, 
//         __global nn_t *restrict B, int ldb,
//         __global nn_t *restrict C, int ldc)
// {
//     int i,j,k;
//     for(i = 0; i < M; ++i){
//         for(j = 0; j < N; ++j){
//             nn_t sum;
//             sum = 0;
//             for(k = 0; k < K; ++k){
//                 sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
//             }
//             C[i*ldc+j] += sum;
//         }
//     }
// }


void gemm(int TA, int TB, int M, int N, int K, nn_t ALPHA, 
        __global nn_t *restrict A, int lda, 
        __global nn_t *restrict B, int ldb,
        nn_t BETA,
        __global nn_t *restrict C, int ldc)
{
#ifdef DEBUG
    printf("gemm: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
#endif
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    //else if(!TA && TB)
       // gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    //else
        //gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}
