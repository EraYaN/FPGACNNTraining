#define SIZEDIV 28
#ifdef AOCL_BOARD_p510t_sch_ax115
#define FMA_LATENCY 5
#else
#define FMA_LATENCY 8
#endif
#define II_CYCLES SIZEDIV*FMA_LATENCY

__kernel void mul_kernel_relu(global float * restrict activations_in, global float * restrict weights_in, global float * restrict bias_in, global float * restrict activations_out, int minibatch_size, int rows_in, int rows_out) {

    //Create shift register with II_CYCLE+1 elements
    float shift_reg[II_CYCLES+1];

    //printf("Calling thread minibatch_size: %d, rows_in: %d rows_out: %d\n", minibatch_size, rows_in, rows_out);
    //Get the global id's
    //#pragma loop_coalesce 2
    for(int i = 0; i<minibatch_size;i++){
        for(int j = 0; j<rows_out; j++){

            //Set first value to bias.
            shift_reg[0] = bias_in[j];
            //Set rest to zero.
            #pragma unroll
            for (int i = 1; i < II_CYCLES + 1; i++)
            {
                shift_reg[i] = 0;
            }

            #pragma unroll SIZEDIV
            //Iterate through every element of input array
            for(int k = 0; k < rows_in; ++k)
            {
                //Load ith element into end of shift register
                //if N > II_CYCLE, add to shift_reg[0] to preserve values
                shift_reg[II_CYCLES] = shift_reg[0] + activations_in[i * rows_in + k] * weights_in[k * rows_out + j];
                #pragma unroll
                //Shift every element of shift register
                for(int j = 0; j < II_CYCLES; ++j)
                {
                    shift_reg[j] = shift_reg[j + 1];
                    //printf("shift_reg[%d] = %f\n",j, shift_reg[j]);
                }
            }
            float temp_sum = 0;

            #pragma unroll
            for(int l = 0; l < II_CYCLES; ++l)
            {
                temp_sum += shift_reg[l];
            }

            //write back result (Relu)
            activations_out[i * rows_out + j] = temp_sum > 0 ? temp_sum : 0;

        }
    }
}