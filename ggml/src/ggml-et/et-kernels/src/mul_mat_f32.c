//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// #include "etsoc/isa.h"


KERNEL_TRAMPOLINE();


int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);
    
    // Matrix dimensions
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    
    // Block size is 8: K_blocks = K / 8
    const int64_t K_blocks = K >> 4; 

    // Data pointers
    const float* src0_data = (const float*)params->src0.data;
    const float* src1_data = (const float*)params->src1.data;
    float* dst_data       = (float*)params->dst.data;

    // Parallelize over M (rows)
    for (int64_t m = global_id; m < M; m += 1024) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            
            // Direct row/column pointers
            const float* q_row = src0_data + (m * K);
            const float* b_col = src1_data + (n * K);

            // Process blocks of 16
            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // (kb << 4) moves 16 float elements forward
                sum += compute_block_dot_product_f32(q_row + (kb << 4), b_col + (kb << 4));
            }
            
            // Atomic store to the destination
            atomic_store_f32((volatile float*)(dst_data + (n * M) + m), sum);
        }
    }
    return 0;
}



