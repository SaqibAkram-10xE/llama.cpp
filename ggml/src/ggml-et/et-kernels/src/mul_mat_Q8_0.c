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

KERNEL_TRAMPOLINE();

// Main entry point for MUL_MAT Q8_0 x F32 kernel
int entry_point(struct ggml_et_binary_params* params, void* env) {
 
    uint64_t hart_id = get_hart_id();
    // Unique ID for every minion across the whole chip (0 to 1023)
    uint64_t minion_id = (hart_id >> 1) & 0x1F; 
    uint64_t shire_id = hart_id >> 6;
    uint64_t global_id = (shire_id << 5) + minion_id;

    // Dimensions
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    const int64_t K_blocks = K >> 5;

    // Data Pointers
    const block_q8_0* src0_data = (const block_q8_0*)params->src0.data;
    const float* src1_data = (const float*)params->src1.data;
    float* dst_data = (float*)params->dst.data;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = global_id; m < M; m += 1024) {
            float sum = 0.0f;
            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // Index: (Column n * Total K) + (Current Block * 32)
                const float* b_ptr = src1_data + (n * K) + (kb << 5);
                // Address for Weight Matrix A (Q8_0)
                // Index: (Row m * Total blocks in a row) + Current Block
                const block_q8_0* q_block = src0_data + (m * K_blocks) + kb;
                float acc_vec[8] __attribute__ ((aligned (32))) = {0.0f};
                //float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector
                static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                // Set mask register to enable all 8 vector elements
                unsigned long temp_mask;
                __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
                __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
                // Load here to avoid reloading every iteration
                __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");
                // Process 32 elements in 4 chunks of 8 elements each
                // becaause we have 8 element in a vector...
                for (int chunk = 0; chunk < 4; chunk++) {
                    int offset = chunk << 3; // chunk * 8
                    __asm__ volatile(
                        "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
                        "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
                        // "csrwi 0x7d3, 1\n"
                        "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
                        "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
                        "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
                        // "csrwi 0x7d3, 0\n"
                        "fsw.ps f10, %[result]\n"                // Store back to accumulator

                        : [result] "=m"(*(float(*)[8])acc_vec)
                        : [acc] "m"(*(const float(*)[8])acc_vec),
                        [a_ptr] "r"(&q_block->qs[offset]),
                        [b_vec] "m"(*(const float(*)[8])&b_ptr[offset]),
                        [scale] "m"(q_block->d)
                        : "f10", "f11", "f12"
                    );
                }
                // Restore original mask
                __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

                // Horizontal sum: reduce 8 accumulator elements to single scalar
                float final_sum;
                final_sum = acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
                    acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];
                // TODO: fcvt instruction 
                const float scale = fp16_to_fp32(q_block->d); 
                sum += final_sum * scale;
            }
            volatile float* c_element = (volatile float*)(dst_data + (n * M) + m);
            uint32_t value_bits = *(uint32_t*)&sum;
            __asm__ volatile(
                "amoswapg.w zero, %1, (%0)"
                :
                : "r"(c_element), "r"(value_bits)
                : "memory"
            );
        }
    }

    return 0;
}
