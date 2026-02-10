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



// Using the block prefetch logic
static inline void prefetch_weight_row(const void* start_ptr, int64_t num_blocks, uint32_t worker_id) {
    const uint64_t cache_line_size = 64;
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (num_blocks * sizeof(block_q8_0));

    // 1. Align to cache lines and calculate range
    uint64_t startCL = (self_ptr + 63) >> 6;
    uint64_t endCL = (self_ptr_end + 63) >> 6;
    if (endCL >= startCL) {
        uint64_t total_lines = endCL - startCL + 1;
        // 2. Load balance across the minions in the Shire (assuming 8 per group for this logic)
        // Adjust worker_id if using global_id
        uint32_t local_worker_id = worker_id % 8; 
        uint64_t lines_per_minion = total_lines >> 3;
        uint64_t extra = total_lines & 7;

        uint64_t offset = local_worker_id * lines_per_minion;
        if (local_worker_id < extra) {
            offset += local_worker_id;
            lines_per_minion++;
        } else {
            offset += extra;
        }
        self_ptr = (startCL + offset) << 6;
        int pending_lines = lines_per_minion;

        // 3. Hardware Prefetch Loop (16 lines at a time)
        for (; pending_lines > 0; pending_lines -= 16) {
            uint64_t current_batch = (pending_lines > 16 ? 16 : pending_lines) - 1;
            uint64_t self_size = current_batch; // bits 3:0

            __asm__ __volatile__ (
                "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
                "addi  x31, zero, 64\n"  // Stride = 64 bytes
                "or    x3, x1, %[ptr]\n"  // Combine Dest + VA
                "or    x3, x3, %[sz]\n"  // Combine with NumLines
                "csrw  0x81f, x3\n"  // prefetch_va
                : 
                : [ptr] "r" (self_ptr),
                  [sz] "r" (self_size)
                : "x1", "x3", "x31", "memory"
            );
            self_ptr += (16 * 64);
        }
    }
}

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    const int64_t K_blocks = K >> 5;

    const block_q8_0* src0_data = (const block_q8_0*)params->src0.data;
    const float* src1_data = (const float*)params->src1.data;
    float* dst_data = (float*)params->dst.data;

    // We distribute M rows across 1024 minions
    for (int64_t m = global_id; m < M; m += 1024) {
        
        // PREFETCH STEP 
        // Prefetch the entire weight row for 'm' into L2 once.
        // This row is reused for every single N column,
        prefetch_weight_row(src0_data + (m * K_blocks), K_blocks, global_id);

        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;
            const block_q8_0* q_row = src0_data + (m * K_blocks);
            const float* b_col = src1_data + (n * K);

            for (int64_t kb = 0; kb < K_blocks; kb++) {

                const float scale = fp16_to_fp32((q_row+kb)->d);
                float acc_vec[8]; 

                // Gather pattern for 8-element byte loading
                static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                
                unsigned long temp_mask;

                __asm__ volatile(
                    // 1. Setup Environment
                    "mova.x.m %[t_mask]\n"             // Save current mask
                    "mov.m.x m0, x0, 0xFF\n"           // Enable all 8 elements
                    "fsub.ps f10, f10, f10\n"          // Initialize accumulator f10 to 0.0
                    "flw.ps f31, %[gather]\n"          // Load gather pattern into f31 once

                    // 2. Pre-load B vector chunks (32 floats total into 4 registers)
                    "flw.ps f1, 0(%[b_ptr])\n"         // B[0-7]
                    "flw.ps f2, 32(%[b_ptr])\n"        // B[8-15]
                    "flw.ps f3, 64(%[b_ptr])\n"        // B[16-23]
                    "flw.ps f4, 96(%[b_ptr])\n"        // B[24-31]

                    // 3. Process Chunk 0 (0-7)
                    "fgb.ps f11, f31(%[a_ptr])\n"      // Gather 8 int8s
                    "fcvt.ps.pw f11, f11\n"            // Convert to float
                    "fmadd.ps f10, f11, f1, f10\n"     // acc += a_0 * b_0

                    // 4. Process Chunk 1 (8-15)
                    "addi t0, %[a_ptr], 8\n"           // Offset for 8 bytes
                    "fgb.ps f11, f31(t0)\n"
                    "fcvt.ps.pw f11, f11\n"
                    "fmadd.ps f10, f11, f2, f10\n"     // acc += a_1 * b_1

                    // 5. Process Chunk 2 (16-23)
                    "addi t0, %[a_ptr], 16\n"
                    "fgb.ps f11, f31(t0)\n"
                    "fcvt.ps.pw f11, f11\n"
                    "fmadd.ps f10, f11, f3, f10\n"     // acc += a_2 * b_2

                    // 6. Process Chunk 3 (24-31)
                    "addi t0, %[a_ptr], 24\n"
                    "fgb.ps f11, f31(t0)\n"
                    "fcvt.ps.pw f11, f11\n"
                    "fmadd.ps f10, f11, f4, f10\n"     // acc += a_3 * b_3


                    // "addi f11, f11, f12\n"
                    // "addi f11, f11, f13\n"
                    // "addi f11, f11, f14\n"


                    // 7. Store Result and Restore Mask
                    "fsw.ps f10, %[res_mem]\n"         // Store final 8-wide sum
                    "mova.m.x %[t_mask]\n"             // Restore mask

                    : [res_mem] "=m"(*(float(*)[8])acc_vec), [t_mask] "=&r"(temp_mask)
                    : [a_ptr] "r"((q_row+kb)->qs),
                    [b_ptr] "r"(b_col + (kb << 5)), // b_col + (kb * 32) = b_col + (kb << 5)
                    [gather] "m"(*(const int32_t(*)[8])gather_pattern)
                    : "f1", "f2", "f3", "f4", "f10", "f11", "f12", "f13", "f14", "f31", "t0", "memory"
                );

                // Horizontal sum: reduce 8 elements to scalar
                float final_sum = acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] + 
                                acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];

                sum += final_sum * scale;

            }

            atomic_store_f32((volatile float*)(dst_data + (n * M) + m), sum);
        }
    }
    return 0;
}