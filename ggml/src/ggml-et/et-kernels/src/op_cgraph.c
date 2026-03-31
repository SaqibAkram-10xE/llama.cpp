//******************************************************************************
// CGraph Kernel
// Compute graph execution
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "quants.h"
#include "math_fp.h"
#include "block_ops.h"

struct ggml_tensor_et {
    int64_t ne[4];      // dimensions
    uint64_t nb[4];     // strides (fixed-width for ABI compatibility)
    enum ggml_type type;
    uint64_t data;      // Device pointer (fixed-width for ABI compatibility)
};

static inline uint64_t tensor_bytes(const struct ggml_tensor_et * t) {
    return (uint64_t)t->ne[3] * (uint64_t)t->nb[3];
}

struct ggml_node_meta_et {
    struct ggml_tensor_et src0;
    struct ggml_tensor_et src1;
    struct ggml_tensor_et src2;
    struct ggml_tensor_et dst;
    int32_t op_params[16];
};

struct ggml_cgraph_et {
    int size;
    int n_nodes;
    int n_leafs;
    struct ggml_tensor ** nodes;

    uint8_t data[];   // flexible array at end
};

// Helper function to convert ggml_tensor_et to ggml_tensor
static inline void convert_to_ggml_tensor(struct ggml_tensor * dst, struct ggml_tensor_et * src, enum ggml_op op) {
    dst->type = src->type;
    dst->data = (void*)(uintptr_t)src->data;  // Cast uint64_t back to pointer
    dst->op   = op;
    for(int j = 0; j < 4; j++){
        dst->ne[j] = src->ne[j];
        dst->nb[j] = (size_t)src->nb[j];  // Cast uint64_t back to size_t
    }
}


// // FCC consume - blocks until a credit is available on the specified FCC register
// inline __attribute__((always_inline)) void fcc_consume(uint64_t fcc_reg)
// {
//     __asm__ __volatile__("csrw fcc, %0\n" : : "r"(fcc_reg));
// }

// // ESR address construction for FCC credit increment registers
// #define ESR_SHIRE_REGION       0x0100340000ULL
// #define ESR_REGION_SHIRE_SHIFT 22
// #define ESR_SHIRE(shire, name) \
//     (ESR_SHIRE_REGION | ((uint64_t)(shire) << ESR_REGION_SHIRE_SHIFT) | (uint64_t)(ESR_##name))
// #define ESR_FCC_CREDINC_0 0xC0

// // Send FCC credit to specified shire/thread/register targeting specific minions
// inline __attribute__((always_inline))
// void fcc_send(uint32_t shire, uint32_t thread, uint32_t fcc_reg, uint64_t hart_mask)
// {
//     volatile uint64_t* fcc_credinc_addr =
//         (uint64_t*)ESR_SHIRE(shire, FCC_CREDINC_0) +
//         ((thread << 1) | fcc_reg);
//     *fcc_credinc_addr = hart_mask;
// }

// // Fast Local Barrier - per-shire hardware barrier
// // Returns 1 if this hart was the last to arrive, 0 otherwise
// inline __attribute__((always_inline))
// uint64_t flbarrier(uint64_t barrier_num, uint64_t match)
// {
//     uint64_t ret;
//     uint64_t flb_arg = (match << 5) | (barrier_num & 0x1F);
//     __asm__ __volatile__("csrrw %0, 0x820, %1" : "=r"(ret) : "r"(flb_arg));
//     return ret;
// }





// // Evict whole shire L1+L2 via firmware syscall
// static inline void __attribute__((always_inline)) flush_shire_l1_l2(void) {
//     register uint64_t a0 __asm__("a0") = 11; // SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2 11
//     register uint64_t a1 __asm__("a1") = 0;
//     register uint64_t a2 __asm__("a2") = 0;
//     register uint64_t a3 __asm__("a3") = 0;
//     __asm__ __volatile__("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3) : "memory");
// }



// ========================================================================
// Entry point — graph execution loop (Updated)
// ========================================================================
int entry_point(struct ggml_cgraph_et* cg, void* env) {
    // Reconstruct pointers on device side
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *) cg->data;
    uint8_t * node_op = (uint8_t *) (node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    device_barrier(32);


    /*for (int i = 0; i < n_nodes; i++) {
        const int op = node_op[i];
        
        if (op == GGML_OP_NONE) {
            continue;
        }

        // Checks if current node is RMS_NORM and next is MUL to fuse them
        if (ggml_et_can_fuse(cg, i, (enum ggml_op[]){ GGML_OP_RMS_NORM, GGML_OP_MUL }, 2)) {
            ggml_et_op_rms_norm_mul(env, &node_meta[i], &node_meta[i + 1]);
            i++; // Skip the next node as it's now fused
            device_barrier(32);
            continue;
        }

        switch (op) {
            case GGML_OP_SQR:           ggml_et_op_sqr(env, &node_meta[i]); break;
            case GGML_OP_UNARY:         ggml_et_op_unary(env, &node_meta[i]); break;
            case GGML_OP_SUM_ROWS:      ggml_et_op_sum_rows(env, &node_meta[i]); break;
            case GGML_OP_MUL:           ggml_et_op_mul(env, &node_meta[i]); break;
            case GGML_OP_ADD:           ggml_et_op_add(env, &node_meta[i]); break;
            case GGML_OP_SUB:           ggml_et_op_sub(env, &node_meta[i]); break;
            case GGML_OP_CUMSUM:        ggml_et_op_cumsum(env, &node_meta[i]); break;
            case GGML_OP_MUL_MAT:       ggml_et_op_mul_mat(env, &node_meta[i]); break;
            case GGML_OP_MUL_MAT_ID:    ggml_et_op_mul_mat_id(env, &node_meta[i]); break;
            case GGML_OP_ROPE:          ggml_et_op_rope(env, &node_meta[i]); break;
            case GGML_OP_RMS_NORM:      ggml_et_op_rms_norm(env, &node_meta[i]); break;
            case GGML_OP_NORM:          ggml_et_op_norm(env, &node_meta[i]); break;
            case GGML_OP_L2_NORM:       ggml_et_op_l2_norm(env, &node_meta[i]); break;
            case GGML_OP_SCALE:         ggml_et_op_scale(env, &node_meta[i]); break;
            case GGML_OP_GLU:           ggml_et_op_glu(env, &node_meta[i]); break;
            case GGML_OP_SOFT_MAX:      ggml_et_op_softmax(env, &node_meta[i]); break;
            case GGML_OP_FLASH_ATTN_EXT: ggml_et_op_flash_attn_ext(env, &node_meta[i]); break;
            case GGML_OP_GET_ROWS:      ggml_et_op_get_rows(env, &node_meta[i]); break;
            case GGML_OP_CONT:          ggml_et_op_cont(env, &node_meta[i]); break;
            case GGML_OP_CPY:           ggml_et_op_cpy(env, &node_meta[i]); break;
            case GGML_OP_CONCAT:        ggml_et_op_concat(env, &node_meta[i]); break;
            case GGML_OP_REPEAT:        ggml_et_op_repeat(env, &node_meta[i]); break;
            case GGML_OP_SSM_CONV:      ggml_et_op_ssm_conv(env, &node_meta[i]); break;
            case GGML_OP_SSM_SCAN:      ggml_et_op_ssm_scan(env, &node_meta[i]); break;
            case GGML_OP_PAD:           ggml_et_op_pad(env, &node_meta[i]); break;
            case GGML_OP_SET_ROWS:      ggml_et_op_set_rows(env, &node_meta[i]); break;
            case GGML_OP_FILL:          ggml_et_op_fill(env, &node_meta[i]); break;
            case GGML_OP_DIAG:          ggml_et_op_diag(env, &node_meta[i]); break;
            case GGML_OP_TRI:           ggml_et_op_tri(env, &node_meta[i]); break;
            case GGML_OP_SOLVE_TRI:     ggml_et_op_solve_tri(env, &node_meta[i]); break;
            case GGML_OP_SET:           ggml_et_op_set(env, &node_meta[i]); break;
            case GGML_OP_RWKV_WKV6:     ggml_et_op_rwkv_wkv6(env, &node_meta[i]); break;
            case GGML_OP_RWKV_WKV7:     ggml_et_op_rwkv_wkv7(env, &node_meta[i]); break;
            case GGML_OP_GATED_DELTA_NET: ggml_et_op_gated_delta_net(env, &node_meta[i]); break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // Metadata-only ops (no compute needed)
                break;

            default:
                // Log error logic would go here if available on device
                return -1; 
        }

        // Skip barrier for metadata-only ops and NONE
        if (op != GGML_OP_RESHAPE &&
            op != GGML_OP_VIEW    &&
            op != GGML_OP_PERMUTE &&
            op != GGML_OP_TRANSPOSE) {
            device_barrier(32);
        }
    }*/

    return 0;
}
