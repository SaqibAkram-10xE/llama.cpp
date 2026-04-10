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
#include <etsoc/common/utils.h>
#include "tensor.h"

//******************************************************************************
// Pull in standalone kernel.c files as monolithic includes.
// Each kernel checks this macro to rename entry_point to its callable name.
//******************************************************************************

#ifdef ENABLE_MONOLITHIC_COMPUTE
// Include kernel implementations
#include "el_map_f32.c"
#include "rms_norm_f32.c"
#include "glu_f32.c"
#include "softmax_f32.c"
#include "get_rows_f32.c"
#include "set_rows_f32.c"
#include "cont_f32.c"
#include "cont_f16.c"
#include "cpy_f32_f16.c"
#include "mul_mat_Q8_0.c"
#include "mul_mat_f16.c"
#include "mul_mat_f32.c"
#include "mul_mat_f16_matrix_engine.c"
#include "mul_mat_f32_matrix_engine.c"
#include "mul_mat_id_f32.c"
#include "rope_f32.c"
#include "flash_attn_ext_f32.c"
#include "flash_attn_ext_f16_me.c"
#include "cumsum_f32.c"
#include "diag_f32.c"
#include "fill_f32.c"
#include "scale_f32.c"
#include "set_f32.c"
#include "sqr_f32.c"
#include "sum_rows_f32.c"
#include "repeat_f32.c"
#include "pad_f32.c"
#include "unary_f32.c"
#include "tri_f32.c"
#include "solve_tri_f32.c"
#include "concat_f32.c"
#include "gated_delta_net_f32.c"
#include "group_norm_f32.c"
#include "norm_f32.c"
#include "l2_norm_f32.c"
#include "rms_norm_mul_f32.c"
#include "rwkv_wkv6_f32.c"
#include "rwkv_wkv7_f32.c"
#include "ssm_conv_f32.c"
#include "ssm_scan_f32.c"
#include "im2col.c"
// memops.c excluded - built as standalone kernel for memset operations
#else
// Stub definitions when ENABLE_MONOLITHIC_COMPUTE is disabled
// These won't be called since the monolithic path is not used

// Parameter structures needed for stub functions
// ggml_et_binary_params is already defined in ggml_tensor.h

struct ggml_et_rms_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    float eps;
};

struct ggml_et_glu_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    int32_t glu_op_type;
    int32_t swapped;
};

struct ggml_et_softmax_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
    float scale;
    float max_bias;
};

struct ggml_et_get_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct ggml_et_set_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct ggml_et_cont_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_cumsum_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_diag_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_fill_params {
    struct ggml_tensor dst;
    float c;
};

struct ggml_et_scale_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    float scale;
    float bias;
};

struct ggml_et_set_params {
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    int32_t nb1;
    int32_t nb2;
    int32_t nb3;
    int32_t offset1;
    int32_t offset2;
    int32_t offset3;
};

struct ggml_et_sqr_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_sum_rows_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_repeat_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
};

struct ggml_et_pad_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t pad_before[4];
    int32_t pad_after[4];
};

struct ggml_et_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t op_type;
};

struct ggml_et_tri_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t tri_type;
};

struct ggml_et_solve_tri_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct ggml_et_concat_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    int32_t dim;
};

struct ggml_et_gated_delta_net_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
};

struct ggml_et_group_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t n_groups;
    float eps;
};

struct ggml_et_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    float eps;
};

struct ggml_et_l2_norm_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    float eps;
};

struct ggml_et_rms_norm_mul_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
    float eps;
};

struct ggml_et_rwkv_wkv6_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor src3;
    struct ggml_tensor src4;
    struct ggml_tensor src5;
    float* td;
    float* state_in;
    float* dst;
    int32_t C;
    int32_t H;
    int32_t S;
    int32_t T;
    int32_t n_seqs;
};

struct ggml_et_rwkv_wkv7_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor src3;
    struct ggml_tensor dst;
    float* state_in;
    float* state_out;
    int32_t B;
    int32_t T;
    int32_t C;
    int32_t H;
};

struct ggml_et_ssm_conv_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct ggml_et_ssm_scan_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor src3;
    struct ggml_tensor src4;
    struct ggml_tensor src5;
    struct ggml_tensor src6;
    struct ggml_tensor dst;
};

// ggml_et_mul_mat_id_params is already defined in ggml_tensor.h

typedef struct {
    int32_t n_past;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t sections[4];
} rope_params_t;

struct ggml_et_rope_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
    rope_params_t rope_params;
};

// memset_params excluded - defined in memops.c which is built as standalone kernel

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor mask;
    struct ggml_tensor dst;
    float scale;
    float max_bias;
    float logit_softcap;
    int32_t has_mask;
};

static inline int el_map_f32_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_Q8_0_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f16_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f32_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f16_matrix_engine_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f32_matrix_engine_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_id_f32_impl(struct ggml_et_mul_mat_id_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rms_norm_f32_impl(struct ggml_et_rms_norm_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int glu_f32_impl(struct ggml_et_glu_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int softmax_f32_impl(struct ggml_et_softmax_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int get_rows_f32_impl(struct ggml_et_get_rows_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int set_rows_f32_impl(struct ggml_et_set_rows_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int cont_f32_impl(struct ggml_et_cont_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int cont_f16_impl(struct ggml_et_cont_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int cpy_f32_f16_impl(struct ggml_et_cont_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rope_f32_impl(struct ggml_et_rope_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int flash_attn_ext_f32_impl(struct ggml_et_flash_attn_ext_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int flash_attn_ext_f16_me_impl(struct ggml_et_flash_attn_ext_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int cumsum_f32_impl(struct ggml_et_cumsum_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int diag_f32_impl(struct ggml_et_diag_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int fill_f32_impl(struct ggml_et_fill_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int scale_f32_impl(struct ggml_et_scale_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int set_f32_impl(struct ggml_et_set_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int sqr_f32_impl(struct ggml_et_sqr_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int sum_rows_f32_impl(struct ggml_et_sum_rows_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int repeat_f32_impl(struct ggml_et_repeat_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int pad_f32_impl(struct ggml_et_pad_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int unary_f32_impl(struct ggml_et_unary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int tri_f32_impl(struct ggml_et_tri_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int solve_tri_f32_impl(struct ggml_et_solve_tri_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int concat_f32_impl(struct ggml_et_concat_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int gated_delta_net_f32_impl(struct ggml_et_gated_delta_net_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int group_norm_f32_impl(struct ggml_et_group_norm_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int norm_f32_impl(struct ggml_et_norm_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int l2_norm_f32_impl(struct ggml_et_l2_norm_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rms_norm_mul_f32_impl(struct ggml_et_rms_norm_mul_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rwkv_wkv6_f32_impl(struct ggml_et_rwkv_wkv6_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rwkv_wkv7_f32_impl(struct ggml_et_rwkv_wkv7_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int ssm_conv_f32_impl(struct ggml_et_ssm_conv_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int ssm_scan_f32_impl(struct ggml_et_ssm_scan_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int im2col_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
// memops_impl excluded - memops is built as standalone kernel
#endif

// ========================================================================
// Compact tensor metadata (ABI-compatible with host ggml_tensor_et)
// ========================================================================
struct ggml_tensor_et {
    int64_t ne[4];
    uint64_t nb[4];
    enum ggml_type type;
    uint64_t data;
};

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
    uint8_t data[];
};

// ========================================================================
// Helpers
// ========================================================================
// Convert ET tensor format to ggml tensor format
static inline void convert_to_ggml_tensor(struct ggml_tensor * d,
                                           struct ggml_tensor_et * s,
                                           enum ggml_op op) {
    memset(d, 0, sizeof(*d));
    d->type = s->type;
    d->data = (void*)(uintptr_t)s->data;
    d->op   = op;
    for (int j = 0; j < 4; j++) {
        d->ne[j] = s->ne[j];
        d->nb[j] = (size_t)s->nb[j];
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
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *)cg->data;
    uint8_t * node_op = (uint8_t *)(node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    // device_barrier(32);


    for (int i = 0; i < n_nodes; i++)
    {
        const int node_op_val = node_op[i];
        if (node_op_val == GGML_OP_NONE) continue;
        // device_barrier(32);
  
        /*switch (node_op_val) {
            case GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, node);
                break;

            case GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, node);
                break;

            case GGML_OP_SUM_ROWS:
                ggml_et_op_sum_rows(dev_ctx, node);
                break;

            case GGML_OP_SUB:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, node);
                break;

            case GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, node);
                break;

            case GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, node);
                break;

            case GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, node);
                break;

            case GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, node);
                break;

            case GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, node);
                break;

            case GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, node);
                break;

            case GGML_OP_GROUP_NORM:
                ggml_et_op_group_norm(dev_ctx, node);
                break;

            case GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, node);
                break;

            case GGML_OP_IM2COL:
                ggml_et_op_im2col(dev_ctx, node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, node);
                break;

            case GGML_OP_GET_ROWS:
                ggml_et_op_get_rows(dev_ctx, node);
                break;

            case GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, node);
                break;

            case GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, node);
                break;

            case GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, node);
                break;

            case GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, node);
                break;

            case GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, node);
                break;

            case GGML_OP_SSM_SCAN:
                ggml_et_op_ssm_scan(dev_ctx, node);
                break;

            case GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, node);
                break;

            case GGML_OP_SET_ROWS:
                ggml_et_op_set_rows(dev_ctx, node);
                break;

            case GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, node);
                break;

            case GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, node);
                break;

            case GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, node);
                break;

            case GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, node);
                break;

            case GGML_OP_SET:
                ggml_et_op_set(dev_ctx, node);
                break;

            case GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, node);
                break;

            case GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, node);
                break;

            case GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, node);
                break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are metadata-only operations that require no computation
                break;

            default:.
                break;
        }*/

        /*switch (op) {
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
        }*/
                
        // Skip barrier for metadata-only ops and NONE
        if (node_op_val != GGML_OP_RESHAPE &&
            node_op_val != GGML_OP_VIEW    &&
            node_op_val != GGML_OP_PERMUTE &&
            node_op_val != GGML_OP_TRANSPOSE) {
            // device_barrier(32);
            et_barrier(ET_BARRIER_GLOBAL);
        }
    }

    return 0;
}
