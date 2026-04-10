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

// Structure definitions needed outside ENABLE_MONOLITHIC_COMPUTE
struct ggml_et_im2col_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

// ROPE parameters (from rope_f32.c) - only define if not already defined
#ifndef rope_params_t_defined
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
#define rope_params_t_defined
#endif

// ROPE kernel parameters structure (from rope_f32.c) - only define if not already defined
#ifndef ggml_et_rope_params_defined
struct ggml_et_rope_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor dst;
    rope_params_t rope_params;
};
#define ggml_et_rope_params_defined
#endif

// Compact ROPE work descriptor (from rope_f32.c) - only define if not already defined
#ifndef rope_f32_work_t_defined
typedef struct {
    const float*   src0_data;
    const int32_t* src1_data;
    const float*   freq_factors;
    float*         dst_data;
    int64_t        ne[4];
    int64_t        src0_nb1, src0_nb2, src0_nb3;
    int64_t        dst_nb1, dst_nb2, dst_nb3;
    rope_params_t  rp;
} rope_f32_work_t;
#define rope_f32_work_t_defined
#endif

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
    int32_t offset;
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
    int32_t lp[4];
    int32_t rp[4];
};

struct ggml_et_unary_params {
    struct ggml_tensor src0;
    struct ggml_tensor dst;
    int32_t unary_op;
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
    struct ggml_tensor q;         // [S_v, H_q, n_tokens, n_seqs_q]
    struct ggml_tensor k;         // [S_v, H_k, n_tokens, n_seqs_k]
    struct ggml_tensor v;         // [S_v, H, n_tokens, n_seqs]
    struct ggml_tensor g;         // [1 or S_v, H, n_tokens, n_seqs]
    struct ggml_tensor beta;      // [1, H, n_tokens, n_seqs]
    struct ggml_tensor state_in;  // [S_v, S_v, H, n_seqs]
    struct ggml_tensor dst;       // [S_v*H, n_tokens*n_seqs + S_v*n_seqs]
    int32_t S_v;        // head dimension
    int32_t H;          // number of value heads
    int32_t H_q;        // number of Q heads
    int32_t H_k;        // number of K heads
    int32_t n_tokens;   // total tokens
    int32_t n_seqs;     // number of sequences
    int32_t n_seqs_q;   // Q sequence count
    int32_t n_seqs_k;   // K sequence count
    int32_t kda;        // 1 if per-element gate, 0 if scalar
    float   scale;      // 1/sqrt(S_v)
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
    float* k;           // src[0]: [S, H, T]  key
    float* v;           // src[1]: [S, H, T]  value
    float* r;           // src[2]: [S, H, T]  receptance
    float* tf;          // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    float* td;          // src[4]: [S, H, T]  time_decay
    float* state_in;    // src[5]: [S*S*H, n_seqs]  initial state
    float* dst;         // [C, T + S*n_seqs]  output + state_out
    int32_t C;          // total channels (S * H)
    int32_t H;          // number of heads
    int32_t S;          // head size
    int32_t T;          // number of tokens
    int32_t n_seqs;     // number of sequences
};

struct ggml_et_rwkv_wkv7_params {
    float* r;           // src[0]: [S, H, T]  receptance
    float* w;           // src[1]: [S, H, T]  decay
    float* k;           // src[2]: [S, H, T]  key
    float* v;           // src[3]: [S, H, T]  value
    float* a;           // src[4]: [S, H, T]  bonus gate
    float* b;           // src[5]: [S, H, T]  bonus key
    float* state_in;    // src[6]: [S*S*H, n_seqs]  initial state
    float* dst;         // [C, T + S*n_seqs]  output + state_out
    int32_t C;          // total channels (S * H)
    int32_t H;          // number of heads
    int32_t S;          // head size
    int32_t T;          // number of tokens
    int32_t n_seqs;     // number of sequences
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
// rope_params_t and rope_f32_work_t are defined outside ENABLE_MONOLITHIC_COMPUTE block
// ggml_et_rope_params is not needed in monolithic mode

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
static inline int im2col_f32_impl(struct ggml_et_im2col_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int im2col_impl(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
#ifndef rope_f32_compute_defined
static inline int rope_f32_compute(const rope_f32_work_t* w, int thread_id, int num_threads) { (void)w; (void)thread_id; (void)num_threads; return -1; }
#define rope_f32_compute_defined
#endif
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
    struct ggml_tensor_et src3;
    struct ggml_tensor_et src4;
    struct ggml_tensor_et src5;
    struct ggml_tensor_et src6;
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

#define GGML_ROPE_TYPE_MROPE  8

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

// Thread setup helper — returns -1 if this hart should not participate
static inline int cg_thread_setup(void * env, int * out_tid, int * out_nth) {
    kernel_environment_t * ke = (kernel_environment_t *)env;
    if (!ke) return -1;
    *out_tid = get_relative_thread_id(ke->shire_mask);
    *out_nth = get_num_threads(ke->shire_mask);
    if (*out_tid < 0) return -1;
    return 0;
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
  
        switch (node_op_val) {
            case GGML_OP_SQR:
                {
                    struct ggml_et_sqr_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SQR);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        sqr_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_UNARY:
                {
                    struct ggml_et_unary_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_UNARY);
                    params.unary_op = node_meta[i].op_params[0];
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        unary_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SUM_ROWS:
                {
                    struct ggml_et_sum_rows_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SUM_ROWS);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        sum_rows_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_MUL:
            case GGML_OP_ADD:
                {
                    struct ggml_et_binary_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    const enum ggml_op el_op = (node_op_val == GGML_OP_MUL) ? GGML_OP_MUL : GGML_OP_ADD;
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, el_op);

                    if (params.dst.type != GGML_TYPE_F32 ||
                        params.src0.type != GGML_TYPE_F32 ||
                        params.src1.type != GGML_TYPE_F32) {
                        break;
                    }
                    el_map_f32_impl(&params, env);
                }
                break;

            case GGML_OP_SUB:
                {
                    struct ggml_et_binary_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    const enum ggml_op el_op = GGML_OP_SUB;
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, el_op);

                    if (params.dst.type != GGML_TYPE_F32 ||
                        params.src0.type != GGML_TYPE_F32 ||
                        params.src1.type != GGML_TYPE_F32) {
                        break;
                    }
                    el_map_f32_impl(&params, env);
                }
                break;

            case GGML_OP_CUMSUM:
                {
                    struct ggml_et_cumsum_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CUMSUM);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        cumsum_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_MUL_MAT:
                {
                    struct ggml_et_binary_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_MUL_MAT);

                    if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_Q8_0 &&
                        params.src1.type == GGML_TYPE_F32) {
                        mul_mat_Q8_0_impl(&params, env);
                    }
                    else if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F16 &&
                        params.src1.type == GGML_TYPE_F16) {
                        // F16 x F16 -> F32 scalar path
                        int tid, nth;
                        if (cg_thread_setup(env, &tid, &nth)) break;
                        if (tid & 1) break; // skip odd threads
                        int eff_tid = tid / 2;
                        int eff_nth = (nth + 1) / 2;

                        const uint16_t * s0 = (const uint16_t *)params.src0.data;
                        const uint16_t * s1 = (const uint16_t *)params.src1.data;
                        float * d = (float *)params.dst.data;

                        const int64_t K = params.src0.ne[0];
                        const int64_t M = params.src0.ne[1];
                        const int64_t N = params.src1.ne[1];
                        const int64_t ne02 = params.src0.ne[2], ne03 = params.src0.ne[3];
                        const int64_t ne12 = params.src1.ne[2], ne13 = params.src1.ne[3];
                        const int64_t ne2  = params.dst.ne[2],  ne3  = params.dst.ne[3];

                        const size_t nb01 = params.src0.nb[1], nb02 = params.src0.nb[2], nb03 = params.src0.nb[3];
                        const size_t nb11 = params.src1.nb[1], nb12 = params.src1.nb[2], nb13 = params.src1.nb[3];
                        const size_t nb1  = params.dst.nb[1],  nb2  = params.dst.nb[2],  nb3  = params.dst.nb[3];

                        const int64_t r2 = ne12 / ne02;
                        const int64_t r3 = ne13 / ne03;

                        const int64_t total = M * N * ne2 * ne3;
                        const int64_t per_thread = 16;
                        const int64_t stride = per_thread * eff_nth;

                        for (int64_t base = eff_tid * per_thread; base < total; base += stride) {
                            for (int64_t j = 0; j < per_thread && (base + j) < total; j++) {
                                const int64_t idx = base + j;
                                const int64_t i3 = idx / (M * N * ne2);
                                const int64_t rem3 = idx % (M * N * ne2);
                                const int64_t i2 = rem3 / (M * N);
                                const int64_t rem2 = rem3 % (M * N);
                                const int64_t n = rem2 / M;
                                const int64_t m = rem2 % M;

                                const int64_t i03 = i3 / r3, i02 = i2 / r2;

                                const uint16_t * a_row = (const uint16_t *)((const char *)s0 + m * nb01 + i02 * nb02 + i03 * nb03);
                                const uint16_t * b_row = (const uint16_t *)((const char *)s1 + n * nb11 + i2 * nb12 + i3 * nb13);

                                float sum = 0.0f;
                                for (int64_t k = 0; k < K; k++) {
                                    sum += fp16_to_fp32(a_row[k]) * fp16_to_fp32(b_row[k]);
                                }

                                volatile float * out = (volatile float *)((char *)d + m * sizeof(float) + n * nb1 + i2 * nb2 + i3 * nb3);
                                atomic_store_f32(out, sum);
                            }
                        }
                    }
                    else if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F16 &&
                        params.src1.type == GGML_TYPE_F32) {
                        mul_mat_f16_impl(&params, env);
                    }
                    else if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F32 &&
                        params.src1.type == GGML_TYPE_F32) {
                        mul_mat_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_MUL_MAT_ID:
                break;

            case GGML_OP_ROPE:
                {
                    // Use compact rope_f32_work_t (~172 B) instead of the full
                    // ggml_et_rope_params (~1400 B with 4 ggml_tensor copies)
                    // to avoid stack overflow on per-hart firmware stacks.
                    if (node_meta[i].dst.type  == GGML_TYPE_F32 &&
                        node_meta[i].src0.type == GGML_TYPE_F32 &&
                        node_meta[i].src1.type == GGML_TYPE_I32) {

                        kernel_environment_t* ke = (kernel_environment_t*)env;
                        int tid = get_relative_thread_id(ke->shire_mask);
                        int nth = get_num_threads(ke->shire_mask);

                        if (tid >= 0) {
                            rope_f32_work_t w;
                            w.src0_data    = (const float*)(uintptr_t)node_meta[i].src0.data;
                            w.src1_data    = (const int32_t*)(uintptr_t)node_meta[i].src1.data;
                            w.freq_factors = node_meta[i].src2.data
                                           ? (const float*)(uintptr_t)node_meta[i].src2.data
                                           : NULL;
                            w.dst_data     = (float*)(uintptr_t)node_meta[i].dst.data;
                            for (int j = 0; j < 4; j++) w.ne[j] = node_meta[i].src0.ne[j];
                            w.src0_nb1 = (int64_t)node_meta[i].src0.nb[1];
                            w.src0_nb2 = (int64_t)node_meta[i].src0.nb[2];
                            w.src0_nb3 = (int64_t)node_meta[i].src0.nb[3];
                            w.dst_nb1  = (int64_t)node_meta[i].dst.nb[1];
                            w.dst_nb2  = (int64_t)node_meta[i].dst.nb[2];
                            w.dst_nb3  = (int64_t)node_meta[i].dst.nb[3];

                            w.rp.n_past     = ((const int32_t *) node_meta[i].op_params)[0];
                            w.rp.n_dims     = ((const int32_t *) node_meta[i].op_params)[1];
                            w.rp.mode       = ((const int32_t *) node_meta[i].op_params)[2];
                            w.rp.n_ctx      = ((const int32_t *) node_meta[i].op_params)[3];
                            w.rp.n_ctx_orig = ((const int32_t *) node_meta[i].op_params)[4];
                            memcpy(&w.rp.freq_base,   (const int32_t *) node_meta[i].op_params +  5, sizeof(float));
                            memcpy(&w.rp.freq_scale,  (const int32_t *) node_meta[i].op_params +  6, sizeof(float));
                            memcpy(&w.rp.ext_factor,  (const int32_t *) node_meta[i].op_params +  7, sizeof(float));
                            memcpy(&w.rp.attn_factor, (const int32_t *) node_meta[i].op_params +  8, sizeof(float));
                            memcpy(&w.rp.beta_fast,   (const int32_t *) node_meta[i].op_params +  9, sizeof(float));
                            memcpy(&w.rp.beta_slow,   (const int32_t *) node_meta[i].op_params + 10, sizeof(float));
                            if (w.rp.mode & GGML_ROPE_TYPE_MROPE) {
                                memcpy(w.rp.sections, (const int32_t *) node_meta[i].op_params + 11, sizeof(int32_t)*4);
                            } else {
                                memset(w.rp.sections, 0, sizeof(w.rp.sections));
                            }

                            rope_f32_compute(&w, tid, nth);
                        }
                    }
                }
                break;

            case GGML_OP_RMS_NORM:
                {
                    struct ggml_et_rms_norm_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_RMS_NORM);
                    memcpy(&params.eps, node_meta[i].op_params, sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        rms_norm_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_NORM:
                {
                    struct ggml_et_norm_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_NORM);
                    memcpy(&params.eps, node_meta[i].op_params, sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        norm_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_L2_NORM:
                {
                    struct ggml_et_l2_norm_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_L2_NORM);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        l2_norm_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_GROUP_NORM:
                {
                    struct ggml_et_group_norm_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GROUP_NORM);
                    memcpy(&params.eps, node_meta[i].op_params, sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        group_norm_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SCALE:
                {
                    struct ggml_et_scale_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SCALE);
                    memcpy(&params.scale, &node_meta[i].op_params[0], sizeof(float));
                    memcpy(&params.bias, &node_meta[i].op_params[1], sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        scale_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_GLU:
                {
                    struct ggml_et_glu_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GLU);
                    memcpy(&params.glu_op_type, &node_meta[i].op_params[0], sizeof(int32_t));
                    memcpy(&params.swapped, &node_meta[i].op_params[1], sizeof(int32_t));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32 &&
                        (params.glu_op_type == GGML_GLU_OP_SWIGLU || params.glu_op_type == GGML_GLU_OP_GEGLU)) {
                        glu_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SOFT_MAX:
                {
                    struct ggml_et_softmax_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src2, &node_meta[i].src2, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SOFT_MAX);
                    memcpy(&params.scale, &node_meta[i].op_params[0], sizeof(float));
                    memcpy(&params.max_bias, &node_meta[i].op_params[1], sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        softmax_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_IM2COL:
                {
                    struct ggml_et_binary_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_IM2COL);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        im2col_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                {
                    struct ggml_et_flash_attn_ext_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src2, &node_meta[i].src2, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_FLASH_ATTN_EXT);
                    memcpy(&params.scale, &node_meta[i].op_params[0], sizeof(float));
                    // Note: mask (src3) and sinks (src4) are not available in ggml_node_meta_et
                    // This implementation is limited to unmasked flash attention
                    params.has_mask = 0;
                    memset(&params.mask, 0, sizeof(params.mask));

                    // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
                    if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F32 &&
                        params.src1.type == GGML_TYPE_F16 &&
                        params.src2.type == GGML_TYPE_F16 &&
                        (params.src0.ne[0] % 32) == 0) {
                        flash_attn_ext_f16_me_impl(&params, env);
                    } else if (params.dst.type == GGML_TYPE_F32 &&
                               params.src0.type == GGML_TYPE_F32) {
                        flash_attn_ext_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_GET_ROWS:
                {
                    struct ggml_et_get_rows_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GET_ROWS);
                    if (params.dst.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_I32 &&
                        (params.src0.type == GGML_TYPE_F32 || params.src0.type == GGML_TYPE_F16 ||
                         params.src0.type == GGML_TYPE_Q8_0 || params.src0.type == GGML_TYPE_Q4_0 ||
                         params.src0.type == GGML_TYPE_Q4_K)) {
                        get_rows_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_CONT:
                {
                    void * src0_data = (void *)(uintptr_t)node_meta[i].src0.data;
                    void * dst_data  = (void *)(uintptr_t)node_meta[i].dst.data;
                    if (!src0_data || !dst_data) break;

                    const int64_t ne0 = node_meta[i].dst.ne[0], ne1 = node_meta[i].dst.ne[1];
                    const int64_t ne2 = node_meta[i].dst.ne[2], ne3 = node_meta[i].dst.ne[3];
                    const int64_t ne00 = node_meta[i].src0.ne[0], ne01 = node_meta[i].src0.ne[1];
                    const int64_t ne02 = node_meta[i].src0.ne[2], ne03 = node_meta[i].src0.ne[3];

                    const size_t nb00 = (size_t)node_meta[i].src0.nb[0], nb01 = (size_t)node_meta[i].src0.nb[1];
                    const size_t nb02 = (size_t)node_meta[i].src0.nb[2], nb03 = (size_t)node_meta[i].src0.nb[3];

                    if (node_meta[i].dst.type != node_meta[i].src0.type) {
                        break;
                    }

                    if (node_meta[i].dst.type == GGML_TYPE_F16) {
                        int tid, nth;
                        if (cg_thread_setup(env, &tid, &nth)) break;

                        const int64_t src_elements = ne00 * ne01 * ne02 * ne03;
                        const int64_t dst_elements = ne0 * ne1 * ne2 * ne3;
                        if (src_elements != dst_elements) {
                            break;
                        }

                        const int64_t total_rows = ne01;
                        const int64_t rows_per_thread = (total_rows + nth - 1) / nth;
                        const int64_t start_row = tid * rows_per_thread;
                        const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

                        if (start_row >= total_rows) {
                            break;
                        }

                        for (int64_t i03 = 0; i03 < ne03; i03++) {
                            for (int64_t i02 = 0; i02 < ne02; i02++) {
                                const int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

                                for (int64_t i01 = start_row; i01 < end_row; i01++) {
                                    const int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                                        const int64_t src_offset_bytes = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                                        const uint16_t* src_ptr = (const uint16_t*)((const char*)src0_data + src_offset_bytes);
                                        const int64_t dst_linear_idx = dst_linear_row_base + i00;

                                        atomic_store_f16((volatile uint16_t*)((char*)dst_data + dst_linear_idx * sizeof(uint16_t)), *src_ptr);
                                    }
                                }
                            }
                        }
                    } else if (node_meta[i].dst.type == GGML_TYPE_F32) {
                        struct ggml_et_cont_params params;
                        convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                        convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CONT);
                        cont_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_CPY:
                {
                    struct ggml_et_cont_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CPY);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        cont_f32_impl(&params, env);
                    } else if (params.dst.type == GGML_TYPE_F16 && params.src0.type == GGML_TYPE_F32) {
                        cpy_f32_f16_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_CONCAT:
                {
                    struct ggml_et_concat_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CONCAT);
                    params.dim = node_meta[i].op_params[0];
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_F32) {
                        concat_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_REPEAT:
                {
                    struct ggml_et_repeat_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_REPEAT);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        repeat_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SSM_CONV:
                {
                    struct ggml_et_ssm_conv_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SSM_CONV);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_F32) {
                        ssm_conv_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SSM_SCAN:
                {
                    struct ggml_et_ssm_scan_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src2, &node_meta[i].src2, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src3, &node_meta[i].src3, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src4, &node_meta[i].src4, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src5, &node_meta[i].src5, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src6, &node_meta[i].src6, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SSM_SCAN);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32 &&
                        params.src1.type == GGML_TYPE_F32 && params.src2.type == GGML_TYPE_F32 &&
                        params.src3.type == GGML_TYPE_F32 && params.src4.type == GGML_TYPE_F32 &&
                        params.src5.type == GGML_TYPE_F32 && params.src6.type == GGML_TYPE_I32) {
                        ssm_scan_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_PAD:
                {
                    struct ggml_et_pad_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_PAD);
                    // op_params: [pad_0_front, pad_0_back, pad_1_front, pad_1_back, pad_2_front, pad_2_back, pad_3_front, pad_3_back]
                    params.lp[0] = node_meta[i].op_params[0];
                    params.rp[0] = node_meta[i].op_params[1];
                    params.lp[1] = node_meta[i].op_params[2];
                    params.rp[1] = node_meta[i].op_params[3];
                    params.lp[2] = node_meta[i].op_params[4];
                    params.rp[2] = node_meta[i].op_params[5];
                    params.lp[3] = node_meta[i].op_params[6];
                    params.rp[3] = node_meta[i].op_params[7];
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        pad_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SET_ROWS:
                {
                    struct ggml_et_set_rows_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SET_ROWS);
                    if (params.src0.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_I64 &&
                        (params.dst.type == GGML_TYPE_F32 || params.dst.type == GGML_TYPE_F16)) {
                        set_rows_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_FILL:
                {
                    struct ggml_et_fill_params params;
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_FILL);
                    memcpy(&params.c, node_meta[i].op_params, sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32) {
                        fill_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_DIAG:
                {
                    struct ggml_et_diag_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_DIAG);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        diag_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_TRI:
                {
                    struct ggml_et_tri_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_TRI);
                    params.tri_type = node_meta[i].op_params[0];
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        tri_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SOLVE_TRI:
                {
                    struct ggml_et_solve_tri_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SOLVE_TRI);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_F32) {
                        solve_tri_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SET:
                {
                    struct ggml_et_set_params params;
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_SET);
                    params.nb1 = node_meta[i].op_params[0];
                    params.nb2 = node_meta[i].op_params[1];
                    params.nb3 = node_meta[i].op_params[2];
                    params.offset = node_meta[i].op_params[3];
                    if (params.dst.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_F32) {
                        set_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_RWKV_WKV6:
                {
                    struct ggml_et_rwkv_wkv6_params params;
                    params.k = (float*)node_meta[i].src0.data;
                    params.v = (float*)node_meta[i].src1.data;
                    params.r = (float*)node_meta[i].src2.data;
                    params.tf = (float*)node_meta[i].src3.data;
                    params.td = (float*)node_meta[i].src4.data;
                    params.state_in = (float*)node_meta[i].src5.data;
                    params.dst = (float*)node_meta[i].dst.data;
                    // Extract dimensions from op_params
                    params.C = node_meta[i].op_params[0];
                    params.H = node_meta[i].op_params[1];
                    params.S = node_meta[i].op_params[2];
                    params.T = node_meta[i].op_params[3];
                    params.n_seqs = node_meta[i].op_params[4];
                    rwkv_wkv6_f32_impl(&params, env);
                }
                break;

            case GGML_OP_RWKV_WKV7:
                {
                    struct ggml_et_rwkv_wkv7_params params;
                    params.r = (float*)node_meta[i].src0.data;
                    params.w = (float*)node_meta[i].src1.data;
                    params.k = (float*)node_meta[i].src2.data;
                    params.v = (float*)node_meta[i].src3.data;
                    params.a = (float*)node_meta[i].src4.data;
                    params.b = (float*)node_meta[i].src5.data;
                    params.state_in = (float*)node_meta[i].src6.data;
                    params.dst = (float*)node_meta[i].dst.data;
                    // Extract dimensions from op_params
                    params.C = node_meta[i].op_params[0];
                    params.H = node_meta[i].op_params[1];
                    params.S = node_meta[i].op_params[2];
                    params.T = node_meta[i].op_params[3];
                    params.n_seqs = node_meta[i].op_params[4];
                    rwkv_wkv7_f32_impl(&params, env);
                }
                break;

            case GGML_OP_GATED_DELTA_NET:
                {
                    struct ggml_et_gated_delta_net_params params;
                    convert_to_ggml_tensor(&params.q, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.k, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.v, &node_meta[i].src2, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.g, &node_meta[i].src3, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.beta, &node_meta[i].src4, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.state_in, &node_meta[i].src5, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GATED_DELTA_NET);
                    // Extract dimensions from op_params
                    params.S_v = node_meta[i].op_params[0];
                    params.H = node_meta[i].op_params[1];
                    params.n_tokens = node_meta[i].op_params[2];
                    params.n_seqs = node_meta[i].op_params[3];
                    params.H_q = node_meta[i].op_params[4];
                    params.H_k = node_meta[i].op_params[5];
                    params.n_seqs_q = node_meta[i].op_params[6];
                    params.n_seqs_k = node_meta[i].op_params[7];
                    params.kda = node_meta[i].op_params[8];
                    memcpy(&params.scale, &node_meta[i].op_params[9], sizeof(float));
                    if (params.dst.type == GGML_TYPE_F32) {
                        gated_delta_net_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are metadata-only operations that require no computation
                break;

            default:
                break;
        }

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
            device_barrier(32);
            // et_barrier(ET_BARRIER_GLOBAL);
        }
    }

    return 0;
}
