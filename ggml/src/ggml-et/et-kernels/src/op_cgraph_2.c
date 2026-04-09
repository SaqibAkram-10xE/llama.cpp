//******************************************************************************
// CGraph Kernel — Full device-side compute graph execution
// All ops implemented inline for single-kernel graph dispatch
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

// Pull in standalone kernel.c files as monolithic includes.
// Each kernel checks this macro to rename entry_point to its callable name.

#ifdef ENABLE_MONOLITHIC_COMPUTE
// Include kernel implementations
#include "el_map_f32.c"
#include "rms_norm_f32.c"
#include "glu_f32.c"
#include "softmax_f32.c"
#include "get_rows_f32.c"
#include "set_rows_f32.c"
#include "cont_f32.c"
#include "rope_f32.c"
#include "mul_mat_Q8_0.c"
#include "mul_mat_f16.c"
#include "mul_mat_f32.c"
#include "flash_attn_ext_f32.c"
#include "flash_attn_ext_f16_me.c"
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

// ggml_et_flash_attn_ext_params is defined in the kernel files when ENABLE_MONOLITHIC_COMPUTE is enabled
// When disabled, define it here for stub functions
#ifndef ENABLE_MONOLITHIC_COMPUTE
struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor src2;
    struct ggml_tensor mask;
    struct ggml_tensor dst;
    float scale;
    int32_t has_mask;
};
#endif

static inline int el_map_f32(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_Q8_0(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f16(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int mul_mat_f32(struct ggml_et_binary_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rms_norm_f32_impl(struct ggml_et_rms_norm_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int glu_f32_impl(struct ggml_et_glu_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int softmax_f32_impl(struct ggml_et_softmax_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int get_rows_f32_impl(struct ggml_et_get_rows_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int set_rows_f32_impl(struct ggml_et_set_rows_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int cont_f32_impl(struct ggml_et_cont_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int rope_f32_impl(struct ggml_et_rope_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int flash_attn_ext_f32_impl(struct ggml_et_flash_attn_ext_params* params, void* env) { (void)params; (void)env; return -1; }
static inline int flash_attn_ext_f16_me_impl(struct ggml_et_flash_attn_ext_params* params, void* env) { (void)params; (void)env; return -1; }
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

// ========================================================================
// Entry point — graph execution loop
// ========================================================================
int entry_point(struct ggml_cgraph_et * cg, void * env) {
    
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *)cg->data;
    uint8_t * node_op = (uint8_t *)(node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    for (int i = 0; i < n_nodes; i++)
    {
        const int node_op_val = node_op[i];
        if (node_op_val == GGML_OP_NONE) continue;

        switch (node_op_val) {
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
                    el_map_f32(&params, env);
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
                        mul_mat_Q8_0(&params, env);
                    }
                    else if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F16 &&
                        params.src1.type == GGML_TYPE_F32) {
                        mul_mat_f16(&params, env);
                    }
                    else if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F32 &&
                        params.src1.type == GGML_TYPE_F32) {
                        mul_mat_f32(&params, env);
                    }
                }
                break;

            case GGML_OP_MUL_MAT_ID:
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

            case GGML_OP_GET_ROWS:
                {
                    struct ggml_et_get_rows_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GET_ROWS);
                    if (params.dst.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_I32 &&
                        (params.src0.type == GGML_TYPE_F32 || params.src0.type == GGML_TYPE_Q8_0 ||
                         params.src0.type == GGML_TYPE_Q4_0 || params.src0.type == GGML_TYPE_Q4_K)) {
                        get_rows_f32_impl(&params, env);
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

            case GGML_OP_CONT:
                {
                    struct ggml_et_cont_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CONT);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        cont_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_ROPE:
                {
                    struct ggml_et_rope_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src2, &node_meta[i].src2, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_ROPE);
                    memcpy(&params.rope_params.n_past, &node_meta[i].op_params[0], sizeof(int32_t));
                    memcpy(&params.rope_params.n_dims, &node_meta[i].op_params[1], sizeof(int32_t));
                    memcpy(&params.rope_params.mode, &node_meta[i].op_params[2], sizeof(int32_t));
                    memcpy(&params.rope_params.n_ctx, &node_meta[i].op_params[3], sizeof(int32_t));
                    memcpy(&params.rope_params.n_ctx_orig, &node_meta[i].op_params[4], sizeof(int32_t));
                    memcpy(&params.rope_params.freq_base, &node_meta[i].op_params[5], sizeof(float));
                    memcpy(&params.rope_params.freq_scale, &node_meta[i].op_params[6], sizeof(float));
                    memcpy(&params.rope_params.ext_factor, &node_meta[i].op_params[7], sizeof(float));
                    memcpy(&params.rope_params.attn_factor, &node_meta[i].op_params[8], sizeof(float));
                    memcpy(&params.rope_params.beta_fast, &node_meta[i].op_params[9], sizeof(float));
                    memcpy(&params.rope_params.beta_slow, &node_meta[i].op_params[10], sizeof(float));
                    for (int j = 0; j < 4; j++) {
                        memcpy(&params.rope_params.sections[j], &node_meta[i].op_params[11 + j], sizeof(int32_t));
                    }
                    if (params.dst.type == GGML_TYPE_F32 &&
                        params.src0.type == GGML_TYPE_F32 &&
                        params.src1.type == GGML_TYPE_I32) {
                        rope_f32_impl(&params, env);
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

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                break;
        }

        // Synchronize all harts across all shires after each compute node.
        // Skip barrier for metadata-only ops (RESHAPE/VIEW/PERMUTE/TRANSPOSE)
        // since they don't touch data.

        if (node_op_val != GGML_OP_RESHAPE &&
            node_op_val != GGML_OP_VIEW &&
            node_op_val != GGML_OP_PERMUTE &&
            node_op_val != GGML_OP_TRANSPOSE &&
            node_op_val != GGML_OP_NONE) {
            // device_barrier_refined(32);
            device_barrier(32);
            // et_barrier()
        }

        // device_barrier(32);
        
    }


    return 0;
}
