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
#include "ggml_tensor.h"
#include "tensor.h"

// Forward declaration for ggml_et_get_rows_params
struct ggml_et_get_rows_params {
    struct ggml_tensor src0;     // Data tensor (F32 or Q8_0)
    struct ggml_tensor src1;     // Row indices tensor (I32)
    struct ggml_tensor dst;      // Output tensor (F32)
};

// Cache-related definitions
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_ELEMENTS(elem_size) (CACHE_LINE_SIZE_BYTES / (elem_size))

// Helper function declarations - these functions are already defined elsewhere in the file


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
// Forward declarations for ROPE helper functions
static inline void compute_rope_cache(
    float * cos_cache, float * sin_cache,
    int32_t n_dims, float theta_scale, int32_t pos,
    const float * freq_factors, float freq_scale,
    const float corr_dims[2], float ext_factor, float attn_factor);

static inline void compute_imrope_cache(
    float * cos_cache, float * sin_cache,
    int32_t n_dims, float theta_scale,
    int32_t pos_t, int32_t pos_h, int32_t pos_w, int32_t pos_e,
    const int32_t sections[4],
    const float * freq_factors, float freq_scale,
    const float corr_dims[2], float ext_factor, float attn_factor);

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

// Horizontal reduce of 8-wide vector accumulator in f10 -> scalar float
static inline float hsum_vec8(void) {
    float result;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );
    return result;
}

// Fusion check: can nodes i..i+n_ops-1 be fused with the given op sequence?
static inline bool ggml_et_can_fuse(struct ggml_cgraph_et * cg, int i,
                                     const uint8_t * node_op, int n_nodes,
                                     enum ggml_op ops[], int n_ops) {
    if (i + n_ops > n_nodes) return false;
    for (int k = 0; k < n_ops; k++) {
        if (node_op[i + k] != (uint8_t)ops[k]) return false;
    }
    // Check data flow: output of first feeds into second
    struct ggml_node_meta_et * meta = (struct ggml_node_meta_et *)cg->data;
    if (n_ops >= 2) {
        uint64_t out0 = meta[i].dst.data;
        if (out0 != meta[i+1].src0.data && out0 != meta[i+1].src1.data)
            return false;
    }
    return true;
}

// ========================================================================
// OP: SQR  —  dst[i] = src0[i] * src0[i]
// ========================================================================
static void ggml_et_op_sqr(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t total = m->dst.ne[0] * m->dst.ne[1] * m->dst.ne[2] * m->dst.ne[3];
    const int64_t elems_per_cl = 16;
    const int64_t total_cl = (total + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_t = (total_cl + nth - 1) / nth;
    const int64_t cl_s = tid * cl_per_t;
    int64_t cl_e = cl_s + cl_per_t;
    if (cl_e > total_cl) cl_e = total_cl;
    if (cl_s >= total_cl) return;

    const int64_t es = cl_s * elems_per_cl;
    int64_t ee = cl_e * elems_per_cl;
    if (ee > total) ee = total;

    for (int64_t i = es; i < ee; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[s]\n"
            "fmul.ps f11, f10, f10\n"
            "fsw.ps f11, %[d]\n"
            : [d] "=m"(*(float(*)[8])&dst[i])
            : [s] "m"(*(const float(*)[8])&src[i])
            : "f10", "f11"
        );
    }
}

// ========================================================================
// OP: SCALE  —  dst[i] = src0[i] * scale + bias
// ========================================================================
static void ggml_et_op_scale(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    float scale, bias;
    memcpy(&scale, &m->op_params[0], sizeof(float));
    memcpy(&bias,  &m->op_params[1], sizeof(float));

    const int64_t total = m->src0.ne[0] * m->src0.ne[1] * m->src0.ne[2] * m->src0.ne[3];
    const int64_t elems_per_cl = 16;
    const int64_t total_cl = (total + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_t = (total_cl + nth - 1) / nth;
    const int64_t cl_s = tid * cl_per_t;
    int64_t cl_e = cl_s + cl_per_t;
    if (cl_e > total_cl) cl_e = total_cl;
    if (cl_s >= total_cl) return;

    const int64_t es = cl_s * elems_per_cl;
    int64_t ee = cl_e * elems_per_cl;
    if (ee > total) ee = total;

    for (int64_t i = es; i < ee; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[s]\n"
            "fbc.ps f20, %[sc]\n"
            "fbc.ps f21, %[bi]\n"
            "fmadd.ps f10, f10, f20, f21\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(float(*)[8])&dst[i])
            : [s] "m"(*(const float(*)[8])&src[i]),
              [sc] "m"(scale), [bi] "m"(bias)
            : "f10", "f20", "f21"
        );
    }
}

// ========================================================================
// OP: FILL  —  dst[i] = constant
// ========================================================================
static void ggml_et_op_fill(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    float * dst = (float *)(uintptr_t)m->dst.data;
    if (!dst) return;

    float c;
    memcpy(&c, &m->op_params[0], sizeof(float));

    const int64_t total = m->dst.ne[0] * m->dst.ne[1] * m->dst.ne[2] * m->dst.ne[3];
    const int64_t elems_per_cl = 16;
    const int64_t total_cl = (total + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_t = (total_cl + nth - 1) / nth;
    const int64_t cl_s = tid * cl_per_t;
    int64_t cl_e = cl_s + cl_per_t;
    if (cl_e > total_cl) cl_e = total_cl;
    if (cl_s >= total_cl) return;

    const int64_t es = cl_s * elems_per_cl;
    int64_t ee = cl_e * elems_per_cl;
    if (ee > total) ee = total;

    __asm__ volatile("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");
    for (int64_t i = es; i < ee; i += 8) {
        __asm__ volatile(
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(float(*)[8])&dst[i])
            :: "f10"
        );
    }
}

// ========================================================================
// OP: UNARY  —  element-wise unary: SILU, GELU, RELU, TANH, etc.
// ========================================================================
static inline float apply_unary_scalar(float x, int32_t uop) {
    switch (uop) {
        case GGML_UNARY_OP_NEG:       return -x;
        case GGML_UNARY_OP_RELU:      return x > 0.0f ? x : 0.0f;
        case GGML_UNARY_OP_SIGMOID: {
            float ex = et_expf(-x);
            return et_fdiv(1.0f, 1.0f + ex);
        }
        case GGML_UNARY_OP_SILU: {
            if (x > 20.0f) return x;
            if (x < -20.0f) return 0.0f;
            float ex = et_expf(-x);
            return et_fdiv(x, 1.0f + ex);
        }
        case GGML_UNARY_OP_GELU: {
            float c = 0.044715f;
            float sq = 0.79788456080286535587989211986876f;
            float inner = sq * x * (1.0f + c * x * x);
            float t = 1.0f + et_expf(-2.0f * inner);
            return 0.5f * x * (1.0f + et_fdiv(2.0f, t) - 1.0f);
        }
        case GGML_UNARY_OP_GELU_QUICK: {
            return x * et_fdiv(1.0f, 1.0f + et_expf(-1.702f * x));
        }
        case GGML_UNARY_OP_TANH: {
            float e2x = et_expf(2.0f * x);
            return et_fdiv(e2x - 1.0f, e2x + 1.0f);
        }
        case GGML_UNARY_OP_EXP:       return et_expf(x);
        case GGML_UNARY_OP_ABS:       return x < 0.0f ? -x : x;
        case GGML_UNARY_OP_SGN:       return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case GGML_UNARY_OP_STEP:      return x > 0.0f ? 1.0f : 0.0f;
        case GGML_UNARY_OP_ELU:       return x >= 0.0f ? x : et_expf(x) - 1.0f;
        case GGML_UNARY_OP_HARDSWISH: {
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return x;
            return x * (x + 3.0f) * (1.0f / 6.0f);
        }
        case GGML_UNARY_OP_HARDSIGMOID: {
            if (x <= -3.0f) return 0.0f;
            if (x >= 3.0f) return 1.0f;
            return (x + 3.0f) * (1.0f / 6.0f);
        }
        default: return x;
    }
}

static void ggml_et_op_unary(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    int32_t uop = m->op_params[0];

    const int64_t total = m->dst.ne[0] * m->dst.ne[1] * m->dst.ne[2] * m->dst.ne[3];
    const int64_t elems_per_cl = 16;
    const int64_t total_cl = (total + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_t = (total_cl + nth - 1) / nth;
    const int64_t cl_s = tid * cl_per_t;
    int64_t cl_e = cl_s + cl_per_t;
    if (cl_e > total_cl) cl_e = total_cl;
    if (cl_s >= total_cl) return;

    const int64_t es = cl_s * elems_per_cl;
    int64_t ee = cl_e * elems_per_cl;
    if (ee > total) ee = total;

    for (int64_t i = es; i < ee; i++) {
        dst[i] = apply_unary_scalar(src[i], uop);
    }
}

// ========================================================================
// OP: DIAG  —  diagonal matrix from vector
// ========================================================================
static void ggml_et_op_diag(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne0  = m->dst.ne[0];
    const int64_t ne1  = m->dst.ne[1];
    const int64_t ne2  = m->dst.ne[2];
    const int64_t ne3  = m->dst.ne[3];
    const int64_t total_rows = ne1 * ne2 * ne3;

    float zero = 0.0f;
    __asm__ volatile("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        const int64_t i3 = row / (ne1 * ne2);

        float * dst_row = dst + (i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0;

        for (int64_t i = 0; i < ne0; i += 8) {
            __asm__ volatile(
                "fsw.ps f12, %[d]\n"
                : [d] "=m"(*(float(*)[8])&dst_row[i])
                :: "f12"
            );
        }
        if (i1 < ne00) {
            dst_row[i1] = src[i1 + i2 * ne00 + i3 * ne2 * ne00];
        }
    }
}

// ========================================================================
// OP: SUM_ROWS  —  row-wise sum reduction
// ========================================================================
static void ggml_et_op_sum_rows(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne03 = m->src0.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];

    const int64_t total_rows = ne01 * ne02 * ne03;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne02 * ne01);
        const int64_t i2 = (row - i3 * ne02 * ne01) / ne01;
        const int64_t i1 = row - i3 * ne02 * ne01 - i2 * ne01;

        const float * src_row = (const float *)((const char *)src + i3*nb03 + i2*nb02 + i1*nb01);

        float zero = 0.0f;
        __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        for (int64_t i0 = 0; i0 < ne00; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[s]\n"
                "fadd.ps f10, f10, f11\n"
                : : [s] "m"(*(const float(*)[8])&src_row[i0])
                : "f10", "f11"
            );
        }

        float sum = hsum_vec8();
        // dst layout: [1, ne01, ne02, ne03] contiguous
        atomic_store_f32((volatile float *)&dst[row], sum);
    }
}

// ========================================================================
// OP: CUMSUM  —  inclusive prefix sum along dim 0
// ========================================================================
static void ggml_et_op_cumsum(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne03 = m->src0.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];

    const int64_t total_rows = ne01 * ne02 * ne03;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne02 * ne01);
        const int64_t i2 = (row - i3 * ne02 * ne01) / ne01;
        const int64_t i1 = row - i3 * ne02 * ne01 - i2 * ne01;

        const float * src_row = (const float *)((const char *)src +
            i3 * (size_t)m->src0.nb[3] + i2 * (size_t)m->src0.nb[2] + i1 * nb01);
        float * dst_row = dst + row * ne00;

        float acc = 0.0f;
        for (int64_t i0 = 0; i0 < ne00; i0++) {
            acc += src_row[i0];
            dst_row[i0] = acc;
        }
    }
}

// ========================================================================
// OP: MUL / ADD / SUB  —  element-wise binary with broadcasting
// ========================================================================

// Block operation implementations using ET vector instructions
static inline void block_mul_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector multiplication
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fmul.ps f12, f10, f11\n"          // dst = src0 * src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_add_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector addition
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fadd.ps f12, f10, f11\n"          // dst = src0 + src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_sub_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector addition
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fsub.ps f12, f10, f11\n"          // dst = src0 + src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}


// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes via fbc.ps
static inline void block_mul_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fmul.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

static inline void block_add_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fadd.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

static inline void block_sub_broadcast(float* dst_block, const float* src0_block, float scalar, int elements) {
    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "fbc.ps f11, %[s]\n"
            "fsub.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [s] "m"(scalar)
            : "f10", "f11", "f12"
        );
    }
}

// ========================================================================
// OP: RMS_NORM  —  y = x / sqrt(mean(x^2) + eps)
// ========================================================================
static void ggml_et_op_rms_norm(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    float eps;
    memcpy(&eps, &m->op_params[0], sizeof(float));

    const int64_t ne0 = m->dst.ne[0];
    const int64_t ne1 = m->dst.ne[1];
    const int64_t ne2 = m->dst.ne[2];
    const int64_t ne3 = m->dst.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];
    const size_t nb1 = (size_t)m->dst.nb[1];
    const size_t nb2 = (size_t)m->dst.nb[2];
    const size_t nb3 = (size_t)m->dst.nb[3];

    const float inv_ne0 = et_fdiv(1.0f, (float)(int32_t)ne0);
    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne2 * ne1);
        const int64_t i2 = (row - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = row - i3 * ne2 * ne1 - i2 * ne1;

        const float * sp = (const float *)((const char *)src + i3*nb03 + i2*nb02 + i1*nb01);
        float * dp = (float *)((char *)dst + i3*nb3 + i2*nb2 + i1*nb1);

        // Sum of squares
        __asm__ volatile("fbci.pi f10, 0" ::: "f10");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                : : [x] "m"(*(const float(*)[8])&sp[i0])
                : "f10", "f11"
            );
        }
        float sum = hsum_vec8();
        const float scale = et_powf(sum * inv_ne0 + eps, -0.5f);

        // Apply scale
        uint32_t sb;
        __asm__ volatile("fmv.x.s %0, %1" : "=r"(sb) : "f"(scale));
        __asm__ volatile("fbcx.ps f13, %[s]\n" : : [s] "r"(sb) : "f13");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f12, %[x]\n"
                "fmul.ps f14, f12, f13\n"
                "fsw.ps f14, %[r]\n"
                : [r] "=m"(*(float(*)[8])&dp[i0])
                : [x] "m"(*(const float(*)[8])&sp[i0])
                : "f12", "f14"
            );
        }
    }
}

// ========================================================================
// OP: RMS_NORM_MUL (fused)  —  y = (x / sqrt(mean(x^2) + eps)) * weights
// ========================================================================
static void ggml_et_op_rms_norm_mul(void * env,
                                     struct ggml_node_meta_et * m_rms,
                                     struct ggml_node_meta_et * m_mul) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m_rms->src0.data;
    float * dst       = (float *)(uintptr_t)m_mul->dst.data;
    if (!src || !dst) return;

    // Weights: the MUL operand that isn't the rms_norm output
    const float * wgt;
    if (m_mul->src0.data == m_rms->dst.data) {
        wgt = (const float *)(uintptr_t)m_mul->src1.data;
    } else {
        wgt = (const float *)(uintptr_t)m_mul->src0.data;
    }
    if (!wgt) return;

    float eps;
    memcpy(&eps, &m_rms->op_params[0], sizeof(float));

    const int64_t ne0 = m_rms->dst.ne[0];
    const int64_t ne1 = m_rms->dst.ne[1];
    const int64_t ne2 = m_rms->dst.ne[2];
    const int64_t ne3 = m_rms->dst.ne[3];
    const size_t nb01 = (size_t)m_rms->src0.nb[1];
    const size_t nb02 = (size_t)m_rms->src0.nb[2];
    const size_t nb03 = (size_t)m_rms->src0.nb[3];
    const size_t nb1 = (size_t)m_mul->dst.nb[1];
    const size_t nb2 = (size_t)m_mul->dst.nb[2];
    const size_t nb3 = (size_t)m_mul->dst.nb[3];

    const float inv_ne0 = et_fdiv(1.0f, (float)(int32_t)ne0);
    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne2 * ne1);
        const int64_t i2 = (row - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = row - i3 * ne2 * ne1 - i2 * ne1;

        const float * sp = (const float *)((const char *)src + i3*nb03 + i2*nb02 + i1*nb01);
        float * dp = (float *)((char *)dst + i3*nb3 + i2*nb2 + i1*nb1);

        __asm__ volatile("fbci.pi f10, 0" ::: "f10");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                : : [x] "m"(*(const float(*)[8])&sp[i0])
                : "f10", "f11"
            );
        }
        float sum = hsum_vec8();
        const float scale = et_powf(sum * inv_ne0 + eps, -0.5f);

        uint32_t sb;
        __asm__ volatile("fmv.x.s %0, %1" : "=r"(sb) : "f"(scale));
        __asm__ volatile("fbcx.ps f13, %[s]\n" : : [s] "r"(sb) : "f13");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f12, %[x]\n"
                "flw.ps f15, %[w]\n"
                "fmul.ps f14, f12, f13\n"
                "fmul.ps f14, f14, f15\n"
                "fsw.ps f14, %[r]\n"
                : [r] "=m"(*(float(*)[8])&dp[i0])
                : [x] "m"(*(const float(*)[8])&sp[i0]),
                  [w] "m"(*(const float(*)[8])&wgt[i0])
                : "f12", "f14", "f15"
            );
        }
    }
}

// ========================================================================
// OP: NORM  —  y = (x - mean) / sqrt(var + eps)
// ========================================================================
static void ggml_et_op_norm(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    float eps;
    memcpy(&eps, &m->op_params[0], sizeof(float));

    const int64_t ne0 = m->dst.ne[0];
    const int64_t ne1 = m->dst.ne[1];
    const int64_t ne2 = m->dst.ne[2];
    const int64_t ne3 = m->dst.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];
    const size_t nb1 = (size_t)m->dst.nb[1];
    const size_t nb2 = (size_t)m->dst.nb[2];
    const size_t nb3 = (size_t)m->dst.nb[3];

    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne2 * ne1);
        const int64_t i2 = (row - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = row - i3 * ne2 * ne1 - i2 * ne1;

        const float * sp = (const float *)((const char *)src + i3*nb03 + i2*nb02 + i1*nb01);
        float * dp = (float *)((char *)dst + i3*nb3 + i2*nb2 + i1*nb1);

        // Pass 1: sum for mean
        float zero = 0.0f;
        __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n" "fadd.ps f10, f10, f11\n"
                : : [x] "m"(*(const float(*)[8])&sp[i0]) : "f10","f11");
        }
        float sum = hsum_vec8();
        const float mean = et_fdiv(sum, (float)(int32_t)ne0);

        // Pass 2: (x - mean) -> dst, accumulate variance
        __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n"
                "fbc.ps f12, %[mp]\n"
                "fsub.ps f13, f11, f12\n"
                "fsw.ps f13, %[r]\n"
                "fmadd.ps f10, f13, f13, f10\n"
                : [r] "=m"(*(float(*)[8])&dp[i0])
                : [x] "m"(*(const float(*)[8])&sp[i0]), [mp] "m"(mean)
                : "f10","f11","f12","f13");
        }
        float var_sum = hsum_vec8();
        const float variance = et_fdiv(var_sum, (float)(int32_t)ne0);
        const float sc = et_powf(variance + eps, -0.5f);

        // Pass 3: apply scale
        uint32_t sb;
        __asm__ volatile("fmv.x.s %0, %1" : "=r"(sb) : "f"(sc));
        __asm__ volatile("fbcx.ps f13, %[s]\n" : : [s] "r"(sb) : "f13");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f12, %[y]\n" "fmul.ps f14, f12, f13\n" "fsw.ps f14, %[r]\n"
                : [r] "=m"(*(float(*)[8])&dp[i0])
                : [y] "m"(*(const float(*)[8])&dp[i0])
                : "f12","f14");
        }
    }
}

// ========================================================================
// OP: L2_NORM  —  y = x / max(||x||_2, eps)
// ========================================================================
static void ggml_et_op_l2_norm(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst       = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    float eps;
    memcpy(&eps, &m->op_params[0], sizeof(float));

    const int64_t ne0 = m->dst.ne[0];
    const int64_t ne1 = m->dst.ne[1];
    const int64_t ne2 = m->dst.ne[2];
    const int64_t ne3 = m->dst.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];
    const size_t nb1 = (size_t)m->dst.nb[1];
    const size_t nb2 = (size_t)m->dst.nb[2];
    const size_t nb3 = (size_t)m->dst.nb[3];

    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i3 = row / (ne2 * ne1);
        const int64_t i2 = (row - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = row - i3 * ne2 * ne1 - i2 * ne1;

        const float * sp = (const float *)((const char *)src + i3*nb03 + i2*nb02 + i1*nb01);
        float * dp = (float *)((char *)dst + i3*nb3 + i2*nb2 + i1*nb1);

        __asm__ volatile("fbci.pi f10, 0" ::: "f10");
        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n" "fmadd.ps f10, f11, f11, f10\n"
                : : [x] "m"(*(const float(*)[8])&sp[i0]) : "f10","f11");
        }
        float sum_sq = hsum_vec8();
        float l2 = et_powf(sum_sq, 0.5f);
        if (l2 < eps) l2 = eps;
        const float sc = et_fdiv(1.0f, l2);

        for (int64_t i0 = 0; i0 < ne0; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x]\n" "fbc.ps f12, %[s]\n"
                "fmul.ps f13, f11, f12\n" "fsw.ps f13, %[r]\n"
                : [r] "=m"(*(float(*)[8])&dp[i0])
                : [x] "m"(*(const float(*)[8])&sp[i0]), [s] "m"(sc)
                : "f11","f12","f13");
        }
    }
}

// ========================================================================
// OP: GLU (SwiGLU / GeGLU)
// ========================================================================

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline float silu_f32(float x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    if (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        return x;
    } else if (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        return 0.0f;
    } else {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        float exp_neg_x = et_expf(-x);
        float denominator = 1.0f + exp_neg_x;
        return et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
static inline void block_geglu(float* dst_block, const float* x_block, const float* g_block, int elements) {
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    float one_const       = 1.0f;
    float coef_a_const    = 0.044715f;
    float sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    float two_log2e_const = 2.8853900817779268f;                 // 2 * log2(e)

    for (int32_t i = 0; i < elements; i += 8) {
        __asm__ volatile(
            // Load inputs
            "flw.ps f10, %[x_vec]\n"             // f10 = x
            "flw.ps f11, %[g_vec]\n"             // f11 = g

            // Broadcast constants
            "fbc.ps f20, %[one_ptr]\n"           // f20 = 1.0
            "fbc.ps f22, %[coef_ptr]\n"          // f22 = 0.044715
            "fbc.ps f23, %[sqrt2pi_ptr]\n"       // f23 = sqrt(2/pi)
            "fbc.ps f24, %[two_log2e_ptr]\n"     // f24 = 2*log2(e)

            // inner = 1 + 0.044715 * x^2
            "fmul.ps f12, f10, f10\n"            // f12 = x^2
            "fmadd.ps f13, f22, f12, f20\n"      // f13 = 1 + 0.044715*x^2

            // z = sqrt(2/pi) * x * inner
            "fmul.ps f14, f23, f10\n"            // f14 = sqrt(2/pi) * x
            "fmul.ps f14, f14, f13\n"            // f14 = z

            // exp(2z) via fexp.ps: feed z * 2*log2(e) since fexp computes 2^input
            "fmul.ps f15, f14, f24\n"            // f15 = 2z * log2(e)
            "fexp.ps f15, f15\n"                 // f15 = exp(2z)

            // gelu(x) = x * (1 - 1/(exp(2z)+1))  [NaN-safe: no inf*0]
            // exp(2z)->inf: rcp(inf)=0, 1-0=1, gelu=x
            // exp(2z)->0:   rcp(1)=1,   1-1=0, gelu=0
            "fadd.ps f16, f15, f20\n"            // f16 = exp(2z) + 1
            "frcp.ps f16, f16\n"                 // f16 = 1/(exp(2z) + 1)
            "fsub.ps f16, f20, f16\n"            // f16 = 1 - 1/(exp(2z)+1)
            "fmul.ps f16, f10, f16\n"            // f16 = gelu(x)

            // Final result
            "fmul.ps f18, f16, f11\n"            // f18 = gelu(x) * g

            "fsw.ps f18, %[dst_out]\n"

            : [dst_out] "=m"(*(float(*)[8])&dst_block[i])
            : [x_vec] "m"(*(const float(*)[8])&x_block[i]),
              [g_vec] "m"(*(const float(*)[8])&g_block[i]),
              [one_ptr] "m"(one_const),
              [coef_ptr] "m"(coef_a_const),
              [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18",
              "f20", "f22", "f23", "f24"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
static inline void block_swiglu(float* dst_block, const float* x_block, const float* g_block, int elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    float zero_const = 0.0f;
    float one_const = 1.0f;
    float log2e_const = 1.4426950408889634f;  // log2(e)

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ volatile(
            // Load input vectors
            "flw.ps f10, %[x_vec]\n"            // f10 = x[0..7]
            "flw.ps f11, %[g_vec]\n"            // f11 = g[0..7]

            // Broadcast constants to vector registers
            "fbc.ps f20, %[zero_ptr]\n"         // f20 = broadcast(0.0f) to all 8 elements
            "fbc.ps f21, %[one_ptr]\n"          // f21 = broadcast(1.0f) to all 8 elements

            // Compute -x (negate x by subtracting from zero)
            "fsub.ps f12, f20, f10\n"           // f12 = 0 - x = -x

            // Convert to base-2 exponent: -x * log2(e) = -x * 1.44269504
            // Load log2(e) constant
            "fbc.ps f22, %[log2e_ptr]\n"        // f22 = broadcast(1.44269504f)
            "fmul.ps f13, f12, f22\n"           // f13 = -x * log2(e)

            // Compute 2^(-x * log2(e)) = exp(-x)
            "fexp.ps f14, f13\n"                // f14 = 2^(-x * log2(e)) = exp(-x)

            // Compute 1 + exp(-x)
            "fadd.ps f15, f14, f21\n"           // f15 = exp(-x) + 1

            // Compute 1 / (1 + exp(-x)) using reciprocal
            "frcp.ps f16, f15\n"                // f16 = 1 / (1 + exp(-x))

            // Compute silu(x) = x * (1 / (1 + exp(-x)))
            "fmul.ps f17, f10, f16\n"           // f17 = x * (1 / (1 + exp(-x))) = silu(x)

            // Compute final result: silu(x) * g
            "fmul.ps f18, f17, f11\n"           // f18 = silu(x) * g

            // Store result
            "fsw.ps f18, %[dst_out]\n"          // Store 8 results to destination

            : [dst_out] "=m"(*(float(*)[8])&dst_block[i])
            : [x_vec] "m"(*(const float(*)[8])&x_block[i]),
              [g_vec] "m"(*(const float(*)[8])&g_block[i]),
              [zero_ptr] "m"(zero_const), // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),   // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const) // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Handle remaining elements (< 8) with scalar operations
    for (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = silu_f32(x_block[i]) * g_block[i];
    }
}

// ========================================================================
// OP: SOFTMAX
// ========================================================================
#define LOG2E_F 1.4426950408889634f

typedef struct {
    float max_val;
    float sum_val;
    uint32_t valid_mask;
} softmax_params_t;

static inline bool softmax_lane_is_valid(float x) {
    return (x == x) && (x != -INFINITY) && (x != INFINITY);
}

static inline softmax_params_t softmax_params_empty(void) {
    softmax_params_t p;
    p.max_val = -INFINITY;
    p.sum_val = 0.0f;
    p.valid_mask = 0;
    return p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
static inline void chunk_transform_ps_8_branchless_mask(
    float       *tmp8,
    const float *src,
    const float *mask,
    float scale,
    float slope)
{
    unsigned long ms;
    const float zero = 0.0f;
    const unsigned long mask_load_m0 = (mask != NULL) ? 0xFFul : 0x00ul;
    const float *mp = (mask != NULL) ? mask : &zero;

    __asm__ volatile (
        "mova.x.m  %[ms]                \n\t"

        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_scale])   \n\t"
        "fbc.ps    f11, 0(%[p_slope])   \n\t"
        "fbc.ps    f1, 0(%[p_zero])    \n\t"

        "mov.m.x   m0, %[maskm0], 0     \n\t" // load mask if needed
        "flw.ps    f1, 0(%[mp])         \n\t"

        "mov.m.x   m0, x0, 0xFF         \n\t"

        "flw.ps    f0, 0(%[sp])         \n\t"
        "fmul.ps   f0, f0, f10          \n\t"
        "fmul.ps   f1, f1, f11          \n\t"
        "fadd.ps   f0, f0, f1, rne      \n\t"
        "fsw.ps    f0, 0(%[tp])         \n\t"

        "mova.m.x  %[ms]                \n\t"
        : [ms] "=&r"(ms)
        : [tp]      "r"(tmp8),
          [sp]      "r"(src),
          [mp]      "r"(mp),
          [p_zero]  "r"(&zero),
          [p_scale] "r"(&scale),
          [p_slope] "r"(&slope),
          [maskm0]  "r"(mask_load_m0)
        : "f0", "f1", "f10", "f11", "memory"
    );
}

// softmax_pass1_range
//
// Computes the numerically-stable softmax scan over a sub-range of a row.
//
// This implements the 1st pass of online softmax
//
//   max' = max(max, x)
//   sum' = sum * exp(old_max - max') + exp(x - max')
//
// and returns a partial result containing:
//
//   - max_val : maximum logit observed in this range
//   - sum_val : exp-normalized sum relative to max_val
//
// These partial results can be merged with softmax_params_merge() to obtain
// the result for the full row.
static inline softmax_params_t softmax_pass1_range(
    const float *src,
    const float *mask,
    int begin,
    int end,
    float scale,
    float slope)
{
    __attribute__((aligned(32))) float lane_max[8];
    __attribute__((aligned(32))) float lane_sum[8];
    __attribute__((aligned(32))) float tmp[8];

    uint8_t valid_mask = 0;

    const float one_f   = 1.0f;
    const float zero_f  = 0.0f;
    const float neg_inf = -INFINITY;
    const float log2e   = LOG2E_F;

    unsigned long ms;

    __asm__ volatile (
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf]  "r"(&neg_inf),
          [p_zero]  "r"(&zero_f),
          [p_one]   "r"(&one_f),
          [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23"
    );

    int i = begin;
    for (; i < end; i += 8) {
        chunk_transform_ps_8_branchless_mask(tmp, src + i, mask ? (mask + i) : NULL, scale, slope);

        uint8_t cur_mask = 0;
        for (int j = 0; j < 8; ++j) {
            if (softmax_lane_is_valid(tmp[j])) {
                cur_mask |= (uint8_t)(1u << j);
            }
        }

        const uint8_t init_mask = (uint8_t)(cur_mask & ~valid_mask);
        const uint8_t upd_mask  = (uint8_t)(cur_mask &  valid_mask);

        if (init_mask || upd_mask) {
            __asm__ volatile (
                "flw.ps    f0, 0(%[p_tmp])       \n\t"

                "mov.m.x   m0, %[initm], 0       \n\t"
                "fcmovm.ps f20, f0,  f20         \n\t"
                "fcmovm.ps f21, f22, f21         \n\t"

                "mov.m.x   m0, %[updm], 0        \n\t"
                "fmax.ps   f1, f20, f0           \n\t"

                "fsub.ps   f2, f20, f1, rne      \n\t"
                "fmul.ps   f2, f2,  f23          \n\t"
                "fexp.ps   f2, f2                \n\t"

                "fsub.ps   f3, f0,  f1, rne      \n\t"
                "fmul.ps   f3, f3,  f23          \n\t"
                "fexp.ps   f3, f3                \n\t"

                "fmul.ps   f21, f21, f2          \n\t"
                "fadd.ps   f21, f21, f3, rne     \n\t"
                "fcmovm.ps f20, f1,  f20         \n\t"

                "mov.m.x   m0, x0, 0xFF          \n\t"
                :
                : [p_tmp] "r"(tmp),
                  [initm] "r"((unsigned long)init_mask),
                  [updm]  "r"((unsigned long)upd_mask)
                : "f0", "f1", "f2", "f3", "memory"
            );

            valid_mask |= cur_mask;
        }
    }

    __asm__ volatile (
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max),
          [p_lsum] "r"(lane_sum),
          [ms]     "r"(ms)
        : "memory"
    );

    softmax_params_t out = softmax_params_empty();
    out.valid_mask = valid_mask;

    for (int k = 0; k < 8; ++k) {
        if (valid_mask & (1u << k)) {
            if (out.valid_mask == (1u << k) || out.max_val == -INFINITY || lane_max[k] > out.max_val) {
                out.max_val = lane_max[k];
            }
        }
    }

    if (out.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        const float neg_max_l2 = -out.max_val * LOG2E_F;
        __attribute__((aligned(32))) float corr[8];
        __asm__ volatile (
            "mova.x.m  %[ms]              \n\t"
            "mov.m.x   m0, x0, 0xFF       \n\t"
            "fbc.ps    f0, 0(%[p_nml2])   \n\t"
            "fbc.ps    f2, 0(%[p_l2e])    \n\t"
            "flw.ps    f1, 0(%[p_lmax])   \n\t"
            "fmadd.ps  f0, f1, f2, f0     \n\t"
            "fexp.ps   f0, f0             \n\t"
            "fsw.ps    f0, 0(%[p_corr])   \n\t"
            "mova.m.x  %[ms]              \n\t"
            :
            : [p_nml2] "r"(&neg_max_l2),
              [p_l2e]  "r"(&log2e),
              [p_lmax] "r"(lane_max),
              [p_corr] "r"(corr),
              [ms]     "r"(ms)
            : "f0", "f1", "f2", "memory"
        );
        for (int k = 0; k < 8; ++k) {
            if (valid_mask & (1u << k)) {
                out.sum_val += lane_sum[k] * corr[k];
            }
        }
    }

    return out;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
static inline void softmax_pass2_range(
    float *dst,
    const float *src,
    const float *mask,
    int begin,
    int end,
    float scale,
    float slope,
    softmax_params_t params)
{
    const float s2      = scale * LOG2E_F;
    const float sl2     = slope * LOG2E_F;
    const float neg_ml2 = -params.max_val * LOG2E_F;
    const float inv_sum = et_fdiv(1.0f, params.sum_val);

    unsigned long ms;

    __asm__ volatile (
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2]   "r"(&s2),
          [p_nml2] "r"(&neg_ml2),
          [p_inv]  "r"(&inv_sum)
        : "f10", "f12", "f13"
    );

    if (mask != NULL) {
        __asm__ volatile (
            "fbc.ps    f11, 0(%[p_sl2]) \n\t"
            :
            : [p_sl2] "r"(&sl2)
            : "f11"
        );

        for (int c = begin; c < end; c += 8) {
            __asm__ volatile (
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(mask + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory"
            );
        }
    } else {
        for (int c = begin; c < end; c += 8) {
            __asm__ volatile (
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [dp] "r"(dst + c)
                : "f0", "memory"
            );
        }
    }

    __asm__ volatile (
        "mova.m.x  %[ms] \n\t"
        :: [ms] "r"(ms)
    );
}

// Single-core row path using the new structure.
static inline void compute_softmax_row(
    float *dst,
    const float *src,
    const float *mask,
    int cols,
    float scale,
    float slope,
    float sink_value,
    bool use_sinks)
{
    softmax_params_t params = softmax_pass1_range(src, mask, 0, cols, scale, slope);

    if (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        float max_val = params.max_val;
        if (sink_value > max_val) max_val = sink_value;

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float x = src[i] * scale;
            if (mask != NULL) x += mask[i] * slope;
            sum += et_expf(x - max_val);
        }
        sum += et_expf(sink_value - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        float inv_sum = et_fdiv(1.0f, sum);
        for (int i = 0; i < cols; ++i) {
            float x = src[i] * scale;
            if (mask != NULL) x += mask[i] * slope;
            dst[i] = et_expf(x - max_val) * inv_sum;
        }
    } else {
        if (!params.valid_mask) {
            return;
        }
        softmax_pass2_range(dst, src, mask, 0, cols, scale, slope, params);
    }
}
// ========================================================================
// OP: ROPE
// ========================================================================
// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_IMROPE 40
#define MAX_ROPE_HALF_DIMS 128  // supports up to n_dims=256

#define ROPE_VEC_WIDTH 8

#define ROPE_PI         3.14159265358979323846f
#define ROPE_TWO_PI     6.28318530717958647693f
#define ROPE_PI_OVER_2  1.57079632679489661923f
#define ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
typedef struct {
    int32_t n_past;
    int32_t n_dims;        // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;          // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;     // Base frequency (usually 10000.0f)
    float   freq_scale;    // Frequency scaling factor
    float   ext_factor;    // Extension factor for YaRN
    float   attn_factor;   // Attention factor for YaRN
    float   beta_fast;     // Fast beta for YaRN
    float   beta_slow;     // Slow beta for YaRN
    int32_t sections[4];   // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
struct ggml_et_rope_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor src1;  // I32 position tensor
    struct ggml_tensor src2;  // F32 frequency factors (optional)
    struct ggml_tensor dst;   // F32 output tensor
    rope_params_t rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    float denom = high - low;
    if (denom < 0.001f) denom = 0.001f;

    const float y = et_fdiv((float)(i0 / 2) - low, denom);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

static inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float beta, float freq_base) {
    return n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta) * 2.0f);
}

static inline void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                                       float beta_fast, float beta_slow, float dims[2]) {
    float start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
    float end   = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (float)(n_dims - 1) ? end : (float)(n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

static const float rope_ps_one[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f
};
static const float rope_ps_c3[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/6.0f,1.0f/6.0f,1.0f/6.0f,1.0f/6.0f,1.0f/6.0f,1.0f/6.0f,1.0f/6.0f,1.0f/6.0f
};
static const float rope_ps_c5[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/120.0f,1.0f/120.0f,1.0f/120.0f,1.0f/120.0f,1.0f/120.0f,1.0f/120.0f,1.0f/120.0f,1.0f/120.0f
};
static const float rope_ps_c7[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f,1.0f/5040.0f
};
static const float rope_ps_c9[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f,1.0f/362880.0f
};
static const float rope_ps_c11[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/39916800.0f,1.0f/39916800.0f,1.0f/39916800.0f,1.0f/39916800.0f,
    1.0f/39916800.0f,1.0f/39916800.0f,1.0f/39916800.0f,1.0f/39916800.0f
};

static inline uint64_t rope_ps_enter_fullmask(void) {
    uint64_t old_mask;
    __asm__ volatile(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory"
    );
    return old_mask;
}

static inline void rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ volatile(
        "mova.m.x %0           \n\t"
        :
        : "r"(old_mask)
        : "memory"
    );
}

static inline void rope_poly_sin_block8(float * out, const float * x) {
    __asm__ volatile(
        "flw.ps    f0,  %[x]           \n\t"
        "fmul.ps   f1,  f0,  f0        \n\t"

        "flw.ps    f2,  %[c11]         \n\t"
        "flw.ps    f3,  %[c9]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c7]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c5]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c3]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[one]         \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "fmul.ps   f4,  f0,  f2        \n\t"
        "fsw.ps    f4,  %[out]         \n\t"
        : [out] "=m"(*(float (*)[ROPE_VEC_WIDTH])out)
        : [x]   "m"(*(const float (*)[ROPE_VEC_WIDTH])x),
          [one] "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_one),
          [c3]  "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_c3),
          [c5]  "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_c5),
          [c7]  "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_c7),
          [c9]  "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_c9),
          [c11] "m"(*(const float (*)[ROPE_VEC_WIDTH])rope_ps_c11)
        : "f0","f1","f2","f3","f4","memory"
    );
}

static inline void rope_sincos_block8(
    float * sin8,
    float * cos8,
    const float * theta8) {

    float sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    float cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
        float x = theta8[i];

        if (x > ROPE_PI || x < -ROPE_PI) {
            float cycles = x * ROPE_INV_TWO_PI;
            int n = (int)cycles;
            if (x < 0.0f) {
                n--;
            }
            x = x - (float)n * ROPE_TWO_PI;
        }

        {
            float y = x;
            float s = 1.0f;
            if (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } else if (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            float y = x + ROPE_PI_OVER_2;
            if (y > ROPE_PI || y < -ROPE_PI) {
                float cycles = y * ROPE_INV_TWO_PI;
                int n = (int)cycles;
                if (y < 0.0f) {
                    n--;
                }
                y = y - (float)n * ROPE_TWO_PI;
            }

            float s = 1.0f;
            if (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } else if (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        const uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ volatile(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(float (*)[ROPE_VEC_WIDTH])sin8),
              [cout] "=m"(*(float (*)[ROPE_VEC_WIDTH])cos8)
            : [sinv]   "m"(*(const float (*)[ROPE_VEC_WIDTH])sin8),
              [sinsgn] "m"(*(const float (*)[ROPE_VEC_WIDTH])sin_sign),
              [cosv]   "m"(*(const float (*)[ROPE_VEC_WIDTH])cos8),
              [cossgn] "m"(*(const float (*)[ROPE_VEC_WIDTH])cos_sign)
            : "f0","f1","f2","f3","f4","f5","memory"
        );

        rope_ps_leave_fullmask(saved_mask);
    }
}

int rope_f32_impl(struct ggml_et_rope_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return -1;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* src2 = &params->src2;
    struct ggml_tensor* dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return -1;
    }

    const float* src0_data    = (const float*)src0->data;
    const int32_t* src1_data  = (const int32_t*)src1->data;
    const float* freq_factors = (src2 && src2->data) ? (const float*)src2->data : NULL;
    float* dst_data           = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1;
    }

    const int64_t head_dim = src0->ne[0];
    const int64_t heads    = src0->ne[1];
    const int64_t seq_len  = src0->ne[2];
    const int64_t batch    = src0->ne[3];

    const rope_params_t* rope_params = &params->rope_params;
    const int32_t n_dims   = rope_params->n_dims;
    const float freq_base  = rope_params->freq_base;
    const float freq_scale = rope_params->freq_scale;
    const int32_t mode     = rope_params->mode;

    if (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        return -1;
    }

    if (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        return -1;
    }

    float cos_cache[MAX_ROPE_HALF_DIMS];
    float sin_cache[MAX_ROPE_HALF_DIMS];

    float corr_dims[2];
    rope_yarn_corr_dims(
        n_dims,
        rope_params->n_ctx_orig,
        freq_base,
        rope_params->beta_fast,
        rope_params->beta_slow,
        corr_dims
    );

    // Distribute by individual heads: total = batch * seq_len * heads.
    const int64_t total_heads = batch * seq_len * heads;
    const int64_t start_wu = (total_heads * thread_id) / num_threads;
    const int64_t end_wu   = (total_heads * (thread_id + 1)) / num_threads;

    if (start_wu >= end_wu) {
        return 0;
    }

    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));
    const int32_t half_dims = n_dims / 2;
    const int is_neox   = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const int is_imrope = (mode == GGML_ROPE_TYPE_IMROPE);
    const int use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    for (int64_t wu = start_wu; wu < end_wu; ++wu) {
        const int64_t h = wu % heads;
        const int64_t s = (wu / heads) % seq_len;
        const int64_t b = wu / (heads * seq_len);

        if (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            const int32_t pt = src1_data[s]              + rope_params->n_past;
            const int32_t ph = src1_data[s + seq_len]    + rope_params->n_past;
            const int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            const int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            if (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(
                    cos_cache, sin_cache,
                    n_dims, theta_scale,
                    pt, ph, pw, pe,
                    rope_params->sections,
                    freq_factors, freq_scale,
                    corr_dims, rope_params->ext_factor, rope_params->attn_factor
                );
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } else {
            const int32_t pos = src1_data[s] + rope_params->n_past;

            if (pos != last_pos) {
                compute_rope_cache(
                    cos_cache, sin_cache,
                    n_dims, theta_scale, pos,
                    freq_factors, freq_scale,
                    corr_dims, rope_params->ext_factor, rope_params->attn_factor
                );
                last_pos = pos;
            }
        }

        const float* head_src = (const float*)((const char*)src0_data +
            b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        float* head_dst = (float*)((char*)dst_data +
            b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        for (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        if (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
            __asm__ volatile("mov.m.x m0, x0, 0xFF");

            for (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ volatile(
                    "flw.ps f0, %[x0_src]       \n\t"
                    "flw.ps f1, %[x1_src]       \n\t"
                    "flw.ps f2, %[sin_cache]    \n\t"
                    "flw.ps f3, %[cos_cache]    \n\t"
                    "fmul.ps f4, f0, f3         \n\t"
                    "fmul.ps f5, f0, f2         \n\t"
                    "fnmsub.ps f4, f1, f2, f4   \n\t"
                    "fmadd.ps f5, f1, f3, f5    \n\t"
                    "fsw.ps f4, %[x0_dst]       \n\t"
                    "fsw.ps f5, %[x1_dst]       \n\t"
                    : [x0_dst] "=m"(*(float(*)[8])&head_dst[dim_idx]),
                      [x1_dst] "=m"(*(float(*)[8])&head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(const float(*)[8])&head_src[dim_idx]),
                      [x1_src] "m"(*(const float(*)[8])&head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(const float(*)[8])&sin_cache[dim_idx]),
                      [cos_cache] "m"(*(const float(*)[8])&cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory"
                );
            }

            __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
        } else {
            // Standard: adjacent pairs (2i, 2i+1)
            for (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                const int32_t dim_in_head = pair_idx * 2;
                const float x0 = head_src[dim_in_head];
                const float x1 = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    return 0;
}

#define GGML_ROPE_TYPE_NEOX_CG   2
#define GGML_ROPE_TYPE_MROPE_CG  8
#define GGML_ROPE_TYPE_IMROPE_CG 40

static void ggml_et_op_rope(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src0_data   = (const float *)(uintptr_t)m->src0.data;
    const int32_t * src1_data = (const int32_t *)(uintptr_t)m->src1.data;
    const float * freq_factors = m->src2.data ? (const float *)(uintptr_t)m->src2.data : 0;
    float * dst_data          = (float *)(uintptr_t)m->dst.data;
    if (!src0_data || !src1_data || !dst_data) return;

    const int64_t head_dim = m->src0.ne[0];
    const int64_t heads    = m->src0.ne[1];
    const int64_t seq_len  = m->src0.ne[2];
    const int64_t batch    = m->src0.ne[3];

    int32_t n_past     = m->op_params[0];
    int32_t n_dims     = m->op_params[1];
    int32_t mode       = m->op_params[2];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int32_t n_ctx_orig = m->op_params[4];
    memcpy(&freq_base,   &m->op_params[5],  sizeof(float));
    memcpy(&freq_scale,  &m->op_params[6],  sizeof(float));
    memcpy(&ext_factor,  &m->op_params[7],  sizeof(float));
    memcpy(&attn_factor, &m->op_params[8],  sizeof(float));
    memcpy(&beta_fast,   &m->op_params[9],  sizeof(float));
    memcpy(&beta_slow,   &m->op_params[10], sizeof(float));

    if (n_dims <= 0 || n_dims > head_dim) return;

    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));
    const int32_t half_dims = n_dims / 2;
    const int is_neox = (mode & GGML_ROPE_TYPE_NEOX_CG) != 0;

    // YaRN correction dimensions
    float corr_dims[2] = {0.0f, 0.0f};
    if (n_ctx_orig > 0 && beta_fast > 0.0f) {
        float cd_s = (float)n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta_fast) * 2.0f);
        float cd_e = (float)n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta_slow) * 2.0f);
        corr_dims[0] = cd_s > 0.0f ? cd_s : 0.0f;
        corr_dims[1] = cd_e < (float)(n_dims - 1) ? cd_e : (float)(n_dims - 1);
    }

    const int64_t total_heads = batch * seq_len * heads;

    for (int64_t wu = tid; wu < total_heads; wu += nth) {
        const int64_t h = wu % heads;
        const int64_t s = (wu / heads) % seq_len;
        const int64_t b = wu / (heads * seq_len);

        const int32_t pos = src1_data[s] + n_past;

        const float * head_src = (const float *)((const char *)src0_data +
            b * (size_t)m->src0.nb[3] + s * (size_t)m->src0.nb[2] + h * (size_t)m->src0.nb[1]);
        float * head_dst = (float *)((char *)dst_data +
            b * (size_t)m->dst.nb[3] + s * (size_t)m->dst.nb[2] + h * (size_t)m->dst.nb[1]);

        // Copy dims beyond n_dims
        for (int64_t d = n_dims; d < head_dim; d++) {
            head_dst[d] = head_src[d];
        }

        // Build cache and apply rotation
        float theta = 1.0f;
        if (is_neox) {
            for (int32_t di = 0; di < half_dims; di++) {
                const float ff = freq_factors ? freq_factors[di] : 1.0f;
                const float theta_base = (float)pos * theta;
                float theta_ext = et_fdiv(theta_base, ff);
                float theta_interp = freq_scale * theta_ext;
                float theta_final = theta_interp;

                if (ext_factor != 0.0f) {
                    float denom = corr_dims[1] - corr_dims[0];
                    if (denom < 0.001f) denom = 0.001f;
                    float y = et_fdiv((float)(di) - corr_dims[0], denom);
                    float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
                    float ramp = (1.0f - clamped) * ext_factor;
                    theta_final = theta_interp * (1.0f - ramp) + theta_ext * ramp;
                }

                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
                }

                float cos_t = et_cosf(theta_final) * mscale;
                float sin_t = et_sinf(theta_final) * mscale;

                float x0 = head_src[di];
                float x1 = head_src[di + half_dims];
                head_dst[di]             = x0 * cos_t - x1 * sin_t;
                head_dst[di + half_dims] = x0 * sin_t + x1 * cos_t;

                theta *= theta_scale;
            }
        } else {
            // Standard adjacent-pair rotation
            for (int32_t di = 0; di < half_dims; di++) {
                const float ff = freq_factors ? freq_factors[di] : 1.0f;
                const float theta_base = (float)pos * theta;
                float theta_ext = et_fdiv(theta_base, ff);
                float theta_interp = freq_scale * theta_ext;
                float theta_final = theta_interp;

                if (ext_factor != 0.0f) {
                    float denom = corr_dims[1] - corr_dims[0];
                    if (denom < 0.001f) denom = 0.001f;
                    float y = et_fdiv((float)(di) - corr_dims[0], denom);
                    float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
                    float ramp = (1.0f - clamped) * ext_factor;
                    theta_final = theta_interp * (1.0f - ramp) + theta_ext * ramp;
                }

                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
                }

                float cos_t = et_cosf(theta_final) * mscale;
                float sin_t = et_sinf(theta_final) * mscale;

                int32_t d = di * 2;
                float x0 = head_src[d];
                float x1 = head_src[d + 1];
                head_dst[d]     = x0 * cos_t - x1 * sin_t;
                head_dst[d + 1] = x0 * sin_t + x1 * cos_t;

                theta *= theta_scale;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
static inline void rope_yarn_scalar(
    float theta_extrap,
    float freq_scale,
    const float corr_dims[2],
    int64_t i0,
    float ext_factor,
    float mscale,
    float * cos_theta,
    float * sin_theta) {

    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
    }

    *cos_theta = et_cosf(theta) * mscale;
    *sin_theta = et_sinf(theta) * mscale;
}

// Populate cos/sin cache for a given position using running theta product
// Experiment 1:
//   - theta construction and YaRN mixing stay scalar
//   - actual sin/cos approximation is done in vec8 blocks
static inline void compute_rope_cache(
    float * cos_cache, float * sin_cache,
    int32_t n_dims, float theta_scale, int32_t pos,
    const float * freq_factors, float freq_scale,
    const float corr_dims[2], float ext_factor, float attn_factor) {

    const int32_t half_dims = n_dims / 2;
    float theta = 1.0f;

    int32_t dim_idx = 0;

    for (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        float theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        float theta_local = theta;
        float mscale = attn_factor;

        if (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            const int32_t pair_idx = dim_idx + i;
            const float ff = freq_factors ? freq_factors[pair_idx] : 1.0f;
            const float theta_base = (float)pos * theta_local;
            const float theta_extrap = et_fdiv(theta_base, ff);

            float theta_interp = freq_scale * theta_extrap;
            float theta_mix = theta_interp;

            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    for (; dim_idx < half_dims; ++dim_idx) {
        const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
        const float theta_base = (float)pos * theta;

        rope_yarn_scalar(
            et_fdiv(theta_base, ff),
            freq_scale,
            corr_dims,
            dim_idx * 2,
            ext_factor,
            attn_factor,
            &cos_cache[dim_idx],
            &sin_cache[dim_idx]
        );

        theta *= theta_scale;
    }
}

//------------------------------------------------------------------------------
// IMROPE cache build (interleaved multi-modal RoPE for Qwen3VL)
//------------------------------------------------------------------------------

// Builds cos/sin cache with 4 interleaved position channels.
// Each dimension pair selects from {theta_t, theta_h, theta_w, theta_e}
// using a mod-3 sector pattern, matching the CPU reference exactly.
static inline void compute_imrope_cache(
    float * cos_cache, float * sin_cache,
    int32_t n_dims, float theta_scale,
    int32_t pos_t, int32_t pos_h, int32_t pos_w, int32_t pos_e,
    const int32_t sections[4],
    const float * freq_factors, float freq_scale,
    const float corr_dims[2], float ext_factor, float attn_factor) {

    const int32_t half_dims = n_dims / 2;
    const int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    float theta_t = (float)pos_t;
    float theta_h = (float)pos_h;
    float theta_w = (float)pos_w;
    float theta_e = (float)pos_e;

    int32_t dim_idx = 0;

    for (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        float theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        float mscale = attn_factor;

        if (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            const int32_t pair_idx = dim_idx + i;
            const int32_t sector = pair_idx % sect_dims;
            const float ff = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            float theta;
            if      (sector % 3 == 1 && sector < 3 * sections[1]) { theta = theta_h; }
            else if (sector % 3 == 2 && sector < 3 * sections[2]) { theta = theta_w; }
            else if (sector % 3 == 0 && sector < 3 * sections[0]) { theta = theta_t; }
            else                                                   { theta = theta_e; }

            const float theta_extrap = et_fdiv(theta, ff);
            float theta_interp = freq_scale * theta_extrap;
            float theta_mix = theta_interp;

            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        for (int i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    for (; dim_idx < half_dims; ++dim_idx) {
        const int32_t sector = dim_idx % sect_dims;
        const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;

        float theta;
        if      (sector % 3 == 1 && sector < 3 * sections[1]) { theta = theta_h; }
        else if (sector % 3 == 2 && sector < 3 * sections[2]) { theta = theta_w; }
        else if (sector % 3 == 0 && sector < 3 * sections[0]) { theta = theta_t; }
        else                                                   { theta = theta_e; }

        rope_yarn_scalar(
            et_fdiv(theta, ff),
            freq_scale,
            corr_dims,
            dim_idx * 2,
            ext_factor,
            attn_factor,
            &cos_cache[dim_idx],
            &sin_cache[dim_idx]
        );

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}


// ========================================================================
// OP: MUL_MAT  —  C[M,N] = A[M,K] * B[K,N]
// Supports Q8_0, F16, F32 weight types; F32 activations
// ========================================================================
#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0   // SCP lines  0..15 for A
#define B_L1_START 16  // SCP lines 16..31 for B

typedef uint16_t et_fp16_t;


#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32
#define TILE_K_TFMA_F32    16
#define TILE_M_TFMA_F32    16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#define TILE_N_TFMA_F32             16
#define CACHEOP_MAX_TFMA_F32        0
#define REP_RATE_TFMA_F32           0
/* ─────────────────────────────────────────────────────────────────── */


static inline void __attribute__((always_inline))
pack_b_interleaved(et_fp16_t *out,
                   const char *src0_batch,
                   int64_t mb, int64_t kb, int64_t nb1_0)
{
    for (int j = 0; j < TILE_M; ++j) {
        const et_fp16_t *row =
            (const et_fp16_t *)(src0_batch + (mb + j) * nb1_0) + kb;
        for (int l = 0; l < TILE_K / 2; ++l) {
            out[l * 32 + j * 2 + 0] = row[2 * l + 0];
            out[l * 32 + j * 2 + 1] = row[2 * l + 1];
        }
    }
}


// ========================================================================
// OP: MUL_MAT_ID  —  Mixture of Experts matmul
// ========================================================================
static void ggml_et_op_mul_mat_id(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    if (tid & 1) return;
    int etid = tid / 2;
    int enth = (nth + 1) / 2;

    const int64_t K = m->src0.ne[0];
    const int64_t M = m->src0.ne[1];
    const int64_t n_expert = m->src0.ne[2];
    const int64_t n_expert_used = m->src2.ne[0];
    const int64_t batch_sz = m->src2.ne[1];

    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb11 = (size_t)m->src1.nb[1];
    const size_t nb12 = (size_t)m->src1.nb[2];
    const size_t nb1  = (size_t)m->dst.nb[1];
    const size_t nb2  = (size_t)m->dst.nb[2];

    const void * src0_data = (const void *)(uintptr_t)m->src0.data;
    const float * src1_data = (const float *)(uintptr_t)m->src1.data;
    const int32_t * src2_data = (const int32_t *)(uintptr_t)m->src2.data;
    float * dst_data = (float *)(uintptr_t)m->dst.data;
    if (!src0_data || !src1_data || !src2_data || !dst_data) return;

    const int src0_type = m->src0.type;
    const int64_t ne11_val = m->src1.ne[1];

    const uint64_t total_elems = (uint64_t)M * n_expert_used * batch_sz;
    const uint64_t per_thread = 16;
    const uint64_t stride = per_thread * enth;

    for (uint64_t base = etid * per_thread; base < total_elems; base += stride) {
        for (uint64_t j = 0; j < per_thread && base + j < total_elems; j++) {
            uint64_t idx = base + j;
            int64_t b_idx = idx / (M * n_expert_used);
            int64_t rem_val = idx % (M * n_expert_used);
            int64_t n_idx = rem_val / M;
            int64_t mm = rem_val % M;

            int32_t expert_id = src2_data[b_idx * n_expert_used + n_idx];
            if (expert_id < 0 || expert_id >= n_expert) {
                volatile float * c = (volatile float *)((char *)dst_data + mm * sizeof(float) + n_idx * nb1 + b_idx * nb2);
                atomic_store_f32(c, 0.0f);
                continue;
            }

            int64_t col_idx = n_idx % ne11_val;
            const char * expert_row = (const char *)src0_data + mm * nb01 + expert_id * nb02;
            float sum = 0.0f;

            if (src0_type == GGML_TYPE_Q8_0) {
                int64_t K_blocks = K / 32;
                const block_q8_0 * qr = (const block_q8_0 *)expert_row;
                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * bp = (const float *)((const char *)src1_data + kb * 32 * sizeof(float) + col_idx * nb11 + b_idx * nb12);
                    sum += compute_block_dot_product_q8_0(&qr[kb], bp);
                }
            } else if (src0_type == GGML_TYPE_F16) {
                const uint16_t * f16r = (const uint16_t *)expert_row;
                int64_t K_blocks = K / QK_F16;
                int64_t K_rem = K % QK_F16;
                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * bp = (const float *)((const char *)src1_data + kb * QK_F16 * sizeof(float) + col_idx * nb11 + b_idx * nb12);
                    sum += compute_block_dot_product_f16_naive(&f16r[kb * QK_F16], bp);
                }
                if (K_rem > 0) {
                    int64_t off = K_blocks * QK_F16;
                    const float * bp = (const float *)((const char *)src1_data + off * sizeof(float) + col_idx * nb11 + b_idx * nb12);
                    sum += compute_block_dot_product_f16_partial(&f16r[off], bp, K_rem);
                }
            } else {
                const float * f32r = (const float *)expert_row;
                int64_t K_blocks = K / QK_F32;
                int64_t K_rem = K % QK_F32;
                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * bp = (const float *)((const char *)src1_data + kb * QK_F32 * sizeof(float) + col_idx * nb11 + b_idx * nb12);
                    sum += compute_block_dot_product_f32(&f32r[kb * QK_F32], bp);
                }
                if (K_rem > 0) {
                    int64_t off = K_blocks * QK_F32;
                    const float * bp = (const float *)((const char *)src1_data + off * sizeof(float) + col_idx * nb11 + b_idx * nb12);
                    sum += compute_block_dot_product_f32_partial(&f32r[off], bp, K_rem);
                }
            }

            volatile float * c = (volatile float *)((char *)dst_data + mm * sizeof(float) + n_idx * nb1 + b_idx * nb2);
            atomic_store_f32(c, sum);
        }
    }
}

// ========================================================================
// OP: GET_ROWS  —  row extraction with dequant (F32/Q8_0/Q4_0/Q4_K)
// ========================================================================


#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_ELEMENTS(elem_size) (CACHE_LINE_SIZE_BYTES / (elem_size))

// Copy a row of F32 data from source to destination
static void copy_f32_row(float* dst, const float* src, int64_t num_elements) {
    // Simple memcpy for F32 data - no conversion needed
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
static void copy_row_cache_align(float* dst, const float* src, int64_t n_bytes) {
    int num_f32_elem = n_bytes / sizeof(float);

    // Unrolled to do an entire cache line at a time
    __asm__ volatile (
        "1: \n\t"
        // --- Process 64 Bytes (1 Cache Line) ---
        // Load 256 bits (32 bytes) into f0 and the other into f1
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"

        // Store 256 bits (32 bytes) from f0 and f1
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"

        // Increment Pointers by 64 bytes
        "addi %[src], %[src], 64 \n\t"
        "addi %[dst], %[dst], 64 \n\t"

        // Decrement count by 16 elements
        "addi %[n], %[n], -16 \n\t"

        // Loop if at least 16 elements remain
        "bge %[n], %[stride_count], 1b \n\t"

        : [dst] "+r" (dst), [src] "+r" (src), [n] "+r" (num_f32_elem)
        : [stride_count] "r" (16L)
        : "f0", "f1", "memory"
    );
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
static void copy_q4_0_row(float* dst, const block_q4_0* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK4_0 - 1) / QK4_0;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK4_0) : QK4_0;

        float temp_buffer[QK4_0];
        dequantize_q4_0_block(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK4_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
static void copy_q4_K_row(float* dst, const block_q4_K* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK_K - 1) / QK_K;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK_K) : QK_K;

        float temp_buffer[QK_K];
        dequantize_q4_K_block(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK_K + i] = temp_buffer[i];
        }
    }
}

static void dequantize_q8_0_block_cache_aligned(const block_q8_0* block, float* dst) {
    const int8_t* qs_ptr = block->qs;

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    const int32_t __attribute__((aligned(32))) vec_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float scale = fp16_to_fp32(block->d);
    __asm__ volatile (
        "fbcx.ps     f0, %0       \n\t" // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t" // Load gether indicies
        :: "r"(scale), "r"(vec_indices)
        : "f0", "f1"
    );

    for (int i = 0; i < 4; i++) {
        __asm__ volatile (
            "fgb.ps      f2, f1(%0)   \n\t" // Loads 8 bytes from (qs_ptr + indices) and sign-extends to 32-bit int.
            "fcvt.ps.pw  f2, f2, rne  \n\t" // Convert Int32 to Float32
            "fmul.ps     f2, f2, f0   \n\t" // f2 = f2 * f0 (scale)
            "fsq2        f2, 0(%1)    \n\t" // Store 256 bits (8 floats) to dst.

            :: "r"(qs_ptr), "r"(dst)
            : "f2", "memory"
        );

        // Advance pointers in C
        qs_ptr += 8;
        dst += 8;
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
static void copy_q4_0_row_cache_aligned(float* dst, const block_q4_0* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    const int32_t __attribute__((aligned(32))) scatter_offsets[8] = {
        0*4, 16*4, 1*4, 17*4, 2*4, 18*4, 3*4, 19*4
    };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    const int32_t __attribute__((aligned(32))) gather_indices[8] = {0, 0, 1, 1, 2, 2, 3, 3};

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile ("mov.m.x m0, x0, 0xFF");          // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ volatile (
        "flq2        f4, 0(%0)    \n\t" // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t" // f1 = gather indices {0,0,1,1,2,2,3,3}
        :: "r"(scatter_offsets), "r"(gather_indices)
        : "f1", "f4"
    );

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_0* block = &src_blocks[block_idx];
        const uint8_t* qs = block->qs;
        float* block_dst = dst + block_idx * QK4_0;

        float scale = fp16_to_fp32(block->d);
        float bias = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ volatile (
            "fbcx.ps     f0, %0       \n\t" // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t" // f3 = broadcast(-8 * scale)
            :: "r"(scale), "r"(bias)
            : "f0", "f3"
        );

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        for (int i = 0; i < 4; i++) {
            __asm__ volatile (
                "fgb.ps      f2, f1(%0)    \n\t" // Gather: [b0,b0,b1,b1,b2,b2,b3,b3]
                "mov.m.x     m0, x0, 0xAA  \n\t" // Odd lanes only (fills gather latency)
                "fsrli.pi    f2, f2, 4     \n\t" // Odd lanes: byte >> 4 (high nibble)
                "mov.m.x     m0, x0, 0xFF  \n\t" // Restore full mask
                "fslli.pi    f2, f2, 28    \n\t" // Isolate low 4 bits: shift left 28
                "fsrli.pi    f2, f2, 28    \n\t" //   then right 28 -> nibble in [3:0]
                "fcvt.ps.pw  f2, f2, rne   \n\t" // Int32 -> Float32
                "fmul.ps     f2, f2, f0    \n\t" // * scale
                "fadd.ps     f2, f2, f3    \n\t" // + bias -> (nibble - 8) * scale
                "fscw.ps     f2, f4(%1)    \n\t" // Scatter to GGML positions

                :: "r"(qs), "r"(block_dst)
                : "f2", "memory"
            );

            qs += 4;         // 4 packed bytes consumed
            block_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
static void copy_q8_0_row_cache_aligned(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    // Number of Q8_0 blocks needed for this row
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        float temp_buffer[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}


// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
static void copy_q4_K_row_cache_aligned(float* dst, const block_q4_K* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    const int32_t __attribute__((aligned(32))) gather_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");            // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ volatile (
        "flq2        f1, 0(%0)    \n\t" // f1 = gather indices {0,1,2,3,4,5,6,7}
        :: "r"(gather_indices)
        : "f1"
    );

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_K* block = &src_blocks[block_idx];
        const uint8_t* qs = block->qs;
        float* block_dst = dst + block_idx * QK_K;

        const float d   = fp16_to_fp32(block->d);
        const float min = fp16_to_fp32(block->dmin);

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            const float d1 = d * sc;
            const float neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            const float d2 = d * sc;
            const float neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ volatile (
                "fbcx.ps     f0, %0       \n\t" // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t" // f3 = broadcast(-m1)
                :: "r"(d1), "r"(neg_m1)
                : "f0", "f3"
            );

            const uint8_t* qs_lo = qs;
            float* dst_lo = block_dst + j;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile (
                    "fgb.ps      f2, f1(%0)   \n\t" // Gather 8 bytes, sign-extend to int32
                    "fandi.pi    f2, f2, 0xF   \n\t" // Mask low nibble (imm10=15)
                    "fcvt.ps.pw  f2, f2, rne   \n\t" // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t" // d1 * nibble + (-m1)
                    "fsq2        f2, 0(%1)     \n\t" // Store 8 floats
                    :: "r"(qs_lo), "r"(dst_lo)
                    : "f2", "memory"
                );
                qs_lo += 8;
                dst_lo += 8;
            }

            // High nibbles: 32 elements using d2, neg_m2
            __asm__ volatile (
                "fbcx.ps     f0, %0       \n\t" // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t" // f3 = broadcast(-m2)
                :: "r"(d2), "r"(neg_m2)
                : "f0", "f3"
            );

            const uint8_t* qs_hi = qs;
            float* dst_hi = block_dst + j + 32;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile (
                    "fgb.ps      f2, f1(%0)   \n\t" // Gather 8 bytes, sign-extend to int32
                    "fsrli.pi    f2, f2, 4     \n\t" // Shift right 4: high nibble
                    "fandi.pi    f2, f2, 0xF   \n\t" // Mask to 4 bits (clean any sign-ext artifacts)
                    "fcvt.ps.pw  f2, f2, rne   \n\t" // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t" // d2 * nibble + (-m2)
                    "fsq2        f2, 0(%1)     \n\t" // Store 8 floats
                    :: "r"(qs_hi), "r"(dst_hi)
                    : "f2", "memory"
                );
                qs_hi += 8;
                dst_hi += 8;
            }

            qs += 32; // Advance to next 32 packed bytes
            is += 2;
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
static int64_t get_elements_per_work_unit(int type) {
    const int64_t elements_per_cacheline = CACHE_LINE_SIZE_BYTES / sizeof(float); // 16
    switch (type) {
        case GGML_TYPE_Q8_0: return QK8_0;  // 32 elements = 2 cachelines
        case GGML_TYPE_Q4_0: return QK4_0;  // 32 elements = 2 cachelines
        case GGML_TYPE_Q4_K: return QK_K;   // 256 elements = 16 cachelines
        default:             return elements_per_cacheline; // 16 elements = 1 cacheline
    }
}
static int get_row_f32_mc_cacheline_aligned(struct ggml_et_get_rows_params* params, void* env)
{
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    struct ggml_tensor* src0 = &params->src0;  // Data tensor
    struct ggml_tensor* src1 = &params->src1;  // Row indices tensor (I32)
    struct ggml_tensor* dst = &params->dst;    // Output tensor (F32)

    const int64_t ne00 = src0->ne[0];  // Source columns (row width)
    const int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    const int64_t ne02 = src0->ne[2];  // Source batch dimension
    const int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    const int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    const int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    const int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    const int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    const int64_t elements_per_wu = get_elements_per_work_unit(src0->type);
    const int64_t wus_per_row = ne00 / elements_per_wu;
    const int64_t total_wus = total_rows_to_extract * wus_per_row;

    // Distribute work units across threads (contiguous ranges)
    const int64_t wus_per_thread = (total_wus + num_threads - 1) / num_threads;
    const int64_t wu_start = thread_id * wus_per_thread;
    int64_t wu_end = wu_start + wus_per_thread;
    if (wu_end > total_wus) wu_end = total_wus;

    void* src0_data = src0->data;
    int32_t* src1_data = (int32_t*)src1->data;
    float* dst_data = (float*)dst->data;

    int64_t wu = wu_start;
    while (wu < wu_end) {
        // Determine which row this work unit belongs to and offset within row
        const int64_t row_idx = wu / wus_per_row;
        const int64_t wu_in_row = wu % wus_per_row;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_remaining_in_row = wus_per_row - wu_in_row;
        int64_t wus_to_process = wu_end - wu;
        if (wus_remaining_in_row < wus_to_process) wus_to_process = wus_remaining_in_row;

        // Calculate multi-dimensional index for this row
        const int64_t i = row_idx;
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                    i12_idx * ne11 * ne10 +
                                    i11_idx * ne10 +
                                    i10_idx;
        const int32_t row_index = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1; // Index out of bounds
        }

        const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                     i12_idx * ne02 * ne01 * ne00 +
                                     i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t elem_offset_in_row = wu_in_row * elements_per_wu;
        const int64_t num_elements = wus_to_process * elements_per_wu;

        float* dst_row = dst_data + row_idx * ne00 + elem_offset_in_row;

        if (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset + elem_offset_in_row;
            copy_row_cache_align(dst_row, src_row, num_elements * sizeof(float));
        }
        else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const int64_t block_start = elem_offset_in_row / QK8_0;
            const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset + block_start;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, num_elements);
        }
        else if (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            const int64_t blocks_per_row = (ne00 + QK4_0 - 1) / QK4_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const int64_t block_start = elem_offset_in_row / QK4_0;
            const block_q4_0* src_blocks = (const block_q4_0*)src0_data + src_block_offset + block_start;
            copy_q4_0_row_cache_aligned(dst_row, src_blocks, num_elements);
        }
        else if (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            const int64_t blocks_per_row = (ne00 + QK_K - 1) / QK_K;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const int64_t block_start = elem_offset_in_row / QK_K;
            const block_q4_K* src_blocks = (const block_q4_K*)src0_data + src_block_offset + block_start;
            copy_q4_K_row_cache_aligned(dst_row, src_blocks, num_elements);
        }

        wu += wus_to_process;
    }

    return 0;
}

// ========================================================================
// OP: SET_ROWS  —  inverse of GET_ROWS (F32 src -> F32/F16 dst)
// ========================================================================

#define CACHE_LINE_SIZE_BYTES  64
#define CACHE_LINE_F32_ELEMS  16   // 64 / 4
#define CACHE_LINE_F16_ELEMS  32   // 64 / 2


// Copy exactly one cache line (64 bytes = 16 F32 elements) using wide loads/stores
static void copy_cache_aligned_f32(float* dst, const float* src) {
    __asm__ volatile (
        "flq2 f0, 0(%[src]) \n\t"    // Load 32 bytes
        "flq2 f1, 32(%[src]) \n\t"   // Load next 32 bytes
        "fsq2 f0, 0(%[dst]) \n\t"    // Store 32 bytes
        "fsq2 f1, 32(%[dst]) \n\t"   // Store next 32 bytes
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory"
    );
}

// Convert and copy one dst cache line worth of F32->F16 (32 elements src -> 64 bytes dst)
static void copy_cache_aligned_f16(uint16_t* dst, const float* src) {
    unsigned long mask_temp;

    // Build offset vector for consecutive 16-bit stores: [0, 2, 4, 6, 8, 10, 12, 14]
    float offset_vec_storage[8];
    uint32_t* offsets = (uint32_t*)offset_vec_storage;
    for (int j = 0; j < 8; j++) {
        offsets[j] = j * 2;
    }

    __asm__ volatile (
        "mova.x.m  %[mask_temp]         \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "flw.ps    f1, 0(%[offsets])    \n\t"
        : [mask_temp] "=&r"(mask_temp)
        : [offsets] "r"(offset_vec_storage)
        : "f1"
    );

    // 4 iterations of 8 elements = 32 F16 elements = 64 bytes = 1 cache line
    for (int i = 0; i < 32; i += 8) {
        __asm__ volatile (
            "flw.ps    f2, 0(%[src_ptr])    \n\t"
            "fcvt.f16.ps f3, f2             \n\t"
            "fsch.ps   f3, f1(%[dst_ptr])   \n\t"
            :
            : [src_ptr] "r"(src + i), [dst_ptr] "r"(dst + i)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile (
        "mova.m.x  %[mask_temp]         \n\t"
        :
        : [mask_temp] "r"(mask_temp)
    );
}


// ========================================================================
// OP: CONT  —  make contiguous (F32 and F16)
// ========================================================================

// Vectorized copy with scalar tail
static inline void vec_copy_f32(float* dst, const float* src, int32_t n) {
    int32_t i = 0;
    const int32_t vec_end = (n / 8) * 8;
    for (; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(float(*)[8])&dst[i])
            : [s] "m"(*(const float(*)[8])&src[i])
            : "f10"
        );
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
static inline void scalar_copy_f32(float* dst, const float* src, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}


// ========================================================================
// OP: CPY  —  copy with type conversion (F32->F16, F32->F32)
// ========================================================================
static void ggml_et_op_cpy(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const char * src_data = (const char *)(uintptr_t)m->src0.data;
    void * dst_data = (void *)(uintptr_t)m->dst.data;
    if (!src_data || !dst_data) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne03 = m->src0.ne[3];
    const size_t nb00 = (size_t)m->src0.nb[0];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];

    const int src_type = m->src0.type;
    const int dst_type = m->dst.type;
    const int64_t total_elems = ne00 * ne01 * ne02 * ne03;
    const int64_t elems_per_cl = 16;
    const int64_t total_cl = (total_elems + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_pt = (total_cl + nth - 1) / nth;
    const int64_t cl_s = tid * cl_pt;
    int64_t cl_e = cl_s + cl_pt;
    if (cl_e > total_cl) cl_e = total_cl;
    if (cl_s >= total_cl) return;
    const int64_t es = cl_s * elems_per_cl;
    int64_t ee = cl_e * elems_per_cl;
    if (ee > total_elems) ee = total_elems;

    if (src_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        uint16_t * dp = (uint16_t *)dst_data;
        const bool src_contig = (nb00 == 4 && nb01 == ne00*4 && nb02 == ne00*ne01*4 && nb03 == ne00*ne01*ne02*4);
        if (src_contig) {
            const float * sp = (const float *)src_data;
            for (int64_t i = es; i < ee; i++) dp[i] = fp32_to_fp16(sp[i]);
        } else {
            for (int64_t idx = es; idx < ee; idx++) {
                int64_t i00 = idx % ne00;
                int64_t r1 = idx / ne00;
                int64_t i01 = r1 % ne01;
                int64_t r2 = r1 / ne01;
                int64_t i02 = r2 % ne02;
                int64_t i03 = r2 / ne02;
                float val = *(const float *)(src_data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                dp[idx] = fp32_to_fp16(val);
            }
        }
    } else {
        // F32->F32 (or same-type copy via CONT)
        // ggml_et_op_cont(env, m);
    }
}

// ========================================================================
// OP: CONCAT
// ========================================================================
static void ggml_et_op_concat(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * s0 = (const float *)(uintptr_t)m->src0.data;
    const float * s1 = (const float *)(uintptr_t)m->src1.data;
    float * dst = (float *)(uintptr_t)m->dst.data;
    if (!s0 || !s1 || !dst) return;

    int32_t dim;
    memcpy(&dim, &m->op_params[0], sizeof(int32_t));

    const int64_t ne00 = m->src0.ne[0], ne01 = m->src0.ne[1], ne02 = m->src0.ne[2];
    const int64_t ne10 = m->src1.ne[0], ne11 = m->src1.ne[1], ne12 = m->src1.ne[2];
    const int64_t ne0 = m->dst.ne[0], ne1 = m->dst.ne[1], ne2 = m->dst.ne[2], ne3 = m->dst.ne[3];

    const size_t nb01 = (size_t)m->src0.nb[1], nb02 = (size_t)m->src0.nb[2], nb03 = (size_t)m->src0.nb[3];
    const size_t nb11 = (size_t)m->src1.nb[1], nb12 = (size_t)m->src1.nb[2], nb13 = (size_t)m->src1.nb[3];
    const size_t dnb1 = (size_t)m->dst.nb[1], dnb2 = (size_t)m->dst.nb[2], dnb3 = (size_t)m->dst.nb[3];

    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);
        float * dr = (float *)((char *)dst + i1*dnb1 + i2*dnb2 + i3*dnb3);

        if (dim == 0) {
            const float * r0 = (const float *)((const char *)s0 + i1*nb01 + i2*nb02 + i3*nb03);
            const float * r1 = (const float *)((const char *)s1 + i1*nb11 + i2*nb12 + i3*nb13);
            for (int64_t k = 0; k < ne00; k++) dr[k] = r0[k];
            for (int64_t k = 0; k < ne10; k++) dr[ne00+k] = r1[k];
        } else if (dim == 1) {
            const float * rp = (i1 < ne01) ?
                (const float *)((const char *)s0 + i1*nb01 + i2*nb02 + i3*nb03) :
                (const float *)((const char *)s1 + (i1-ne01)*nb11 + i2*nb12 + i3*nb13);
            for (int64_t k = 0; k < ne0; k++) dr[k] = rp[k];
        } else if (dim == 2) {
            const float * rp = (i2 < ne02) ?
                (const float *)((const char *)s0 + i1*nb01 + i2*nb02 + i3*nb03) :
                (const float *)((const char *)s1 + i1*nb11 + (i2-ne02)*nb12 + i3*nb13);
            for (int64_t k = 0; k < ne0; k++) dr[k] = rp[k];
        } else {
            const float * rp = (i3 < (int64_t)m->src0.ne[3]) ?
                (const float *)((const char *)s0 + i1*nb01 + i2*nb02 + i3*nb03) :
                (const float *)((const char *)s1 + i1*nb11 + i2*nb12 + (i3-(int64_t)m->src0.ne[3])*nb13);
            for (int64_t k = 0; k < ne0; k++) dr[k] = rp[k];
        }
    }
}

// ========================================================================
// OP: REPEAT  —  tile src into dst
// ========================================================================
static void ggml_et_op_repeat(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t ne00 = m->src0.ne[0], ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne0 = m->dst.ne[0], ne1 = m->dst.ne[1], ne2 = m->dst.ne[2], ne3 = m->dst.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1], nb02 = (size_t)m->src0.nb[2], nb03 = (size_t)m->src0.nb[3];
    const size_t dnb1 = (size_t)m->dst.nb[1], dnb2 = (size_t)m->dst.nb[2], dnb3 = (size_t)m->dst.nb[3];
    const int32_t nr0 = (int32_t)(ne0 / ne00);

    const int64_t total_rows = ne1 * ne2 * ne3;
    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);
        int64_t k1 = i1 % ne01, k2 = i2 % ne02, k3 = i3 % (int64_t)m->src0.ne[3];

        const float * sr = (const float *)((const char *)src + k1*nb01 + k2*nb02 + k3*nb03);
        float * dr = (float *)((char *)dst + i1*dnb1 + i2*dnb2 + i3*dnb3);

        for (int32_t r = 0; r < nr0; r++) {
            for (int64_t k = 0; k < ne00; k++) dr[r * ne00 + k] = sr[k];
        }
    }
}

// ========================================================================
// OP: PAD  —  zero-pad dimensions 1-3
// ========================================================================
static void ggml_et_op_pad(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src = (const float *)(uintptr_t)m->src0.data;
    float * dst = (float *)(uintptr_t)m->dst.data;
    if (!src || !dst) return;

    const int64_t ne0 = m->dst.ne[0], ne1 = m->dst.ne[1];
    const int64_t ne2 = m->dst.ne[2], ne3 = m->dst.ne[3];
    const size_t nb1_src = (size_t)m->src0.nb[1];
    const size_t nb2_src = (size_t)m->src0.nb[2];
    const size_t nb3_src = (size_t)m->src0.nb[3];

    int32_t lp[4], rp[4];
    memcpy(lp, &m->op_params[0], 4 * sizeof(int32_t));
    memcpy(rp, &m->op_params[4], 4 * sizeof(int32_t));

    const int64_t total_rows = ne1 * ne2 * ne3;

    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t i3 = row / (ne1 * ne2);
        int64_t i2 = (row / ne1) % ne2;
        int64_t i1 = row % ne1;
        float * dr = dst + row * ne0;

        if (i1 >= lp[1] && i1 < ne1 - rp[1] &&
            i2 >= lp[2] && i2 < ne2 - rp[2] &&
            i3 >= lp[3] && i3 < ne3 - rp[3]) {
            const float * sr = (const float *)((const char *)src +
                (i1-lp[1])*nb1_src + (i2-lp[2])*nb2_src + (i3-lp[3])*nb3_src);
            for (int64_t k = 0; k < ne0; k++) dr[k] = sr[k];
        } else {
            for (int64_t k = 0; k < ne0; k++) dr[k] = 0.0f;
        }
    }
}

// ========================================================================
// OP: SET  —  inplace write of F32 view into contiguous dst
// ========================================================================
static void ggml_et_op_set(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src1 = (const float *)(uintptr_t)m->src1.data;
    float * dst = (float *)(uintptr_t)m->dst.data;
    if (!src1 || !dst) return;

    const int64_t ne10 = m->src1.ne[0];
    const int64_t ne11 = m->src1.ne[1];
    const int64_t ne12 = m->src1.ne[2];
    const int64_t ne13 = m->src1.ne[3];
    const size_t nb11 = (size_t)m->src1.nb[1];
    const size_t nb12 = (size_t)m->src1.nb[2];
    const size_t nb13 = (size_t)m->src1.nb[3];

    // op_params encode destination view strides and offset
    int32_t dnb1, dnb2, dnb3, offset;
    memcpy(&dnb1,   &m->op_params[0], sizeof(int32_t));
    memcpy(&dnb2,   &m->op_params[1], sizeof(int32_t));
    memcpy(&dnb3,   &m->op_params[2], sizeof(int32_t));
    memcpy(&offset, &m->op_params[3], sizeof(int32_t));

    const int64_t total_rows = ne11 * ne12 * ne13;
    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t i1 = row % ne11;
        int64_t i2 = (row / ne11) % ne12;
        int64_t i3 = row / (ne11 * ne12);

        const float * sr = (const float *)((const char *)src1 + i1*nb11 + i2*nb12 + i3*nb13);
        float * dr = (float *)((char *)dst + offset + i1*dnb1 + i2*dnb2 + i3*dnb3);

        for (int64_t k = 0; k < ne10; k++) dr[k] = sr[k];
    }
}

// ========================================================================
// OP: FLASH_ATTN_EXT  —  scalar F32 flash attention (GQA + masking)
// ========================================================================

// #define FA_DV_MAX_F32 128
// Maximum head dimension supported (128 covers all common LLMs).
#define FA_DV_MAX 128

// Read element d from a row, handling F16 or F32 type.
// row_base points to the start of the row (byte address).
// nb0 is the stride per element (2 for F16, 4 for F32).
static inline float read_kv_f32(const char * row_base, int64_t d,
                                int64_t nb0, int type) {
    if (type == GGML_TYPE_F32) {
        return *(const float *)(row_base + d * nb0);
    }
    // F16
    return fp16_to_fp32(*(const uint16_t *)(row_base + d * nb0));
}

// Dot product of F32 query vector with a K row (F16 or F32).
static inline float dot_qk(const float * q, const char * k_row,
                            int64_t dk, int64_t k_nb0, int k_type) {
    float acc = 0.0f;
    if (k_type == GGML_TYPE_F32) {
        const float * kf = (const float *) k_row;
        for (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * kf[i];
        }
    } else {
        // F16 stride-aware read
        for (int64_t i = 0; i < dk; ++i) {
            acc += q[i] * fp16_to_fp32(*(const uint16_t *)(k_row + i * k_nb0));
        }
    }
    return acc;
}

static inline float get_mask_val(const struct ggml_tensor * mask,
                                 int64_t iq1, int64_t ik1,
                                 int64_t iq2, int64_t iq3) {
    // mask layout: [nk, nq, ne2, ne3] -> broadcast via modulo
    const char * base = (const char *) mask->data
        + iq1 * mask->nb[1]
        + (iq2 % mask->ne[2]) * mask->nb[2]
        + (iq3 % mask->ne[3]) * mask->nb[3];

    if (mask->type == GGML_TYPE_F32) {
        return *(const float *)(base + ik1 * mask->nb[0]);
    }
    // F16
    return fp16_to_fp32(*(const uint16_t *)(base + ik1 * mask->nb[0]));
}



// ========================================================================
// OP: FLASH_ATTN_EXT  —  scalar F16 flash attention 
// ========================================================================

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#define TILE_KV 16
#define TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#define A_L1_START 0
#define B_L1_START 16

// Max head dimensions
#define FA_DV_MAX_F16 256   // max value head dim (dv)
#define FA_DK_MAX_F16 256   // max key head dim (dk) - some models use hsk > hsv

// Per-minion accumulator stride in L2 SCP (1024 bytes for dv=256 F32).
#define L2SCP_ACC_STRIDE  (FA_DV_MAX_F16 * sizeof(float))

typedef uint16_t et_fp16_t;

#define ET_NEG_INF_F (-3.402823466e+38f)

struct ggml_et_flash_attn_ext_params {
    struct ggml_tensor src0;     // Q (F32)
    struct ggml_tensor src1;     // K (F16)
    struct ggml_tensor src2;     // V (F16)
    struct ggml_tensor mask;     // mask (F16 or F32), zeroed when absent
    struct ggml_tensor dst;      // Output (F32)
    float scale;
    int32_t has_mask;
};

static inline const char * get_mask_row_base(const struct ggml_tensor * mask,
                                             int64_t iq1, int64_t iq2, int64_t iq3) {
    return (const char *) mask->data
        + iq1 * mask->nb[1]
        + (iq2 % mask->ne[2]) * mask->nb[2]
        + (iq3 % mask->ne[3]) * mask->nb[3];
}

static inline float get_mask_val_from_base(const struct ggml_tensor * mask,
                                           const char * base, int64_t ik1) {
    if (mask->type == GGML_TYPE_F32) {
        return *(const float *)(base + ik1 * mask->nb[0]);
    }
    return fp16_to_fp32(*(const uint16_t *)(base + ik1 * mask->nb[0]));
}

// Build B panel input for TensorLoadTranspose16.
//
// TensorLoadTranspose16 does: L1Scp[c].h[i] = input[i].h[c]
// We want the interleaved B: L1Scp[l].h[n*2+r] = K[n][dk_start + 2*l + r]
// So we need: input[n*2+r].h[l] = K[n][dk_start + 2*l + r]
//
// For each KV position n, produce two rows (de-interleave even/odd dk elements):
//   Row 2n:   K[n][dk+0], K[n][dk+2], ..., K[n][dk+30]  (16 evens)
//   Row 2n+1: K[n][dk+1], K[n][dk+3], ..., K[n][dk+31]  (16 odds)
//
// Output buffer: 32 rows × 32 halfwords (64-byte stride, 16 hw data + 16 hw pad)
//

// Prefetch KV rows for one chunk into L2 using the platform l2_prefetch primitive.
static inline void __attribute__((always_inline))
prefetch_kv_to_l2(const char * head, int64_t kv_start, int64_t d_start,
                  int64_t kv_count, int64_t nb1)
{
    const void *base = (const void *)(head + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t)kv_count, (uint64_t)nb1);
}

static inline void __attribute__((always_inline))
pack_k_for_transpose16(et_fp16_t * out,
                       const char * k_base,
                       int64_t kv_start,
                       int64_t dk_start,
                       int64_t kv_count,
                       int64_t nb1_k)
{
    // save registers we use becayse TensorFMA uses all FP registers
    // and this function is used in the middle TensorFMAs
    uint32_t save_f28[8] __attribute__((aligned(32)));
    uint32_t save_f29[8] __attribute__((aligned(32)));
    uint32_t save_f30[8] __attribute__((aligned(32)));
    uint32_t save_f31[8] __attribute__((aligned(32)));
    unsigned long old_mask;

    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "fsw.ps    f28, 0(%[save28])\n\t"
        "fsw.ps    f29, 0(%[save29])\n\t"
        "fsw.ps    f30, 0(%[save30])\n\t"
        "fsw.ps    f31, 0(%[save31])\n\t"
        : [ms] "=&r"(old_mask)
        : [save28] "r"(save_f28),
          [save29] "r"(save_f29),
          [save30] "r"(save_f30),
          [save31] "r"(save_f31)
        : "f28", "f29", "f30", "f31", "memory"
    );

    for (int j = 0; j < (int)kv_count; ++j) {
        const et_fp16_t * k_row =
            (const et_fp16_t *)(k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t * even_row = out + (j * 2)     * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        {
            __asm__ volatile(
                "flw.ps    f30, 0(%[src0])  \n\t"
                "flw.ps    f31, 0(%[src1])  \n\t"
                "fpackreph.pi f28, f30      \n\t"
                "fsrli.pi  f29, f30, 16     \n\t"
                "fpackreph.pi f29, f29      \n\t"
                "fpackreph.pi f30, f31      \n\t"
                "fsrli.pi  f31, f31, 16     \n\t"
                "fpackreph.pi f31, f31      \n\t"
                "mov.m.x   m0, x0, 0x0F     \n\t"
                "fcmovm.ps f28, f28, f30    \n\t"
                "fcmovm.ps f29, f29, f31    \n\t"
                "mov.m.x   m0, x0, 0xFF     \n\t"
                "fsw.ps    f28, 0(%[even])  \n\t"
                "fsw.ps    f29, 0(%[odd])   \n\t"
                :
                : [src0] "r"(k_row),
                  [src1] "r"(k_row + 16),
                  [even] "r"(even_row),
                  [odd] "r"(odd_row)
                : "f28", "f29", "f30", "f31", "memory"
            );
        }
    }

    __asm__ volatile(
        "flw.ps    f28, 0(%[save28])\n\t"
        "flw.ps    f29, 0(%[save29])\n\t"
        "flw.ps    f30, 0(%[save30])\n\t"
        "flw.ps    f31, 0(%[save31])\n\t"
        "mova.m.x  %[ms]            \n\t"
        :
        : [ms] "r"(old_mask),
          [save28] "r"(save_f28),
          [save29] "r"(save_f29),
          [save30] "r"(save_f30),
          [save31] "r"(save_f31)
        : "f28", "f29", "f30", "f31", "memory"
    );

    for (int j = (int)kv_count; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2)     * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        for (int l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
//
// K dimension  = kv_count KV positions (up to TILE_KV = 16)
// N dimension  = 16 dv values per chunk
// Output       = 8 SCP lines × 32 halfwords (512 bytes, 64-byte stride)
//
//   out[l*32 + n*2 + r] = V[kv_base + 2*l + r][dv_start + n]
//   l = 0..7 (K/2), n = 0..15 (output cols), r = 0..1 (even/odd K)
//
// Zero-pads KV positions beyond kv_count.
static inline void __attribute__((always_inline))
pack_v_interleaved(et_fp16_t *out,
                   const char *v_head,
                   int64_t kv_base,
                   int64_t dv_start,
                   int64_t kv_count,
                   int64_t nb1_v)
{
    for (int k = 0; k < TILE_KV; ++k) {
        const int l = k >> 1;
        const int r = k & 1;
        et_fp16_t * const dst = out + l * 32 + r;
        if (k < (int)kv_count) {
            const et_fp16_t *v_row =
                (const et_fp16_t *)(v_head + (kv_base + k) * nb1_v) + dv_start;
            for (int n = 0; n < 16; ++n)
                dst[n * 2] = v_row[n];
        } else {
            for (int n = 0; n < 16; ++n)
                dst[n * 2] = 0;
        }
    }
}


static inline void __attribute__((always_inline))
convert_q_row_f32_to_f16(et_fp16_t * dst, const float * src, int64_t n) {
    static const int32_t __attribute__((aligned(32))) offsets[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1"
    );

    for (int64_t d = 0; d < n; d += 8) {
        __asm__ volatile(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[ms]             \n\t"
        :
        : [ms] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
accumulate_v_row_f16_contig(float * acc, const char * pv, int64_t dv, float vs) {
    static const int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[gidx])    \n\t"
        : [ms] "=&r"(old_mask)
        : [gidx] "r"(gather_idx)
        : "f1"
    );

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps      f2, 0(%[p_vs])   \n\t"
            "fgh.ps      f3, f1(%[pv])    \n\t"
            "fcvt.ps.f16 f3, f3           \n\t"
            "flw.ps      f4, 0(%[pa])     \n\t"
            "fmadd.ps    f4, f3, f2, f4   \n\t"
            "fsw.ps      f4, 0(%[pa])     \n\t"
            :
            : [p_vs] "r"(&vs), [pv] "r"(pv + d * 2), [pa] "r"(acc + d)
            : "f2", "f3", "f4", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[ms]             \n\t"
        :
        : [ms] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
rescale_accumulate_v_row_f16_contig(float * acc, const char * pv, int64_t dv, float ms) {
    static const int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 2, 4, 6, 8, 10, 12, 14
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[msk]            \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[gidx])    \n\t"
        : [msk] "=&r"(old_mask)
        : [gidx] "r"(gather_idx)
        : "f1"
    );

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps      f2, 0(%[p_ms])   \n\t"
            "fgh.ps      f3, f1(%[pv])    \n\t"
            "fcvt.ps.f16 f3, f3           \n\t"
            "flw.ps      f4, 0(%[pa])     \n\t"
            "fmadd.ps    f4, f4, f2, f3   \n\t"
            "fsw.ps      f4, 0(%[pa])     \n\t"
            :
            : [p_ms] "r"(&ms), [pv] "r"(pv + d * 2), [pa] "r"(acc + d)
            : "f2", "f3", "f4", "memory"
        );
    }

    __asm__ volatile(
        "mova.m.x  %[msk]            \n\t"
        :
        : [msk] "r"(old_mask)
    );
}

static inline void __attribute__((always_inline))
zero_acc_vec(float * acc, int64_t dv) {
    const float zero = 0.0f;
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps  f2, 0(%[z])     \n\t"
            "fsw.ps  f2, 0(%[a])     \n\t"
            :
            : [z] "r"(&zero), [a] "r"(acc + d)
            : "f2", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

static inline void __attribute__((always_inline))
scale_acc_vec(float * acc, int64_t dv, float scale) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

static inline void __attribute__((always_inline))
normalize_store_vec(float * out, float * acc, int64_t dv, float inv, int use_fast_store) {
    unsigned long old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    for (int64_t d = 0; d < dv; d += 8) {
        __asm__ volatile(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory"
        );
        if (use_fast_store) {
            __asm__ volatile(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory"
            );
        } else {
            atomic_store_f32((volatile float *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((volatile float *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((volatile float *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((volatile float *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((volatile float *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((volatile float *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((volatile float *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((volatile float *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

// ========================================================================
// Entry point — graph execution loop
// ========================================================================
int entry_point(struct ggml_cgraph_et * cg, void * env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *)cg->data;
    uint8_t * node_op = (uint8_t *)(node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    for (int i = 0; i < n_nodes; i++) {
        const int op = node_op[i];

        if (op == GGML_OP_NONE) {
            continue;
        }
        // Fusion: RMS_NORM + MUL -> fused RMS_NORM_MUL
        // if (op == GGML_OP_RMS_NORM &&
        //     ggml_et_can_fuse(cg, i, node_op, n_nodes,
        //                      (enum ggml_op[]){ GGML_OP_RMS_NORM, GGML_OP_MUL }, 2)) {
        //     ggml_et_op_rms_norm_mul(env, &node_meta[i], &node_meta[i + 1]);
        //     i++;
        //     continue;
        // }
       
        int thread_id = get_relative_thread_id(kernel_env->shire_mask);
        int num_threads = get_num_threads(kernel_env->shire_mask);
        uint64_t shire_id = get_shire_id();

        void * src0_data = (void *)(uintptr_t)node_meta[i].src0.data;
        void * src1_data = (void *)(uintptr_t)node_meta[i].src1.data;
        void * src2_data = (void *)(uintptr_t)node_meta[i].src2.data;
        void * dst_data  = (void *)(uintptr_t)node_meta[i].dst.data;

        // // Basic null pointer checks
        // if (!src0_data || !dst_data) {
        //     continue;
        // }

        const int64_t ne0 = node_meta[i].dst.ne[0], ne1 = node_meta[i].dst.ne[1];
        const int64_t ne2 = node_meta[i].dst.ne[2], ne3 = node_meta[i].dst.ne[3];
        const int64_t ne00 = node_meta[i].src0.ne[0], ne01 = node_meta[i].src0.ne[1];
        const int64_t ne02 = node_meta[i].src0.ne[2], ne03 = node_meta[i].src0.ne[3];
        const int64_t ne10 = node_meta[i].src1.ne[0], ne11 = node_meta[i].src1.ne[1];
        const int64_t ne12 = node_meta[i].src1.ne[2], ne13 = node_meta[i].src1.ne[3];
        const int64_t ne20 = node_meta[i].src2.ne[0]; // Used in FLASH_ATTN_EXT

        const size_t nb0 = (size_t)node_meta[i].dst.nb[0], nb1 = (size_t)node_meta[i].dst.nb[1];
        const size_t nb2 = (size_t)node_meta[i].dst.nb[2], nb3 = (size_t)node_meta[i].dst.nb[3];
        const size_t nb00 = (size_t)node_meta[i].src0.nb[0], nb01 = (size_t)node_meta[i].src0.nb[1];
        const size_t nb02 = (size_t)node_meta[i].src0.nb[2], nb03 = (size_t)node_meta[i].src0.nb[3];
        const size_t nb10 = (size_t)node_meta[i].src1.nb[0], nb11 = (size_t)node_meta[i].src1.nb[1];
        const size_t nb12 = (size_t)node_meta[i].src1.nb[2], nb13 = (size_t)node_meta[i].src1.nb[3];
        const size_t nb20 = (size_t)node_meta[i].src2.nb[0], nb21 = (size_t)node_meta[i].src2.nb[1];
        const size_t nb22 = (size_t)node_meta[i].src2.nb[2], nb23 = (size_t)node_meta[i].src2.nb[3];
        
        // device_barrier(32);

        if(op == GGML_OP_ADD || op == GGML_OP_MUL || op == GGML_OP_SUB) {
            if (!src0_data || !src1_data || !dst_data) {
                continue;
            } 
            if ((node_meta[i].src0.type != GGML_TYPE_F32) || 
                (node_meta[i].src1.type != GGML_TYPE_F32) ||
                (node_meta[i].dst.type != GGML_TYPE_F32)){
                continue; // Only support F32 for now
            }
            const size_t elem_size = 4; // F32
            const bool cache_aligned = (ne0 % 16 == 0);
            if(!cache_aligned) {
                continue;
            }

            // Fast path: no broadcasting, contiguous
            const bool no_broadcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
            const bool all_contiguous = (nb0 == elem_size && nb00 == elem_size && nb10 == elem_size &&
                                        nb1 == ne0 * elem_size && nb01 == ne0 * elem_size && nb11 == ne0 * elem_size);

            if (no_broadcast && all_contiguous) {
                const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
                const int64_t elements_per_cacheline = 16;  // 64 bytes / element_size
                const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

                const int64_t cl_per_thread = (total_cachelines + num_threads - 1) / num_threads;
                const int64_t cl_start = thread_id * cl_per_thread;
                int64_t cl_end = cl_start + cl_per_thread;
                if (cl_end > total_cachelines) cl_end = total_cachelines;

                if (cl_start >= total_cachelines) {
                    // et_printf("CACHE LINES PASSED\n");
                    continue;
                }

                const int64_t elem_start = cl_start * elements_per_cacheline;
                int64_t elem_end = cl_end * elements_per_cacheline;
                if (elem_end > total_elements) elem_end = total_elements;
                const int32_t count = (int32_t)(elem_end - elem_start);

                switch (op) {
                    case GGML_OP_MUL:
                        block_mul_cache_aligned((float*)dst_data + elem_start, (float*)src0_data + elem_start, (float*)src1_data + elem_start, count);
                        break;
                    case GGML_OP_ADD:
                        block_add_cache_aligned((float*)dst_data + elem_start, (float*)src0_data + elem_start, (float*)src1_data + elem_start, count);
                        break;
                    case GGML_OP_SUB:
                        block_sub_cache_aligned((float*)dst_data + elem_start, (float*)src0_data + elem_start, (float*)src1_data + elem_start, count);
                        break;
                    default:
                        break;
                }
                
            } else {
                // Slow path: broadcasting or non-contiguous: row based or bcast on last row
                const int64_t total_rows = ne1 * ne2 * ne3;

                const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
                const int64_t start_row = thread_id * rows_per_thread;
                const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

                if (start_row >= total_rows) {
                    continue;
                }

                for (int64_t ir = start_row; ir < end_row; ir++) {
                    // Convert flat row index to 3D coordinates
                    const int64_t i03 = ir / (ne2 * ne1);
                    const int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                    const int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                    // Handle broadcasting: src1 coordinates with modulo
                    const int64_t i13 = i03 % ne13;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i11 = i01 % ne11;

                    // Calculate base pointers for this row using stride-based addressing
                    void* dst_ptr = (void*)((char*)dst_data + i03*nb3 + i02*nb2 + i01*nb1);
                    const void* src0_ptr = (const void*)((const char*)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
                    const void* src1_ptr = (const void*)((const char*)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

                    if (ne10 == 1) {
                        // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
                        float scalar = ((const float*)src1_ptr)[0];
                        switch (op) {
                            case GGML_OP_MUL:
                                block_mul_broadcast((float*)dst_ptr, (const float*)src0_ptr, scalar, (int)ne0);
                                break;
                            case GGML_OP_ADD:
                                block_add_broadcast((float*)dst_ptr, (const float*)src0_ptr, scalar, (int)ne0);
                                break;
                            case GGML_OP_SUB:
                                block_sub_broadcast((float*)dst_ptr, (const float*)src0_ptr, scalar, (int)ne0);
                                break;
                            default:
                                break;
                        }
                    } else {
                        // Broadcasting in dimension 0: src1 repeats across src0
                        const int64_t nr0 = ne0 / ne10;

                        for (int64_t r = 0; r < nr0; r++) {
                            const float* src0_block = (const float*)src0_ptr + r * ne10;
                            float* dst_block = (float*)dst_ptr + r * ne10;

                            switch (op) {
                                case GGML_OP_MUL:
                                    block_mul_cache_aligned(dst_block, src0_block, (const float*)src1_ptr, (int)ne10);
                                    break;
                                case GGML_OP_ADD:
                                    block_add_cache_aligned(dst_block, src0_block, (const float*)src1_ptr, (int)ne10);
                                    break;
                                case GGML_OP_SUB:
                                    block_sub_cache_aligned(dst_block, src0_block, (const float*)src1_ptr, (int)ne10);
                                    break;
                                default:
                                    break;
                            }
                        }
                    }
                }
            
            } // end of else block for slow path
        
        
        }else if (op == GGML_OP_GLU) {
            if (!src0_data || !dst_data) continue;
            const bool is_split_mode = node_meta[i].src1.data != 0;
            if ((node_meta[i].src0.type != GGML_TYPE_F32) || 
                ((is_split_mode) && (node_meta[i].src1.type != GGML_TYPE_F32)) ||
                (node_meta[i].dst.type != GGML_TYPE_F32)){
                continue; // Only support F32 for now
            }
            const int32_t glu_op_type;         // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
            const int32_t swapped;             // Whether gate and value are swapped
            // FIXME: can we remove memcpy
            memcpy(&glu_op_type, &node_meta[i].op_params[0], sizeof(int32_t));
            memcpy(&swapped, &node_meta[i].op_params[1], sizeof(int32_t));
    
            // Get tensor dimensions
            const int64_t nc = ne0;  // Output columns (input columns / 2)
            const int64_t nr = ne1 * ne2 * ne3;  // Total rows

            // Get strides
            const size_t src0_stride = nb01;  // Stride between rows in src0
            const size_t src1_stride = is_split_mode ? nb11 : nb01;  // Stride between rows in src1
            const size_t dst_stride = nb1;    // Stride between rows in dst

            // Validate dimensions for split SwiGLU
            if (is_split_mode) {
                // Split tensor mode: src0 and src1 should have same shape as dst
                if (node_meta[i].src0.ne[0] != nc || ne10 != nc) {
                    return -1; // Dimension mismatch in split mode
                }
            } else {
                // Single tensor mode: src0 should have 2*nc columns
                if (node_meta[i].src0.ne[0] != 2 * nc) {
                    return -1; // Dimension mismatch in single tensor mode
                }
            }

            // Calculate total elements for cache line distribution
            const int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
            const int64_t total_elements = nr * nc;
            const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

            // Distribute cache lines across threads
            int64_t cachelines_per_thread = (total_cachelines + num_threads - 1) / num_threads;
            int64_t start_cacheline = thread_id * cachelines_per_thread;
            int64_t end_cacheline = start_cacheline + cachelines_per_thread;

            // Clamp end_cacheline to actual number of cache lines
            if (end_cacheline > total_cachelines) {
                end_cacheline = total_cachelines;
            }

            // Thread should return if no work to do
            if (start_cacheline >= total_cachelines) {
                return 0;
            }

            // Process cache lines assigned to this thread
            for (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
                // Map cache line back to element coordinates
                int64_t global_element_start = cl * elements_per_cacheline;
                int64_t row = global_element_start / nc;
                int64_t col = global_element_start % nc;

                // Skip if we're past the end of data
                if (global_element_start >= total_elements) {
                    break;
                }

                // Calculate how many elements to process in this cache line
                int64_t elements_remaining = total_elements - global_element_start;
                int elements_this_block = (int)((elements_remaining < elements_per_cacheline) ?
                                            elements_remaining : elements_per_cacheline);

                // Process elements that span across rows
                int64_t elements_processed = 0;
                while (elements_processed < elements_this_block && row < nr) {
                    // Calculate elements to process in current row
                    int64_t elements_in_row = nc - col;
                    int64_t elements_to_process = elements_this_block - elements_processed;
                    if (elements_to_process > elements_in_row) {
                        elements_to_process = elements_in_row;
                    }

                    // Get pointers for current row and column range
                    float* dst_ptr = (float*)((char*)dst_data + row * dst_stride) + col;

                    float* x_ptr;
                    float* g_ptr;

                    if (is_split_mode) {
                        // Split tensor mode
                        x_ptr = (float*)((char*)src0_data + row * src0_stride) + col;
                        g_ptr = (float*)((char*)src1_data + row * src1_stride) + col;
                    } else {
                        // Single tensor mode - src0 contains both x and g
                        float* src0_row = (float*)((char*)src0_data + row * src0_stride);
                        if (swapped) {
                            g_ptr = src0_row + col;                // First half is gate
                            x_ptr = src0_row + nc + col;           // Second half is value
                        } else {
                            x_ptr = src0_row + col;                // First half is value
                            g_ptr = src0_row + nc + col;           // Second half is gate
                        }
                    }

                    // Process this segment
                    if (glu_op_type == GGML_GLU_OP_GEGLU) {
                        block_geglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);
                    } else if (glu_op_type == GGML_GLU_OP_SWIGLU) {
                        block_swiglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);
                    } else {
                        break;
                    }

                    // Update counters
                    elements_processed += elements_to_process;
                    col += elements_to_process;

                    // Move to next row if current row is complete
                    if (col >= nc) {
                        row++;
                        col = 0;
                    }
                }
            }
    
        } else if (op == GGML_OP_SOFT_MAX) {
            void * src2_data = (void *)(uintptr_t)node_meta[i].src2.data;
            const float scale;         // Scale factor
            const float max_bias;      // ALiBi max bias
            memcpy((void*)&scale, &node_meta[i].op_params[0], sizeof(float));
            memcpy((void*)&max_bias, &node_meta[i].op_params[1], sizeof(float));
            
            // Validate tensor types (F32 only)
            if((node_meta[i].src0.type != GGML_TYPE_F32) || (node_meta[i].dst.type != GGML_TYPE_F32)){
                continue; // Unsupported type combination
            }

            // Check if mask is used and validate type
            bool use_mask = (node_meta[i].src1.data != NULL && (node_meta[i].src1.type == GGML_TYPE_F32 || node_meta[i].src1.type == GGML_TYPE_F16));

            bool use_sinks = (node_meta[i].src2.data != NULL && node_meta[i].src2.type == GGML_TYPE_F32);

            float* src0_data_f32 = (float*)src0_data;
            float* dst_data_f32 = (float*)dst_data;
            float* mask_data = use_mask ? (float*)src1_data : NULL;
            float* sinks_data = use_sinks ? (float*)src2_data : NULL;

            if (!src0_data_f32 || !dst_data_f32) {
                continue; // Null data pointer
            }

            // Use pre-extracted dimensions (ne0, ne1, ne2, ne3 are dst dimensions)
            const int64_t ne00 = ne0;  // Sequence length (columns) - same as dst ne[0]
            const int64_t ne01 = ne1;  // Number of rows - same as dst ne[1]
            const int64_t ne02 = ne2;  // Batch/head dimension - same as dst ne[2]
            const int64_t ne03 = ne3;  // Outer batch dimension - same as dst ne[3]

            const int64_t mask_ne10 = use_mask ? ne10 : 0;  // Mask sequence length
            const int64_t mask_ne11 = use_mask ? ne11 : 0;  // Mask rows
            const int64_t mask_ne12 = use_mask ? ne12 : 0;  // Mask batch/head dimension
            const int64_t mask_ne13 = use_mask ? ne13 : 0;  // Mask outer batch dimension

            if (use_mask) {
                // - Dimension 0: mask must equal input exactly
                // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
                // - Dimension 2: input must be divisible by mask (modulo broadcasting)
                // - Dimension 3: input must be divisible by mask (modulo broadcasting)
                if (mask_ne10 != ne00 ||                    // Dimension 0: exact match required
                    mask_ne11 < ne01 ||                     // Dimension 1: mask >= input
                    (mask_ne12 > 0 && ne02 % mask_ne12 != 0) ||  // Dimension 2: input % mask == 0
                    (mask_ne13 > 0 && ne03 % mask_ne13 != 0)) {  // Dimension 3: input % mask == 0
                    continue; // Incompatible dimensions for ggml softmax broadcasting
                }
            }

            // ALiBi slope calculation - compute per attention head
            const uint32_t n_head = (uint32_t)ne02;
            uint32_t n_head_log2 = 0;
            float m0 = 1.0f;
            float m1 = 1.0f;

            if (max_bias > 0.0f) {
                // This is equivalent to: 1 << floor(log2(n_head))
                n_head_log2 = 1;
                while (n_head_log2 < n_head) {
                    n_head_log2 <<= 1;
                }
                if (n_head_log2 > n_head) {
                    n_head_log2 >>= 1;
                }

                // Compute base slopes for ALiBi
                // m0 = 2^(-max_bias / n_head_log2)
                // m1 = 2^(-max_bias / (2 * n_head_log2))
                float inv_n_head_log2 = et_fdiv(1.0f, (float)n_head_log2);
                m0 = et_expf(-max_bias * 0.69314718f * inv_n_head_log2);  // 0.69314718 = ln(2)
                m1 = et_expf(-max_bias * 0.69314718f * inv_n_head_log2 * 0.5f);
            }

            // Process tensor row by row in parallel across flattened rows.
            // Flattened row index spans [i03, i02, i01] with row length ne00.
            const int64_t rows_per_i03 = ne02 * ne01;
            const int64_t total_rows = ne03 * rows_per_i03;

            for (int64_t row = thread_id; row < total_rows; row += num_threads) {
                const int64_t i03 = row / rows_per_i03;
                const int64_t rem = row % rows_per_i03;
                const int64_t i02 = rem / ne01;
                const int64_t i01 = rem % ne01;

                // Calculate ALiBi slope for this attention head
                float slope = 1.0f;
                if (max_bias > 0.0f) {
                    const uint32_t h = (uint32_t)i02;  // head index
                    if (h < n_head_log2) {
                        // slope = m0^(h+1) for first half of heads
                        slope = m0;
                        for (uint32_t i = 0; i < h; i++) {
                            slope *= m0;
                        }
                    } else {
                        // slope = m1^(2*(h - n_head_log2) + 1) for second half
                        const uint32_t exp = 2 * (h - n_head_log2) + 1;
                        slope = m1;
                        for (uint32_t i = 1; i < exp; i++) {
                            slope *= m1;
                        }
                    }
                }

                float sink_value = 0.0f;
                if (use_sinks && sinks_data) {
                    // Sinks tensor is 1D array indexed by head (i02)
                    sink_value = sinks_data[i02];
                }

                const int64_t src_offset = i03 * ne02 * ne01 * ne00 +
                                        i02 * ne01 * ne00 +
                                        i01 * ne00;

                const float* src_row = src0_data_f32 + src_offset;
                float* dst_row = dst_data_f32 + src_offset;
                const float* mask_row = NULL;

                // Calculate mask row offset using ggml's broadcasting rules
                if (use_mask && mask_data) {
                    // ggml broadcasting logic:
                    // - i11 = i01 (direct mapping for dimension 1, even if mask is larger)
                    // - i12 = i02 % ne12 (modulo broadcasting for dimension 2)
                    // - i13 = i03 % ne13 (modulo broadcasting for dimension 3)
                    const int64_t mask_i03 = (mask_ne13 > 0) ? i03 % mask_ne13 : 0;
                    const int64_t mask_i02 = (mask_ne12 > 0) ? i02 % mask_ne12 : 0;
                    const int64_t mask_i01 = i01;  // Direct mapping (mask >= input guaranteed)

                    const int64_t mask_offset = mask_i03 * mask_ne12 * mask_ne11 * mask_ne10 +
                                            mask_i02 * mask_ne11 * mask_ne10 +
                                            mask_i01 * mask_ne10;

                    mask_row = mask_data + mask_offset;
                }

                compute_softmax_row(dst_row, src_row, mask_row, (int)ne00, scale, slope, sink_value, use_sinks);
            }
        } else if (op == GGML_OP_FLASH_ATTN_EXT) {
                if (node_meta[i].dst.type != GGML_TYPE_F32 || node_meta[i].src0.type != GGML_TYPE_F32) {
                    continue;
                }
                // K and V can be F16 or F32
                if ((node_meta[i].src1.type != GGML_TYPE_F32 && node_meta[i].src1.type != GGML_TYPE_F16) ||
                    (node_meta[i].src2.type != GGML_TYPE_F32 && node_meta[i].src2.type != GGML_TYPE_F16)) {
                    continue;
                }
                if (node_meta[i].src2.data != 0) {
                    continue;
                }                   
                // Mask is optional; if present must be F16 or F32
                if (node_meta[i].src1.data != 0 &&
                    node_meta[i].src1.type != GGML_TYPE_F32 &&
                    node_meta[i].src1.type != GGML_TYPE_F16) {
                    continue;
                }
                // Q and dst must be row-contiguous F32
                // TODO: Add contiguity checks using pre-extracted strides
                // For now, skip these checks
                // continue; // Skip until contiguity checks are properly implemented

                // K/V must have element-sized stride in dim 0
                const size_t k_elem = node_meta[i].src1.type == GGML_TYPE_F16 ? 2 : 4;
                const size_t v_elem = node_meta[i].src2.type == GGML_TYPE_F16 ? 2 : 4;
                if (nb10 != k_elem || nb12 != v_elem) {
                    continue;
                }
                float scale = 1.0f;
                float max_bias = 0.0f;
                float logit_softcap = 0.0f;
                memcpy(&scale,         &node_meta[i].op_params[0], sizeof(scale));
                memcpy(&max_bias,      &node_meta[i].op_params[1], sizeof(max_bias));
                memcpy(&logit_softcap, &node_meta[i].op_params[2], sizeof(logit_softcap));
                if (max_bias != 0.0f || logit_softcap != 0.0f) {
                    continue;
                }
                // TODO: Add precision check when available in node_meta
                // For now, assume F32 precision
                // dk must match between Q and K; dv must match between V and dst
                if (ne0 != ne10) {
                    continue;
                }
                // TODO: Add dst dimension check when available
                // For now, skip this check
                if (ne0 > 256) {
                    continue;
                }
                // GQA: n_head_q must be a multiple of n_head_kv
                const int64_t nhq = ne2;  // Using pre-extracted dst ne[2] as src0 ne[2]
                const int64_t nhk = ne12; // Using pre-extracted src1 ne[2]
                if (nhq % nhk != 0) {
                    continue;
                }
                // K and V must have matching sequence length, heads, and batch dims
                if (ne11 != ne13 ||  // src1 ne[1] vs src2 ne[1] - using ne11 for src1 ne[1], need src2 ne[1]
                    ne12 != ne12 ||  // src1 ne[2] vs src2 ne[2] - same dimension
                    ne13 != ne13) {  // src1 ne[3] vs src2 ne[3] - same dimension
                    // TODO: Add proper dimension comparison when all src2 dimensions are available
                    continue;
                }
                // dst layout checks: [dv, nhq, nq, no]
                // TODO: Add dst layout checks when all dimensions are properly mapped
                // Batch dims: Q batch must match K batch
                if (ne3 != ne13) {
                    continue;
                }
                
                // Use matrix engine kernel when K/V are F16 and dk is a multiple of 32
                if (node_meta[i].src1.type == GGML_TYPE_F16 &&
                    node_meta[i].src2.type == GGML_TYPE_F16 &&
                    (ne0 % 32) == 0) {
                    // TODO: F16 FLASH attention implementation temporarily disabled
                    // due to tensor engine integration complexity
                    continue;

                } else {
                    // -----------------------------------------------------
                    // FA-F32
                    // -----------------------------------------------------
                    
                    // For debugging: use single thread
                    const int fa_thread_id = 0;
                    const int fa_num_threads = 1;

                    // Use pre-extracted data pointers and metadata
                    const char * q_data   = (const char *)src0_data;
                    const char * k_data   = (const char *)src1_data;
                    const char * v_data   = (const char *)src2_data;
                    char * fa_dst_data    = (char *)dst_data;

                    const int k_type = node_meta[i].src1.type;
                    const int v_type = node_meta[i].src2.type;
                    const int64_t k_nb0 = nb10;  // Pre-extracted K stride
                    const int64_t v_nb0 = nb20;  // Pre-extracted V stride

                    // Use pre-extracted dimensions
                    const int64_t dk  = ne0;      // Q ne[0] = K ne[0]
                    const int64_t nq  = ne1;      // Q ne[1]
                    const int64_t fa_nhq = ne2;   // Q ne[2]
                    const int64_t no  = ne3;      // Q ne[3]
                    const int64_t nk  = ne11;     // K ne[1]
                    const int64_t fa_nhk = ne12;  // K ne[2]
                    const int64_t dv  = ne20;     // V ne[0] (correct value head dimension)

                    if (dv > FA_DV_MAX) {
                        continue;
                    }

                    // GQA: query heads per kv head
                    const int64_t gqa_ratio = fa_nhq / fa_nhk;

                    // Extract scale from op_params
                    const float scale_f32;
                    memcpy((void*)&scale_f32, &node_meta[i].op_params[0], sizeof(float));
                    
                    // For FLASH_ATTN_EXT, mask information is stored in op_params[1] as has_mask flag
                    // The actual mask tensor is not directly accessible in node_meta, so we assume no mask for now
                    bool use_mask = false;  // Simplified - assume no mask until proper mask integration
                    
                    const int64_t total_rows = nq * fa_nhq * no;

                    // When dv is a multiple of 16 (64 bytes = cache line), output rows are
                    // cache-line aligned and we can use fast normal stores. Otherwise we must
                    // use atomic stores to avoid cache-line sharing corruption.
                    const int use_fast_store = (dv % 16 == 0);

                    for (int64_t row = fa_thread_id; row < total_rows; row += fa_num_threads) {
                        const int64_t iq3 = row / (fa_nhq * nq);
                        const int64_t rem = row % (fa_nhq * nq);
                        const int64_t iq2 = rem / nq;           // query head index
                        const int64_t iq1 = rem % nq;           // query position

                        // Map query head -> kv head for GQA
                        const int64_t ik2 = iq2 / gqa_ratio;

                        // Q is always F32
                        const float * pq = (const float *) (q_data + iq1*nb01 + iq2*nb02 + iq3*nb03);

                        // dst layout: [dv, nhq, nq, no]
                        float * out = (float *) (fa_dst_data + iq2*nb1 + iq1*nb2 + iq3*nb3);

                        // Base byte offsets for K and V head+batch slice
                        const int64_t kv_base = ik2*nb12 + iq3*nb13;
                        const int64_t vv_base = ik2*nb22 + iq3*nb23;

                            float acc[FA_DV_MAX];
                            for (int64_t d = 0; d < dv; ++d) {
                                acc[d] = 0.0f;
                            }

                            float M = -3.402823466e+38f;
                            float S = 0.0f;

                            for (int64_t ik1 = 0; ik1 < nk; ++ik1) {

                                // Skip mask processing for now - assume no mask
                                const char * pk = k_data + ik1*nb11 + kv_base;
                                const char * pv = v_data + ik1*nb21 + vv_base;

                                float s = dot_qk(pq, pk, dk, k_nb0, k_type) * scale_f32;
                                const float Mold = M;

                                float ms = 1.0f;
                                float vs = 1.0f;
                                if (s > M) {
                                    M = s;
                                    ms = et_expf(Mold - M);
                                    for (int64_t d = 0; d < dv; ++d) {
                                        acc[d] *= ms;
                                    }
                                } else {
                                    vs = et_expf(s - M);
                                }

                                // Accumulate weighted V
                                if (v_type == GGML_TYPE_F32) {
                                    const float * pvf = (const float *) pv;
                                    for (int64_t d = 0; d < dv; ++d) {
                                        acc[d] += pvf[d] * vs;
                                    }
                                } else {
                                    for (int64_t d = 0; d < dv; ++d) {
                                        acc[d] += fp16_to_fp32(*(const uint16_t *)(pv + d * v_nb0)) * vs;
                                    }
                                }

                                S = S * ms + vs;
                            }

                            const float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
                            if (use_fast_store) {
                                for (int64_t d = 0; d < dv; ++d) {
                                    out[d] = acc[d] * S_inv;
                                }
                            } else {
                                for (int64_t d = 0; d < dv; ++d) {
                                    atomic_store_f32((volatile float *) &out[d], acc[d] * S_inv);
                                }
                            }
                        }
                    }
            // flash_attn_ext(env, &node_meta[i]);
        } else if (op == GGML_OP_GET_ROWS) {
            
            // Basic null pointer checks
            if (!src0_data || !dst_data) {
                continue;
            }

            // Basic type checks
            if((node_meta[i].src0.type == GGML_TYPE_F32 || node_meta[i].src0.type == GGML_TYPE_Q8_0 || node_meta[i].src0.type == GGML_TYPE_Q4_0 || node_meta[i].src0.type == GGML_TYPE_Q4_K) && node_meta[i].src1.type == GGML_TYPE_I32 && node_meta[i].dst.type == GGML_TYPE_F32
                && node_meta[i].dst.ne[0] % CACHE_ELEMENTS(sizeof(float)) == 0) {
                struct ggml_et_get_rows_params params;
                convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_GET_ROWS);
                get_row_f32_mc_cacheline_aligned(&params, env);
                continue;
            }
            
            const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

            // Naive single-threaded implementation - process all rows sequentially
            // Only thread 0 should execute this to avoid race conditions
            if (thread_id != 0) {
                continue;
            }

            // Cache src0 type before inner loop to avoid shadowed-variable bug
            const int src0_type = node_meta[i].src0.type;

            // XXX: Do we really need a single-threaded implementation?
            for (int64_t ri = 0; ri < total_rows_to_extract; ri++) {
                // Calculate multi-dimensional index for the current output position
                const int64_t i13_idx = ri / (ne12 * ne11 * ne10);
                const int64_t i12_idx = (ri - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
                const int64_t i11_idx = (ri - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
                const int64_t i10_idx = ri - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

                // Get the row index from src1
                const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                            i12_idx * ne11 * ne10 +
                                            i11_idx * ne10 +
                                            i10_idx;
                const int32_t row_index = ((const int32_t*)src1_data)[index_offset];

                if (row_index < 0 || row_index >= ne01) {
                    return -1; // Index out of bounds
                }

                const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                            i12_idx * ne02 * ne01 * ne00 +
                                            i13_idx * ne03 * ne02 * ne01 * ne00;

                const int64_t dst_offset = ri;

                if (src0_type == GGML_TYPE_F32) {
                    // F32 source: direct copy
                    const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset;
                    float* dst_row = (float*)dst_data + dst_offset * ne00;
                    copy_f32_row(dst_row, src_row, ne00);

                } else if (src0_type == GGML_TYPE_Q8_0) {
                    // Q8_0 source: dequantize while copying
                    const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
                    const int64_t src_block_offset = (row_index * blocks_per_row) +
                                                (batch_offset / ne00) * blocks_per_row;
                    const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset;
                    float* dst_row = (float*)dst_data + dst_offset * ne00;
                    copy_q8_0_row(dst_row, src_blocks, ne00);
                } else if (src0_type == GGML_TYPE_Q4_0) {
                    // Q4_0 source: dequantize while copying
                    const int64_t blocks_per_row = (ne00 + QK4_0 - 1) / QK4_0;
                    const int64_t src_block_offset = (row_index * blocks_per_row) +
                                                (batch_offset / ne00) * blocks_per_row;
                    const block_q4_0* src_blocks = (const block_q4_0*)src0_data + src_block_offset;
                    float* dst_row = (float*)dst_data + dst_offset * ne00;
                    copy_q4_0_row(dst_row, src_blocks, ne00);
                } else if (src0_type == GGML_TYPE_Q4_K) {
                    // Q4_K source: dequantize while copying
                    const int64_t blocks_per_row = (ne00 + QK_K - 1) / QK_K;
                    const int64_t src_block_offset = (row_index * blocks_per_row) +
                                                (batch_offset / ne00) * blocks_per_row;
                    const block_q4_K* src_blocks = (const block_q4_K*)src0_data + src_block_offset;
                    float* dst_row = (float*)dst_data + dst_offset * ne00;
                    copy_q4_K_row(dst_row, src_blocks, ne00);
                }
            }
        } else if (op == GGML_OP_SET_ROWS) {
            if (node_meta[i].src0.type == GGML_TYPE_F32 &&
                node_meta[i].src1.type == GGML_TYPE_I64 &&
                (node_meta[i].dst.type == GGML_TYPE_F32 || node_meta[i].dst.type == GGML_TYPE_F16)) {

                if (ne10 != ne01) {
                    return -1; // Number of indices must match number of source rows
                }

                const int64_t total_rows = ne01 * ne02 * ne03;

                // Determine cache-line element count based on destination type
                const int64_t dst_cl_elems = (node_meta[i].dst.type == GGML_TYPE_F16) ? CACHE_LINE_F16_ELEMS
                                                                        : CACHE_LINE_F32_ELEMS;

                // Check if rows are cache-line aligned in the destination
                const bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

                if (row_cache_aligned) {
                    // Cache-aligned path: distribute dst cache lines across threads
                    // Each thread owns complete cache lines -> no coherence conflicts
                    const int64_t cls_per_row    = ne00 / dst_cl_elems;
                    const int64_t total_cls      = total_rows * cls_per_row;
                    const int64_t cls_per_thread = (total_cls + num_threads - 1) / num_threads;
                    const int64_t my_start       = thread_id * cls_per_thread;
                    int64_t       my_end         = my_start + cls_per_thread;
                    if (my_end > total_cls) my_end = total_cls;
                    if (my_start >= total_cls) return 0;

                    for (int64_t cl = my_start; cl < my_end; cl++) {
                        // Map flat cache-line index -> (row, offset within row)
                        const int64_t row_flat  = cl / cls_per_row;
                        const int64_t cl_in_row = cl % cls_per_row;

                        // Decompose flat row -> (i03, i02, i01)
                        const int64_t i01 = row_flat % ne01;
                        const int64_t tmp = row_flat / ne01;
                        const int64_t i02 = tmp % ne02;
                        const int64_t i03 = tmp / ne02;

                        // Look up destination row index
                        const int64_t i12 = i03 % ne12;
                        const int64_t i11 = i02 % ne11;
                        const int64_t i10 = i01;
                        const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
                        const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

                        if (dst_row_index < 0 || dst_row_index >= ne1) {
                            continue;
                        }

                        // Source pointer: row base + cache-line offset (always F32 source)
                        const int64_t elem_offset = cl_in_row * dst_cl_elems;
                        const float* src_ptr = (const float*)((char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03) + elem_offset;

                        // Destination pointer: scattered row base + cache-line offset
                        char* dst_row_base = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

                        if (node_meta[i].dst.type == GGML_TYPE_F32) {
                            float* dst_ptr = (float*)dst_row_base + elem_offset;
                            copy_cache_aligned_f32(dst_ptr, src_ptr);
                        } else {
                            uint16_t* dst_ptr = (uint16_t*)dst_row_base + elem_offset;
                            copy_cache_aligned_f16(dst_ptr, src_ptr);
                        }
                    }
                } else {
                    // Non-aligned path: distribute rows across threads, atomic stores
                    // amoswapg.w / shg bypass local caches -> safe on non-coherent HW
                    for (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += num_threads) {
                        const int64_t i01 = row_flat % ne01;
                        const int64_t tmp = row_flat / ne01;
                        const int64_t i02 = tmp % ne02;
                        const int64_t i03 = tmp / ne02;

                        // Look up destination row index
                        const int64_t i12 = i03 % ne12;
                        const int64_t i11 = i02 % ne11;
                        const int64_t i10 = i01;
                        const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
                        const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

                        if (dst_row_index < 0 || dst_row_index >= ne1) {
                            continue;
                        }

                        const float* src_row = (const float*)((char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03);
                        char* dst_row_base = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

                        if (node_meta[i].dst.type == GGML_TYPE_F32) {
                            volatile float* dst_row = (volatile float*)dst_row_base;
                            for (int64_t i = 0; i < ne00; i++) {
                                atomic_store_f32(dst_row + i, src_row[i]);
                            }
                        } else {
                            volatile uint16_t* dst_row = (volatile uint16_t*)dst_row_base;
                            for (int64_t i = 0; i < ne00; i++) {
                                atomic_store_f16(dst_row + i, fp32_to_fp16(src_row[i]));
                            }
                        }
                    }
                }

            }
        } else if (op == GGML_OP_CONT) {
            if (node_meta[i].dst.type != node_meta[i].src0.type) {
                continue;
            }
            if (node_meta[i].dst.type == GGML_TYPE_F32){
                const int64_t total_elements = ne00 * ne01 * ne02 * ne03;
                if (total_elements == 0) {
                    continue;
                }
                // Create a ggml_tensor structure for src0 to check contiguity
                struct ggml_tensor src0_tensor = {
                    .ne = {ne00, ne01, ne02, ne03},
                    .nb = {nb00, nb01, nb02, nb03},
                    .type = node_meta[i].src0.type,
                    .data = src0_data
                };
                const bool src_contiguous = ggml_tensor_is_contiguous(&src0_tensor, 4);
                //==========================================================================
                // Fast path: src is contiguous: flat vectorized copy by cache lines
                //==========================================================================
                if (src_contiguous) {
                    const int64_t elems_per_cl = 16;
                    const int64_t total_cl = (total_elements + elems_per_cl - 1) / elems_per_cl;

                    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
                    const int64_t cl_start = thread_id * cl_per_thread;
                    int64_t cl_end = cl_start + cl_per_thread;
                    if (cl_end > total_cl) { cl_end = total_cl; }
                    if (cl_start >= total_cl) { continue; }

                    const int64_t es = cl_start * elems_per_cl;
                    int64_t ee = cl_end * elems_per_cl;
                    if (ee > total_elements) { ee = total_elements; }

                    vec_copy_f32((float*)((char*)dst_data + es * sizeof(float)), (float*)((char*)src0_data + es * sizeof(float)), (int32_t)(ee - es));
                    continue;
                }

                //==========================================================================
                // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
                //==========================================================================
                if (nb00 != 4) {
                    // Fully non-contiguous scalar fallback — distribute by cache lines
                    const int64_t elems_per_cl = 16;
                    const int64_t total_cl = (total_elements + elems_per_cl - 1) / elems_per_cl;

                    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
                    const int64_t cl_start = thread_id * cl_per_thread;
                    int64_t cl_end = cl_start + cl_per_thread;
                    if (cl_end > total_cl) { cl_end = total_cl; }
                    if (cl_start >= total_cl) { continue; }

                    const int64_t es = cl_start * elems_per_cl;
                    int64_t ee = cl_end * elems_per_cl;
                    if (ee > total_elements) { ee = total_elements; }

                    for (int64_t idx = es; idx < ee; idx++) {
                        const int64_t i00 = idx % ne00;
                        const int64_t rem1 = idx / ne00;
                        const int64_t i01 = rem1 % ne01;
                        const int64_t rem2 = rem1 / ne01;
                        const int64_t i02 = rem2 % ne02;
                        const int64_t i03 = rem2 / ne02;

                        const float* sp = (const float*)((const char*)src0_data +
                                        i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        ((float*)dst_data)[idx] = *sp;
                    }
                    continue;
                }

                // nb00 == 4 from here: dim 0 is contiguous in src

                //==========================================================================
                // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
                //==========================================================================
                if (ne00 % 16 == 0) {
                    const int64_t total_rows = ne01 * ne02 * ne03;
                    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
                    const int64_t start_row = thread_id * rows_per_thread;
                    const int64_t end_row = (start_row + rows_per_thread < total_rows)
                                        ? (start_row + rows_per_thread) : total_rows;

                    if (start_row >= total_rows) { continue; }

                    for (int64_t ir = start_row; ir < end_row; ir++) {
                        const int64_t i03 = ir / (ne02 * ne01);
                        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
                        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

                        const float* src_row = (const float*)((const char*)src0_data +
                                            i01*nb01 + i02*nb02 + i03*nb03);
                        float* dst_row = (float*)((char*)dst_data + ir * ne00 * sizeof(float));

                        vec_copy_f32(dst_row, src_row, (int32_t)ne00);
                    }
                    continue;
                }

                //==========================================================================
                // Unaligned path: ne00 % 16 != 0, nb00 == 4
                // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
                //==========================================================================
                {
                    const int64_t elems_per_cl = 16;
                    const int64_t total_cl = (total_elements + elems_per_cl - 1) / elems_per_cl;

                    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
                    const int64_t cl_start = thread_id * cl_per_thread;
                    int64_t cl_end = cl_start + cl_per_thread;
                    if (cl_end > total_cl) { cl_end = total_cl; }
                    if (cl_start >= total_cl) { continue; }

                    const int64_t es = cl_start * elems_per_cl;
                    int64_t ee = cl_end * elems_per_cl;
                    if (ee > total_elements) { ee = total_elements; }

                    int64_t pos = es;

                    // Compute starting row coordinates
                    int64_t row_idx = pos / ne00;
                    int64_t col     = pos % ne00;

                    while (pos < ee) {
                        // Decompose row_idx -> (i01, i02, i03)
                        const int64_t i03 = row_idx / (ne02 * ne01);
                        const int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
                        const int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

                        const float* src_row = (const float*)((const char*)src0_data +
                                            i01*nb01 + i02*nb02 + i03*nb03);

                        // How many elements left in this row and in our chunk
                        int64_t row_remaining = ne00 - col;
                        int64_t chunk_remaining = ee - pos;
                        int32_t n = (int32_t)(row_remaining < chunk_remaining ? row_remaining : chunk_remaining);

                        vec_copy_f32((float*)((char*)dst_data + pos * sizeof(float)), src_row + col, n);

                        pos += n;
                        col = 0;  // subsequent rows start at column 0
                        row_idx++;
                    }
                }
            } else if (node_meta[i].dst.type == GGML_TYPE_F16) {
                // F16 CONT implementation - based on cont_f16.c reference
                const int64_t src_elements = ne00 * ne01 * ne02 * ne03;
                const int64_t dst_elements = ne0 * ne1 * ne2 * ne3;
                if (src_elements != dst_elements) {
                    continue; // Element count mismatch
                }

                // Parallelize by rows (dimension 1)
                const int64_t total_rows = ne01;
                const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
                const int64_t start_row = thread_id * rows_per_thread;
                const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

                if (start_row >= total_rows) {
                    continue;
                }

                // Iterate over source tensor dimensions
                for (int64_t i03 = 0; i03 < ne03; i03++) {
                    for (int64_t i02 = 0; i02 < ne02; i02++) {
                        // Calculate base linear index for this (i03, i02) slice in destination
                        const int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

                        // Process this thread's assigned rows
                        for (int64_t i01 = start_row; i01 < end_row; i01++) {
                            // Linear index for start of this row in destination
                            const int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                            // Inner loop over dimension 0
                            for (int64_t i00 = 0; i00 < ne00; i00++) {
                                // Source offset using non-contiguous strides
                                const int64_t src_offset_bytes = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                                const uint16_t* src_ptr = (const uint16_t*)((const char*)src0_data + src_offset_bytes);

                                // Destination linear index (contiguous layout)
                                const int64_t dst_linear_idx = dst_linear_row_base + i00;

                                // Use atomic store for thread safety
                                atomic_store_f16((volatile uint16_t*)((char*)dst_data + dst_linear_idx * sizeof(uint16_t)), *src_ptr);
                            }
                        }
                    }
                }
            } else {
                continue;
            }
            // ggml_et_op_cont(env, &node_meta[i]);
        } else if (op == GGML_OP_MUL_MAT) {
            if (node_meta[i].dst.type == GGML_TYPE_F32 &&
                node_meta[i].src0.type == GGML_TYPE_Q8_0 &&
                node_meta[i].src1.type == GGML_TYPE_F32) {
                    // Q8_0 x F32 matrix multiplication
                    const int64_t K = node_meta[i].src0.ne[0];
                    const int64_t M = node_meta[i].src0.ne[1];
                    const int64_t N = node_meta[i].src1.ne[1];
                    // ne02, ne03, ne12, ne13, ne2, ne3 already defined above
                    
                    const int64_t K_blocks = K / 32;
                    const int64_t r2 = ne12 / ne02;
                    const int64_t r3 = ne13 / ne03;
                    // src0_data, src1_data, dst_data already defined above
                    
                    for (int64_t i3 = 0; i3 < ne3; i3++) {
                        const int64_t i03 = i3 / r3;
                        char* dst_ptr3 = (char*)dst_data + i3 * nb3;

                        for (int64_t i2 = 0; i2 < ne2; i2++) {
                            const int64_t i02 = i2 / r2;
                            const char* src0_ptr2 = (const char*)src0_data + i02 * nb02 + i03 * nb03;
                            const char* src1_ptr2 = (const char*)src1_data + i2 * nb12 + i3 * nb13;
                            char* dst_ptr2 = dst_ptr3 + i2 * nb2;

                            for (int64_t n = 0; n < N; n++) {
                                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                                for (int64_t m = thread_id; m < M; m += num_threads) {
                                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                                    float sum = compute_row_dot_q8_0(q_row, b_col_base, K_blocks);

                                    float* dst_entry = (float*)(dst_ptr2 + n * nb1 + m * sizeof(float));
                                    atomic_store_f32((volatile float*)dst_entry, sum);
                                }
                            }
                        }
                    }
                continue;

            } else if (node_meta[i].dst.type == GGML_TYPE_F32 &&
                        node_meta[i].src0.type == GGML_TYPE_F16 &&
                        node_meta[i].src1.type == GGML_TYPE_F16 &&
                        node_meta[i].src0.ne[0] % 16 == 0 &&
                        node_meta[i].src0.ne[1] % 16 == 0 &&
                        node_meta[i].src1.ne[0] != 1) {
                // F16 x F16 matrix multiplication with matrix engine
                uint64_t hart_id = get_hart_id();
                uint64_t shire_id = get_shire_id();
 
                if (shire_id >= NUM_COMPUTE_SHIRES) continue;
                if (hart_id & 1) continue;
 
                uint64_t local_minion = (hart_id >> 1) & 0x1F;
                uint64_t my_minion_id = get_minion_id();

                const int64_t K = node_meta[i].src0.ne[0];
                const int64_t M = node_meta[i].src0.ne[1];
                const int64_t N = node_meta[i].src1.ne[1];
                // ne02, ne03 (ne2_0, ne3_0) and ne12, ne13 (ne2_1, ne3_1) already defined above
                // nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3 already defined above
                // src0_data, src1_data, dst_data already defined above

                const char *src0_base = (const char *) src0_data;
                const char *src1_base = (const char *) src1_data;
                char       *dst_base  = (char *) dst_data;

                setup_cache_scp();
            #if CACHEOP_MAX > 0 || REP_RATE > 0
                ucache_control(1, REP_RATE, CACHEOP_MAX);
            #endif
                CLEAR_TENSOR_ERROR;

                if ((M % TILE_M) != 0) continue;
                if ((K % TILE_K) != 0) continue;

                const int64_t m_tiles = M / TILE_M;
                const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
                const int64_t batch_count = ne12 * ne13;  // ne12, ne13 already defined
                const int64_t base_tiles = m_tiles * n_tiles * batch_count;

                const int64_t r2 = ne12 / ne02;
                const int64_t r3 = ne13 / ne03;

                const int64_t total_harts = NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
                const int64_t k_steps = K / TILE_K;

                int64_t k_splits = 1;
                if (base_tiles < total_harts) {
                    k_splits = (total_harts + base_tiles - 1) / base_tiles;
                    int64_t ks = 1;
                    while (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
                        ks *= 2;
                    }
                    k_splits = ks;
                }

                const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
                const int64_t k_split = local_minion % k_splits;
                const int64_t local_tile_idx = local_minion / k_splits;
                const int64_t tiles_stride = (int64_t)NUM_COMPUTE_SHIRES * tiles_per_shire;

                const int64_t k_steps_per_split = k_steps / k_splits;
                const int64_t k_start = k_split * k_steps_per_split * TILE_K;
                const int64_t k_end   = k_start + k_steps_per_split * TILE_K;

                const uint64_t group_base_global = my_minion_id - k_split;

                // Interleaved B panel: 16 lines x 32 fp16 = 1024 bytes
                et_fp16_t bpanel[16 * 32] __attribute__((aligned(64)));

                for (int64_t tile = (int64_t)shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
                    tile < base_tiles;
                    tile += tiles_stride) {

                    const int64_t tiles_per_batch = m_tiles * n_tiles;
                    const int64_t batch_idx       = tile / tiles_per_batch;
                    const int64_t tile_in_batch   = tile % tiles_per_batch;

                    const int64_t nb_idx = tile_in_batch / m_tiles;
                    const int64_t mb_idx = tile_in_batch % m_tiles;

                    const int64_t i3   = batch_idx / ne12;
                    const int64_t i2   = batch_idx % ne12;
                    const int64_t i2_0 = i2 / r2;
                    const int64_t i3_0 = i3 / r3;

                    const char *src0_batch = src0_base + i3_0 * nb03 + i2_0 * nb02;
                    const char *src1_batch = src1_base + i3   * nb13 + i2   * nb12;
                    char       *dst_batch  = dst_base  + i3   * nb3  + i2   * nb2;

                    const int64_t mb = mb_idx * TILE_M;
                    const int64_t nb = nb_idx * TILE_N;
                    const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

                    // Set tensor_mask for partial N tiles: bit i = 1 means row i is active
                    if (n_cur < TILE_N) {
                        uint64_t mask = (1ULL << n_cur) - 1;
                        __asm__ __volatile__("csrw 0x805, %0" : : "r"(mask));
                    }

                    for (int64_t kb = k_start; kb < k_end; kb += TILE_K) {

                        // Load A from src1. n_cur rows x 32 FP16 = n_cur x 64B
                        // Use tensor_mask when n_cur < 16 to skip invalid rows
                        tensor_load(
                            (n_cur < TILE_N), false,
                            A_L1_START,
                            TENSOR_LOAD_PLAIN,
                            0, // use_tenb
                            (uint64_t)(src1_batch + nb * nb11 + kb * (int64_t)sizeof(et_fp16_t)),
                            0,
                            n_cur - 1,
                            (uint64_t)nb11,
                            0
                        );

                        // Build interleaved B panel from src0 and flush to L2
                        // so the tensor load (which bypasses L1) can see it
                        // There is no TensorLoadInterleavedTranpose16 so we
                        // interleave outselves and then TensorLoad
                        pack_b_interleaved(bpanel, src0_batch, mb, kb, nb01);

                        FENCE;
                        flush_to_l2(bpanel, 16, 64);
                        WAIT_CACHEOPS;

                        // Load B from manually interleaved data, 16 lines x 64B
                        tensor_load(
                            false, false,
                            B_L1_START,
                            TENSOR_LOAD_PLAIN,
                            0, // use_tenb
                            (uint64_t)bpanel,
                            0,
                            15, // 16 lines
                            64, // contiguous 64B stride
                            1
                        );

                        tensor_wait(TENSOR_LOAD_WAIT_0);
                        tensor_wait(TENSOR_LOAD_WAIT_1);

                        // TensorFMA16A32:
                        //   BCOLS  = 3       -> (3+1)*4 = 16 output columns
                        //   AROWS  = n_cur-1 -> n_cur A rows
                        //   ACOLS  = 15      -> 2*(15+1) = 32 FP16 K-values
                        tensor_fma(
                            (n_cur < TILE_N), // use_tmask
                            3,                // b_num_col
                            n_cur - 1,        // a_num_rows
                            15,               // a_num_cols
                            0,                // offset
                            false,            // tenc_loc
                            false,            // tenb_unsigned
                            false,            // tena_unsigned
                            false,            // tenb_loc: B in L1SCP
                            B_L1_START,
                            A_L1_START,
                            TENSOR_FMA_OP_FP16,
                            (kb == k_start)   // first_pass
                        );

                        tensor_wait(TENSOR_FMA_WAIT);
                    }

                    // K-split ring reduce
                    if (k_splits > 1) {
                        const uint64_t num_regs = (uint64_t)n_cur * 2;

                        if (k_split > 0) {
                            tensor_reduce_recv(
                                0, TENSOR_REDUCE_OP_FADD,
                                num_regs,
                                group_base_global + k_split - 1
                            );
                            tensor_wait(TENSOR_REDUCE_WAIT);
                        }

                        if (k_split < k_splits - 1) {
                            tensor_reduce_send(
                                0, num_regs,
                                group_base_global + k_split + 1
                            );
                            tensor_wait(TENSOR_REDUCE_WAIT);
                        }
                    }

                    // Store FP32 result tile
                    if (k_split == k_splits - 1) {
                        tensor_store(
                            0, 0, 3, n_cur - 1,
                            (uint64_t)(dst_batch + nb * nb1 + mb * (int64_t)sizeof(float)),
                            0, (uint64_t)nb1
                        );
                        tensor_wait(TENSOR_STORE_WAIT);
                    }
                }

                FENCE;
                continue;

            } else if (node_meta[i].dst.type == GGML_TYPE_F32 &&
                    node_meta[i].src0.type == GGML_TYPE_F16 &&
                    node_meta[i].src1.type == GGML_TYPE_F32) {
                // F16 x F32 matrix multiplication
                int effective_thread_id = thread_id / 2;
                int effective_num_threads = (num_threads + 1) / 2;

                // Validate: src0 is F16, others are F32
                if (node_meta[i].src0.type != GGML_TYPE_F16 || node_meta[i].src1.type != GGML_TYPE_F32 || node_meta[i].dst.type != GGML_TYPE_F32) {
                    continue;
                }

                // Dimensions: K, M, N
                const int64_t K = node_meta[i].src0.ne[0];
                const int64_t M = node_meta[i].src0.ne[1];
                const int64_t N = node_meta[i].src1.ne[1];
                // ne02, ne03, ne12, ne13, ne2, ne3 already defined above

                // F16 specific block size (Usually QK_F16)
                const int block_size = QK_F16;
                const int64_t K_blocks = K / block_size;
                const int64_t K_remainder = K % block_size;

                // Threading distribution
                const uint64_t total_elements = M * N * ne2 * ne3;
                const uint64_t per_thread = 16;
                const uint64_t threads_stride = per_thread * effective_num_threads;

                if (effective_thread_id * per_thread >= total_elements) continue;

                // Broadcasting support
                const int64_t r2 = ne12 / ne02;
                const int64_t r3 = ne13 / ne03;

                for (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
                    for (uint64_t j = 0; j < per_thread; j++) {
                        const uint64_t idx = base_idx + j;
                        if (idx >= total_elements) break;

                        // Index decoding
                        const int64_t i3 = idx / (M * N * ne2);
                        const int64_t rem3 = idx % (M * N * ne2);
                        const int64_t i2 = rem3 / (M * N);
                        const int64_t rem2 = rem3 % (M * N);
                        const int64_t n = rem2 / M;
                        const int64_t m = rem2 % M;

                        const int64_t i03 = i3 / r3, i02 = i2 / r2;
                        const int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

                        float sum = 0.0f;
                        const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

                        // Process full blocks using vectorized F16 dot product
                        for (int64_t kb = 0; kb < K_blocks; kb++) {
                            const float* b_col_ptr = (const float*)((const char*)src1_data + (kb * block_size) * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                            sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_ptr);
                        }

                        // Handle partial remainder
                        if (K_remainder > 0) {
                            const int64_t offset = K_blocks * block_size;
                            const float* b_col_ptr = (const float*)((const char*)src1_data + offset * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                            sum += compute_block_dot_product_f16_partial(&f16_row[offset], b_col_ptr, K_remainder);
                        }

                        // Atomic store for output
                        volatile float* c_element = (volatile float*)((char*)dst_data + m * nb0 + n * nb1 + i2 * nb2 + i3 * nb3);
                        atomic_store_f32(c_element, sum);
                    }
                }
                continue;

            } else if (node_meta[i].dst.type == GGML_TYPE_F32 &&
                        node_meta[i].src0.type == GGML_TYPE_F32 &&
                        node_meta[i].src1.type == GGML_TYPE_F32 &&
                        node_meta[i].src0.ne[0] % 16 == 0 &&
                        node_meta[i].src0.ne[1] % 16 == 0 &&
                        node_meta[i].src1.ne[0] != 1) { 
                // GEMV is faster with the generic path
                // F32 x F32 matrix multiplication with matrix engine
                uint64_t hart_id = get_hart_id();
                uint64_t shire_id = get_shire_id();

                if (shire_id >= NUM_COMPUTE_SHIRES) continue;
                if (hart_id & 1) continue;

                uint64_t local_minion = (hart_id >> 1) & 0x1F;
                uint64_t my_minion_id = get_minion_id();

                const int64_t K = node_meta[i].src0.ne[0];
                const int64_t M = node_meta[i].src0.ne[1];
                const int64_t N = node_meta[i].src1.ne[1];

                // ne02, ne03, ne12, ne13 already defined above
                // nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3 already defined above

                const char* src0_base = (const char*)src0_data;
                const char* src1_base = (const char*)src1_data;
                char*       dst_base  = (char*)dst_data;

                setup_cache_scp();
            #if CACHEOP_MAX_TFMA_F32 > 0 || REP_RATE_TFMA_F32 > 0
                ucache_control(1, REP_RATE_TFMA_F32, CACHEOP_MAX_TFMA_F32);
            #endif
                CLEAR_TENSOR_ERROR;

                const int64_t m_tiles = M / TILE_M_TFMA_F32;
                const int64_t n_tiles = (N + TILE_N_TFMA_F32 - 1) / TILE_N_TFMA_F32;
                const int64_t batch_count = ne12 * ne13;
                const int64_t base_tiles = m_tiles * n_tiles * batch_count;

                const int64_t r2 = ne12 / ne02;
                const int64_t r3 = ne13 / ne03;

                const int64_t total_harts = NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
                const int64_t k_steps = K / TILE_K_TFMA_F32;
                int64_t k_splits = 1;
                if (base_tiles < total_harts) {
                    k_splits = (total_harts + base_tiles - 1) / base_tiles;
                    int64_t ks = 1;
                    while (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
                        ks *= 2;
                    }
                    k_splits = ks;
                }

                const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
                const int64_t k_split = local_minion % k_splits;
                const int64_t local_tile_idx = local_minion / k_splits;
                const int64_t tiles_stride = (int64_t)NUM_COMPUTE_SHIRES * tiles_per_shire;

                const int64_t k_steps_per_split = k_steps / k_splits;
                const int64_t k_start = k_split * k_steps_per_split * TILE_K_TFMA_F32;
                const int64_t k_end   = k_start + k_steps_per_split * TILE_K_TFMA_F32;

                const uint64_t group_base_global = my_minion_id - k_split;

                for (int64_t tile = (int64_t)shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
                    tile < base_tiles;
                    tile += tiles_stride) {

                    const int64_t tiles_per_batch = m_tiles * n_tiles;
                    const int64_t batch_idx     = tile / tiles_per_batch;
                    const int64_t tile_in_batch = tile % tiles_per_batch;
                    const int64_t nb_idx = tile_in_batch / m_tiles;
                    const int64_t mb_idx = tile_in_batch % m_tiles;

                    const int64_t i3   = batch_idx / ne12;
                    const int64_t i2   = batch_idx % ne12;
                    const int64_t i2_0 = i2 / r2;
                    const int64_t i3_0 = i3 / r3;

                    const char* src0_batch = src0_base + i3_0 * nb03 + i2_0 * nb02;
                    const char* src1_batch = src1_base + i3   * nb13 + i2   * nb12;
                    char*       dst_batch  = dst_base  + i3   * nb3  + i2   * nb2;

                    const int64_t mb = mb_idx * TILE_M_TFMA_F32;
                    const int64_t nb = nb_idx * TILE_N_TFMA_F32;
                    const int64_t n_cur = (nb + TILE_N_TFMA_F32 <= N) ? TILE_N_TFMA_F32 : (N - nb);

                    for (int64_t kb = k_start; kb < k_end; kb += TILE_K_TFMA_F32) {

                        tensor_load(
                            false, false, 0, 0, 0,
                            (uint64_t)(src1_batch + nb * nb11 + kb * sizeof(float)),
                            0, n_cur - 1, (uint64_t)nb11, 0
                        );

                        tensor_load(
                            false, false, TILE_K_TFMA_F32, 7, 0,
                            (uint64_t)(src0_batch + mb * nb01 + kb * sizeof(float)),
                            0, TILE_K_TFMA_F32 - 1, (uint64_t)nb01, 1
                        );

                        tensor_wait(TENSOR_LOAD_WAIT_0);
                        tensor_wait(TENSOR_LOAD_WAIT_1);

                        tensor_fma(
                            false, 3, n_cur - 1, TILE_K_TFMA_F32 - 1, 0,
                            false, false, false, false,
                            TILE_K_TFMA_F32, 0, 0,
                            (kb == k_start)
                        );

                        tensor_wait(TENSOR_FMA_WAIT);
                    }

                    if (k_splits > 1) {
                        const uint64_t num_regs = (uint64_t)n_cur * 2;

                        if (k_split > 0) {
                            tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD,
                                            num_regs,
                                            group_base_global + k_split - 1);
                            tensor_wait(TENSOR_REDUCE_WAIT);
                        }
                        if (k_split < k_splits - 1) {
                            tensor_reduce_send(0, num_regs,
                                            group_base_global + k_split + 1);
                            tensor_wait(TENSOR_REDUCE_WAIT);
                        }
                    }

                    if (k_split == k_splits - 1) {
                        tensor_store(
                            0, 0, 3, n_cur - 1,
                            (uint64_t)(dst_batch + nb * nb1 + mb * sizeof(float)),
                            0, (uint64_t)nb1
                        );
                        tensor_wait(TENSOR_STORE_WAIT);
                    }
                }

                FENCE;
                
                continue;
            } else if (node_meta[i].dst.type == GGML_TYPE_F32 &&
                    node_meta[i].src0.type == GGML_TYPE_F32 &&
                    node_meta[i].src1.type == GGML_TYPE_F32) {
                // F32 x F32 matrix multiplication
                int effective_thread_id = thread_id / 2;
                int effective_num_threads = (num_threads + 1) / 2;

                // Use node_meta[i] for tensor metadata
                const int64_t K = node_meta[i].src0.ne[0];
                const int64_t M = node_meta[i].src0.ne[1];
                const int64_t N = node_meta[i].src1.ne[1];

                // ne02, ne03, ne12, ne13, ne2, ne3 already defined above
                // nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3 already defined above
                // src0_data, src1_data, dst_data already defined above

                // F32 specific block size and counts
                const int block_size = QK_F32;
                const int64_t K_blocks = K / block_size;
                const int64_t K_remainder = K % block_size;

                // Threading distribution
                const uint64_t total_elements = M * N * ne2 * ne3;
                const uint64_t per_thread = 16;
                const uint64_t threads_stride = per_thread * effective_num_threads;

                if (effective_thread_id * per_thread >= total_elements) return 0;

                // Broadcasting support
                const int64_t r2 = ne12 / ne02;
                const int64_t r3 = ne13 / ne03;

                for (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
                    for (uint64_t j = 0; j < per_thread; j++) {
                        const uint64_t idx = base_idx + j;
                        if (idx >= total_elements) break;

                        // Index decoding
                        const int64_t i3 = idx / (M * N * ne2);
                        const int64_t rem3 = idx % (M * N * ne2);
                        const int64_t i2 = rem3 / (M * N);
                        const int64_t rem2 = rem3 % (M * N);
                        const int64_t n = rem2 / M;
                        const int64_t m = rem2 % M;

                        const int64_t i03 = i3 / r3, i02 = i2 / r2;
                        const int64_t i13 = (ne13 > 1) ? i3 : 0, i12 = (ne12 > 1) ? i2 : 0;

                        float sum = 0.0f;
                        const float* f32_row = (const float*)((const char*)src0_data + m * nb01 + i02 * nb02 + i03 * nb03);

                        // Process full blocks
                        for (int64_t kb = 0; kb < K_blocks; kb++) {
                            const float* b_col_ptr = (const float*)((const char*)src1_data + (kb * block_size) * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                            sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_ptr);
                        }

                        // Handle partial remainder
                        if (K_remainder > 0) {
                            const int64_t offset = K_blocks * block_size;
                            const float* b_col_ptr = (const float*)((const char*)src1_data + offset * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                            sum += compute_block_dot_product_f32_partial(&f32_row[offset], b_col_ptr, K_remainder);
                        }

                        // Atomic store for output
                        volatile float* c_element = (volatile float*)((char*)dst_data + m * nb0 + n * nb1 + i2 * nb2 + i3 * nb3);
                        atomic_store_f32(c_element, sum);
                    }
                }
                
                continue;
            } else {
                continue; // Unsupported type combination
            }
            // ggml_et_op_mul_mat(env, &node_meta[i]);
        } else if (op == GGML_OP_ROPE) {

            // struct ggml_et_rope_params params;
            // convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
            // convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
            // convert_to_ggml_tensor(&params.src2, &node_meta[i].src2, GGML_OP_NONE);
            // convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_ROPE);
            // memcpy(&params.rope_params.n_past, &node_meta[i].op_params[0], sizeof(int32_t));
            // memcpy(&params.rope_params.n_dims, &node_meta[i].op_params[1], sizeof(int32_t));
            // memcpy(&params.rope_params.mode, &node_meta[i].op_params[2], sizeof(int32_t));
            // memcpy(&params.rope_params.n_ctx, &node_meta[i].op_params[3], sizeof(int32_t));
            // memcpy(&params.rope_params.n_ctx_orig, &node_meta[i].op_params[4], sizeof(int32_t));
            // memcpy(&params.rope_params.freq_base, &node_meta[i].op_params[5], sizeof(float));
            // memcpy(&params.rope_params.freq_scale, &node_meta[i].op_params[6], sizeof(float));
            // memcpy(&params.rope_params.ext_factor, &node_meta[i].op_params[7], sizeof(float));
            // memcpy(&params.rope_params.attn_factor, &node_meta[i].op_params[8], sizeof(float));
            // memcpy(&params.rope_params.beta_fast, &node_meta[i].op_params[9], sizeof(float));
            // memcpy(&params.rope_params.beta_slow, &node_meta[i].op_params[10], sizeof(float));
            // for (int j = 0; j < 4; j++) {
            //     memcpy(&params.rope_params.sections[j], &node_meta[i].op_params[11 + j], sizeof(int32_t));
            // }
            // if (params.dst.type == GGML_TYPE_F32 &&
            //     params.src0.type == GGML_TYPE_F32 &&
            //     params.src1.type == GGML_TYPE_I32) {
            //     rope_f32_impl(&params, env);
            // }
            ggml_et_op_rope(env, &node_meta[i]);

            // ggml_et_op_rope(env, &node_meta[i]);
        } else if (op == GGML_OP_RMS_NORM) {
            const float inv_ne0 = et_fdiv(1.0f, (float)(int32_t)ne0);
            const int32_t total_rows = (int32_t)(ne1 * ne2 * ne3);
            float eps;
            memcpy(&eps, node_meta[i].op_params, sizeof(float));
            // Intra-row cooperation only works within a single shire (barrier + L2SCP
            // are shire-local). Use per-shire thread count for the threshold.
            const int shire_threads = SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION; // 64

            if (total_rows >= shire_threads) {
                // Row-parallel: each thread processes whole rows
                for (int64_t i3 = 0; i3 < ne3; i3++) {
                    for (int64_t i2 = 0; i2 < ne2; i2++) {
                        for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {

                        const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
                        float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

                        // Set mask to enable all 8 vector lanes
                        unsigned long saved_mask;
                        __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                        __asm__ volatile("mov.m.x m0, x0, 0xFF");

                        // Step 1: Compute sum of squares using 8-wide vectors
                        __asm__ volatile("fbci.pi f10, 0" ::: "f10");

                        for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                            __asm__ volatile(
                                "flw.ps f11, %[x_vec]\n"
                                "fmadd.ps f10, f11, f11, f10\n"
                                :
                                : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                                : "f10", "f11"
                            );
                        }

                        // Horizontal reduce
                        float sum;
                        __asm__ __volatile__ (
                            "fswizz.ps f1, f10, 0xB1 \n\t"
                            "fadd.ps   f2, f10, f1, rne \n\t"
                            "fswizz.ps f3, f2, 0x4E \n\t"
                            "fadd.ps   f4, f2, f3, rne \n\t"
                            "fmvz.x.ps t0, f4, 4 \n\t"
                            "fbcx.ps   f5, t0 \n\t"
                            "fadd.ps   %[vout], f4, f5, rne \n\t"
                            : [vout] "=f" (sum)
                            :: "t0", "f1", "f2", "f3", "f4", "f5"
                        );

                        // Step 2: scale = rsqrt(mean + eps)
                        const float scale = et_powf(sum * inv_ne0 + eps, -0.5f);

                        if (!(scale > 0.0f)) {
                            __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
                            return -1;
                        }

                        // Step 3: Apply scaling: broadcast scale once, reuse across loop
                        uint32_t scale_bits;
                        __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                        __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                        for (int32_t i0 = 0; i0 < (int32_t)ne0; i0 += 8) {
                            __asm__ volatile(
                                "flw.ps f12, %[x_vec]\n"
                                "fmul.ps f14, f12, f13\n"
                                "fsw.ps f14, %[result]\n"
                                : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                                : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                                : "f12", "f14"
                            );
                        }

                        __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
                        }
                    }
                }
            } else {
                // Intra-row: threads within each shire cooperate on rows via L2 SCP.
                // L2 SCP + barrier are shire-local, so use shire-local thread index.
                int shire_tid = thread_id % shire_threads;  // 0..63 within this shire
                int threads_per_row = shire_threads / total_rows;
                int my_row    = shire_tid / threads_per_row;
                int local_tid = shire_tid % threads_per_row;
                int group_base = my_row * threads_per_row; // shire-local group base

                // Excess threads within this shire, barrier and leave
                if (my_row >= total_rows) {
                    FENCE;
                    et_barrier(ET_BARRIER_SHIRE);
                    return 0;
                }

                // Unflatten row index
                int64_t i1 = my_row % ne1;
                int64_t i2 = (my_row / ne1) % ne2;
                int64_t i3 = my_row / (ne1 * ne2);

                const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
                float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

                // Chunk boundaries aligned to 16 floats (64-byte cache line)
                const int32_t elems_per_cl = 16;
                int32_t total_cls = ((int32_t)ne0 + elems_per_cl - 1) / elems_per_cl;
                int32_t cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
                int32_t my_start = local_tid * cls_per_thread * elems_per_cl;
                int32_t my_end   = my_start + cls_per_thread * elems_per_cl;
                if (my_end > (int32_t)ne0) my_end = (int32_t)ne0;
                if (my_start >= (int32_t)ne0) { my_start = 0; my_end = 0; }

                unsigned long saved_mask;
                __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
                __asm__ volatile("mov.m.x m0, x0, 0xFF");

                // Phase 1: each thread computes partial sum of squares on its chunk
                __asm__ volatile("fbci.pi f10, 0" ::: "f10");
                for (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                    __asm__ volatile(
                        "flw.ps f11, %[x_vec]\n"
                        "fmadd.ps f10, f11, f11, f10\n"
                        :
                        : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                        : "f10", "f11"
                    );
                }

                // Horizontal reduce to scalar
                float partial_sum;
                __asm__ __volatile__ (
                    "fswizz.ps f1, f10, 0xB1 \n\t"
                    "fadd.ps   f2, f10, f1, rne \n\t"
                    "fswizz.ps f3, f2, 0x4E \n\t"
                    "fadd.ps   f4, f2, f3, rne \n\t"
                    "fmvz.x.ps t0, f4, 4 \n\t"
                    "fbcx.ps   f5, t0 \n\t"
                    "fadd.ps   %[vout], f4, f5, rne \n\t"
                    : [vout] "=f" (partial_sum)
                    :: "t0", "f1", "f2", "f3", "f4", "f5"
                );

                // Phase 2: write partial sum to L2 SCP, evict from L1D
                volatile float* my_slot = (volatile float*)et_shire_l2scp_local(
                    (uint64_t)shire_tid * 64);
                *my_slot = partial_sum;
                FENCE;
                evict_to_l2((const void*)my_slot, 1, 64);
                WAIT_CACHEOPS;

                et_barrier(ET_BARRIER_SHIRE);

                // Phase 3: ALL threads read partial sums, compute scale, apply to own chunk.
                // Each thread independently reduces to avoid a second barrier.
                int workers = threads_per_row < total_cls ? threads_per_row : total_cls;

                // Evict stale L1D entries for worker slots
                for (int t = 0; t < workers; t++) {
                    volatile float* slot = (volatile float*)et_shire_l2scp_local(
                        (uint64_t)(group_base + t) * 64);
                    evict_to_l2((const void*)slot, 1, 64);
                }
                WAIT_CACHEOPS;

                // Every thread reduces the same partial sums -> same scale
                float total_sum = 0.0f;
                for (int t = 0; t < workers; t++) {
                    volatile float* slot = (volatile float*)et_shire_l2scp_local(
                        (uint64_t)(group_base + t) * 64);
                    total_sum += *slot;
                }

                const float scale = et_powf(total_sum * inv_ne0 + eps, -0.5f);
                if (!(scale > 0.0f)) {
                    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
                    return -1;
                }

                // Each thread applies scale to its own chunk only
                if (my_start < my_end) {
                    uint32_t scale_bits;
                    __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    for (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f12, %[x_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                            : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                            : "f12", "f14"
                        );
                    }
                }

                __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
            }
            // ggml_et_op_rms_norm(env, &node_meta[i]);
        } else if (op == GGML_OP_SQR) {
            ggml_et_op_sqr(env, &node_meta[i]);
        } else if (op == GGML_OP_UNARY) {
            ggml_et_op_unary(env, &node_meta[i]);
        } else if (op == GGML_OP_SUM_ROWS) {
            ggml_et_op_sum_rows(env, &node_meta[i]);
        } else if (op == GGML_OP_CUMSUM) {
            ggml_et_op_cumsum(env, &node_meta[i]);
        } else if (op == GGML_OP_MUL_MAT_ID) {
            ggml_et_op_mul_mat_id(env, &node_meta[i]);
        } else if (op == GGML_OP_NORM) {
            ggml_et_op_norm(env, &node_meta[i]);
        } else if (op == GGML_OP_L2_NORM) {
            ggml_et_op_l2_norm(env, &node_meta[i]);
        } else if (op == GGML_OP_SCALE) {
            ggml_et_op_scale(env, &node_meta[i]);
        } else if (op == GGML_OP_CPY) {
            ggml_et_op_cpy(env, &node_meta[i]);
        } else if (op == GGML_OP_CONCAT) {
            ggml_et_op_concat(env, &node_meta[i]);
        } else if (op == GGML_OP_REPEAT) {
            ggml_et_op_repeat(env, &node_meta[i]);
        } else if (op == GGML_OP_PAD) {
            ggml_et_op_pad(env, &node_meta[i]);
        } else if (op == GGML_OP_SET) {
            ggml_et_op_set(env, &node_meta[i]);
        } else if (op == GGML_OP_FILL) {
            ggml_et_op_fill(env, &node_meta[i]);
        } else if (op == GGML_OP_DIAG) {
            ggml_et_op_diag(env, &node_meta[i]);
        } else if (op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE) {
            // No-op operations
        }
        else {
                return -1;
        }

        // if (op != GGML_OP_RESHAPE &&
        //     op != GGML_OP_VIEW    &&
        //     op != GGML_OP_PERMUTE &&
        //     op != GGML_OP_TRANSPOSE) {
            // device_barrier(32);
        // }
    }

    return 0;
}
