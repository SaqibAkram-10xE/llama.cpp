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
static void ggml_et_op_elmap(void * env, struct ggml_node_meta_et * m,
                              enum ggml_op operation) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    float * src0 = (float *)(uintptr_t)m->src0.data;
    float * src1 = (float *)(uintptr_t)m->src1.data;
    float * dst  = (float *)(uintptr_t)m->dst.data;
    if (!src0 || !src1 || !dst) return;

    const int64_t ne0 = m->dst.ne[0], ne1 = m->dst.ne[1];
    const int64_t ne2 = m->dst.ne[2], ne3 = m->dst.ne[3];
    const int64_t ne10 = m->src1.ne[0], ne11 = m->src1.ne[1];
    const int64_t ne12 = m->src1.ne[2], ne13 = m->src1.ne[3];

    const size_t nb0 = (size_t)m->dst.nb[0], nb1 = (size_t)m->dst.nb[1];
    const size_t nb2 = (size_t)m->dst.nb[2], nb3 = (size_t)m->dst.nb[3];
    const size_t nb00 = (size_t)m->src0.nb[0], nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2], nb03 = (size_t)m->src0.nb[3];
    const size_t nb10 = (size_t)m->src1.nb[0], nb11 = (size_t)m->src1.nb[1];
    const size_t nb12 = (size_t)m->src1.nb[2], nb13 = (size_t)m->src1.nb[3];

    // Fast path: no broadcasting, contiguous
    const bool no_bcast = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    const bool all_contig = (nb0 == 4 && nb00 == 4 && nb10 == 4 &&
                             nb1 == ne0*4 && nb01 == ne0*4 && nb11 == ne0*4);

    if (no_bcast && all_contig && (ne0 % 16 == 0)) {
        const int64_t total = ne0 * ne1 * ne2 * ne3;
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
            if (operation == GGML_OP_MUL) {
                __asm__ volatile(
                    "flw.ps f10, %[a]\n" "flw.ps f11, %[b]\n"
                    "fmul.ps f12, f10, f11\n" "fsw.ps f12, %[d]\n"
                    : [d] "=m"(*(float(*)[8])&dst[i])
                    : [a] "m"(*(const float(*)[8])&src0[i]),
                      [b] "m"(*(const float(*)[8])&src1[i])
                    : "f10","f11","f12");
            } else if (operation == GGML_OP_ADD) {
                __asm__ volatile(
                    "flw.ps f10, %[a]\n" "flw.ps f11, %[b]\n"
                    "fadd.ps f12, f10, f11\n" "fsw.ps f12, %[d]\n"
                    : [d] "=m"(*(float(*)[8])&dst[i])
                    : [a] "m"(*(const float(*)[8])&src0[i]),
                      [b] "m"(*(const float(*)[8])&src1[i])
                    : "f10","f11","f12");
            } else {
                __asm__ volatile(
                    "flw.ps f10, %[a]\n" "flw.ps f11, %[b]\n"
                    "fsub.ps f12, f10, f11\n" "fsw.ps f12, %[d]\n"
                    : [d] "=m"(*(float(*)[8])&dst[i])
                    : [a] "m"(*(const float(*)[8])&src0[i]),
                      [b] "m"(*(const float(*)[8])&src1[i])
                    : "f10","f11","f12");
            }
        }
        return;
    }

    // Slow path: broadcasting
    const int64_t total_rows = ne1 * ne2 * ne3;
    for (int64_t ir = tid; ir < total_rows; ir += nth) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
        const int64_t i13 = i3 % ne13, i12 = i2 % ne12, i11 = i1 % ne11;

        float * dp  = (float *)((char *)dst  + i3*nb3  + i2*nb2  + i1*nb1);
        const float * s0p = (const float *)((const char *)src0 + i3*nb03 + i2*nb02 + i1*nb01);
        const float * s1p = (const float *)((const char *)src1 + i13*nb13 + i12*nb12 + i11*nb11);

        if (ne10 == 1) {
            float scalar = s1p[0];
            for (int64_t i = 0; i < ne0; i++) {
                if (operation == GGML_OP_MUL) dp[i] = s0p[i] * scalar;
                else if (operation == GGML_OP_ADD) dp[i] = s0p[i] + scalar;
                else dp[i] = s0p[i] - scalar;
            }
        } else {
            for (int64_t i = 0; i < ne0; i++) {
                float a = s0p[i], b = s1p[i % ne10];
                if (operation == GGML_OP_MUL) dp[i] = a * b;
                else if (operation == GGML_OP_ADD) dp[i] = a + b;
                else dp[i] = a - b;
            }
        }
    }
}

static void ggml_et_op_mul(void * env, struct ggml_node_meta_et * m) {
    ggml_et_op_elmap(env, m, GGML_OP_MUL);
}
static void ggml_et_op_add(void * env, struct ggml_node_meta_et * m) {
    ggml_et_op_elmap(env, m, GGML_OP_ADD);
}
static void ggml_et_op_sub(void * env, struct ggml_node_meta_et * m) {
    ggml_et_op_elmap(env, m, GGML_OP_SUB);
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
static inline float silu_scalar(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return et_fdiv(x, 1.0f + et_expf(-x));
}

static void ggml_et_op_glu(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    float * src0 = (float *)(uintptr_t)m->src0.data;
    float * src1 = m->src1.data ? (float *)(uintptr_t)m->src1.data : 0;
    float * dst  = (float *)(uintptr_t)m->dst.data;
    if (!src0 || !dst) return;

    int32_t glu_op = m->op_params[0];
    int32_t swapped = m->op_params[1];

    const int64_t nc = m->dst.ne[0];
    const int64_t nr = m->dst.ne[1] * m->dst.ne[2] * m->dst.ne[3];
    const size_t src0_stride = (size_t)m->src0.nb[1];
    const size_t src1_stride = src1 ? (size_t)m->src1.nb[1] : src0_stride;
    const size_t dst_stride  = (size_t)m->dst.nb[1];

    const int64_t total_rows = nr;
    for (int64_t row = tid; row < total_rows; row += nth) {
        float * dp = (float *)((char *)dst + row * dst_stride);
        float * xp, * gp;

        if (src1) {
            xp = (float *)((char *)src0 + row * src0_stride);
            gp = (float *)((char *)src1 + row * src1_stride);
        } else {
            float * s0_row = (float *)((char *)src0 + row * src0_stride);
            if (swapped) { gp = s0_row; xp = s0_row + nc; }
            else         { xp = s0_row; gp = s0_row + nc; }
        }

        if (glu_op == GGML_GLU_OP_SWIGLU) {
            for (int64_t i = 0; i < nc; i++) {
                dp[i] = silu_scalar(xp[i]) * gp[i];
            }
        } else if (glu_op == GGML_GLU_OP_GEGLU) {
            for (int64_t i = 0; i < nc; i++) {
                float x = xp[i];
                float inner = 0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x);
                float e2z = et_expf(2.0f * inner);
                float gelu = x * et_fdiv(e2z, e2z + 1.0f);
                dp[i] = gelu * gp[i];
            }
        } else {
            for (int64_t i = 0; i < nc; i++) dp[i] = xp[i] * gp[i];
        }
    }
}

// ========================================================================
// OP: SOFTMAX
// ========================================================================
#define LOG2E_CG 1.4426950408889634f

static void ggml_et_op_softmax(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src0 = (const float *)(uintptr_t)m->src0.data;
    float * dst        = (float *)(uintptr_t)m->dst.data;
    if (!src0 || !dst) return;

    const float * mask_data = m->src1.data ? (const float *)(uintptr_t)m->src1.data : 0;

    float scale, max_bias;
    memcpy(&scale,    &m->op_params[0], sizeof(float));
    memcpy(&max_bias, &m->op_params[1], sizeof(float));

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne03 = m->src0.ne[3];

    const int64_t ne10 = mask_data ? m->src1.ne[0] : 0;
    const int64_t ne11 = mask_data ? m->src1.ne[1] : 0;
    const int64_t ne12 = mask_data ? m->src1.ne[2] : 0;
    const int64_t ne13 = mask_data ? m->src1.ne[3] : 0;

    // ALiBi slope calculation
    const uint32_t n_head = (uint32_t)ne02;
    uint32_t n_head_log2 = 1;
    float m0 = 1.0f, m1 = 1.0f;

    if (max_bias > 0.0f) {
        while (n_head_log2 < n_head) n_head_log2 <<= 1;
        if (n_head_log2 > n_head) n_head_log2 >>= 1;
        float inv = et_fdiv(1.0f, (float)n_head_log2);
        m0 = et_expf(-max_bias * 0.69314718f * inv);
        m1 = et_expf(-max_bias * 0.69314718f * inv * 0.5f);
    }

    const int64_t rows_per_batch = ne02 * ne01;
    const int64_t total_rows = ne03 * rows_per_batch;

    for (int64_t row = tid; row < total_rows; row += nth) {
        const int64_t i03 = row / rows_per_batch;
        const int64_t rem = row % rows_per_batch;
        const int64_t i02 = rem / ne01;
        const int64_t i01 = rem % ne01;

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            uint32_t h = (uint32_t)i02;
            if (h < n_head_log2) {
                slope = m0;
                for (uint32_t k = 0; k < h; k++) slope *= m0;
            } else {
                uint32_t exp_val = 2 * (h - n_head_log2) + 1;
                slope = m1;
                for (uint32_t k = 1; k < exp_val; k++) slope *= m1;
            }
        }

        const int64_t src_off = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;
        const float * sr = src0 + src_off;
        float * dr = dst + src_off;
        const float * mr = 0;

        if (mask_data) {
            int64_t mi3 = ne13 > 0 ? i03 % ne13 : 0;
            int64_t mi2 = ne12 > 0 ? i02 % ne12 : 0;
            mr = mask_data + mi3 * ne12 * ne11 * ne10 + mi2 * ne11 * ne10 + i01 * ne10;
        }

        // Find max
        float max_val = -3.402823466e+38f;
        for (int64_t i = 0; i < ne00; i++) {
            float v = sr[i] * scale;
            if (mr) v += mr[i] * slope;
            if (v > max_val) max_val = v;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t i = 0; i < ne00; i++) {
            float v = sr[i] * scale;
            if (mr) v += mr[i] * slope;
            float e = et_expf(v - max_val);
            dr[i] = e;
            sum += e;
        }

        // Normalize
        float inv_sum = sum == 0.0f ? 0.0f : et_fdiv(1.0f, sum);
        for (int64_t i = 0; i < ne00; i++) {
            dr[i] *= inv_sum;
        }
    }
}

// ========================================================================
// OP: ROPE
// ========================================================================
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

// ========================================================================
// OP: MUL_MAT  —  C[M,N] = A[M,K] * B[K,N]
// Supports Q8_0, F16, F32 weight types; F32 activations
// ========================================================================
static void ggml_et_op_mul_mat(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    // Use only even threads to avoid minion resource contention
    if (tid & 1) return;
    int etid = tid / 2;
    int enth = (nth + 1) / 2;

    const int64_t K    = m->src0.ne[0];
    const int64_t M    = m->src0.ne[1];
    const int64_t N    = m->src1.ne[1];
    const int64_t ne02 = m->src0.ne[2], ne03 = m->src0.ne[3];
    const int64_t ne12 = m->src1.ne[2], ne13 = m->src1.ne[3];
    const int64_t ne2  = m->dst.ne[2],  ne3  = m->dst.ne[3];

    const size_t nb01 = (size_t)m->src0.nb[1], nb02 = (size_t)m->src0.nb[2], nb03 = (size_t)m->src0.nb[3];
    const size_t nb11 = (size_t)m->src1.nb[1], nb12 = (size_t)m->src1.nb[2], nb13 = (size_t)m->src1.nb[3];
    const size_t nb1  = (size_t)m->dst.nb[1],  nb2  = (size_t)m->dst.nb[2],  nb3  = (size_t)m->dst.nb[3];

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const void * src0_data = (const void *)(uintptr_t)m->src0.data;
    const float * src1_data = (const float *)(uintptr_t)m->src1.data;
    float * dst_data = (float *)(uintptr_t)m->dst.data;
    if (!src0_data || !src1_data || !dst_data) return;

    const int src0_type = m->src0.type;

    const uint64_t total_elems = (uint64_t)M * N * ne2 * ne3;
    const uint64_t per_thread = 16;
    const uint64_t stride = per_thread * enth;

    if (src0_type == GGML_TYPE_Q8_0) {
        const int64_t K_blocks = K / 32;
        for (uint64_t base = etid * per_thread; base < total_elems; base += stride) {
            for (uint64_t j = 0; j < per_thread && base + j < total_elems; j++) {
                uint64_t idx = base + j;
                int64_t i3 = idx / (M * N * ne2);
                int64_t rem3 = idx % (M * N * ne2);
                int64_t i2 = rem3 / (M * N);
                int64_t rem2 = rem3 % (M * N);
                int64_t n = rem2 / M;
                int64_t mm = rem2 % M;

                int64_t i03 = i3 / r3, i02 = i2 / r2;
                int64_t i13 = (ne13 > 1) ? i3 : 0;
                int64_t i12 = (ne12 > 1) ? i2 : 0;

                const block_q8_0 * q_row = (const block_q8_0 *)((const char *)src0_data + mm * nb01 + i02 * nb02 + i03 * nb03);
                const float * b_col = (const float *)((const char *)src1_data + n * nb11 + i12 * nb12 + i13 * nb13);
                float sum = compute_row_dot_q8_0(q_row, b_col, K_blocks);

                volatile float * c = (volatile float *)((char *)dst_data + mm * sizeof(float) + n * nb1 + i2 * nb2 + i3 * nb3);
                atomic_store_f32(c, sum);
            }
        }
    } else if (src0_type == GGML_TYPE_F16) {
        const int64_t K_blocks = K / QK_F16;
        const int64_t K_rem = K % QK_F16;
        for (uint64_t base = etid * per_thread; base < total_elems; base += stride) {
            for (uint64_t j = 0; j < per_thread && base + j < total_elems; j++) {
                uint64_t idx = base + j;
                int64_t i3 = idx / (M * N * ne2);
                int64_t rem3 = idx % (M * N * ne2);
                int64_t i2 = rem3 / (M * N);
                int64_t rem2 = rem3 % (M * N);
                int64_t n = rem2 / M;
                int64_t mm = rem2 % M;

                int64_t i03 = i3 / r3, i02 = i2 / r2;
                int64_t i13 = (ne13 > 1) ? i3 : 0;
                int64_t i12 = (ne12 > 1) ? i2 : 0;

                const uint16_t * f16_row = (const uint16_t *)((const char *)src0_data + mm * nb01 + i02 * nb02 + i03 * nb03);
                float sum = 0.0f;
                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * bp = (const float *)((const char *)src1_data + kb * QK_F16 * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_naive(&f16_row[kb * QK_F16], bp);
                }
                if (K_rem > 0) {
                    int64_t off = K_blocks * QK_F16;
                    const float * bp = (const float *)((const char *)src1_data + off * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f16_partial(&f16_row[off], bp, K_rem);
                }

                volatile float * c = (volatile float *)((char *)dst_data + mm * sizeof(float) + n * nb1 + i2 * nb2 + i3 * nb3);
                atomic_store_f32(c, sum);
            }
        }
    } else {
        // F32
        const int64_t K_blocks = K / QK_F32;
        const int64_t K_rem = K % QK_F32;
        for (uint64_t base = etid * per_thread; base < total_elems; base += stride) {
            for (uint64_t j = 0; j < per_thread && base + j < total_elems; j++) {
                uint64_t idx = base + j;
                int64_t i3 = idx / (M * N * ne2);
                int64_t rem3 = idx % (M * N * ne2);
                int64_t i2 = rem3 / (M * N);
                int64_t rem2 = rem3 % (M * N);
                int64_t n = rem2 / M;
                int64_t mm = rem2 % M;

                int64_t i03 = i3 / r3, i02 = i2 / r2;
                int64_t i13 = (ne13 > 1) ? i3 : 0;
                int64_t i12 = (ne12 > 1) ? i2 : 0;

                const float * f32_row = (const float *)((const char *)src0_data + mm * nb01 + i02 * nb02 + i03 * nb03);
                float sum = 0.0f;
                for (int64_t kb = 0; kb < K_blocks; kb++) {
                    const float * bp = (const float *)((const char *)src1_data + kb * QK_F32 * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32(&f32_row[kb * QK_F32], bp);
                }
                if (K_rem > 0) {
                    int64_t off = K_blocks * QK_F32;
                    const float * bp = (const float *)((const char *)src1_data + off * sizeof(float) + n * nb11 + i12 * nb12 + i13 * nb13);
                    sum += compute_block_dot_product_f32_partial(&f32_row[off], bp, K_rem);
                }

                volatile float * c = (volatile float *)((char *)dst_data + mm * sizeof(float) + n * nb1 + i2 * nb2 + i3 * nb3);
                atomic_store_f32(c, sum);
            }
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
static void ggml_et_op_get_rows(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const void * src0_data = (const void *)(uintptr_t)m->src0.data;
    const int32_t * src1_data = (const int32_t *)(uintptr_t)m->src1.data;
    float * dst_data = (float *)(uintptr_t)m->dst.data;
    if (!src0_data || !src1_data || !dst_data) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne10 = m->src1.ne[0];
    const int64_t ne11 = m->src1.ne[1];
    const int64_t ne12 = m->src1.ne[2];
    const int64_t ne13 = m->src1.ne[3];
    const int src0_type = m->src0.type;

    const int64_t total_rows = ne10 * ne11 * ne12 * ne13;

    for (int64_t i = tid; i < total_rows; i += nth) {
        int32_t row_index = src1_data[i];
        if (row_index < 0 || row_index >= ne01) continue;

        float * dst_row = dst_data + i * ne00;

        if (src0_type == GGML_TYPE_F32) {
            const float * src_row = (const float *)src0_data + row_index * ne00;
            for (int64_t k = 0; k < ne00; k++) dst_row[k] = src_row[k];
        } else if (src0_type == GGML_TYPE_Q8_0) {
            int64_t bpr = (ne00 + QK8_0 - 1) / QK8_0;
            const block_q8_0 * blocks = (const block_q8_0 *)src0_data + row_index * bpr;
            for (int64_t bi = 0; bi < bpr; bi++) {
                float temp[QK8_0];
                dequantize_q8_0_block(&blocks[bi], temp);
                int64_t elems = (bi == bpr - 1) ? (ne00 - bi * QK8_0) : QK8_0;
                for (int64_t k = 0; k < elems; k++) dst_row[bi * QK8_0 + k] = temp[k];
            }
        } else if (src0_type == GGML_TYPE_Q4_0) {
            int64_t bpr = (ne00 + QK4_0 - 1) / QK4_0;
            const block_q4_0 * blocks = (const block_q4_0 *)src0_data + row_index * bpr;
            for (int64_t bi = 0; bi < bpr; bi++) {
                float temp[QK4_0];
                dequantize_q4_0_block(&blocks[bi], temp);
                int64_t elems = (bi == bpr - 1) ? (ne00 - bi * QK4_0) : QK4_0;
                for (int64_t k = 0; k < elems; k++) dst_row[bi * QK4_0 + k] = temp[k];
            }
        } else if (src0_type == GGML_TYPE_Q4_K) {
            int64_t bpr = (ne00 + QK_K - 1) / QK_K;
            const block_q4_K * blocks = (const block_q4_K *)src0_data + row_index * bpr;
            for (int64_t bi = 0; bi < bpr; bi++) {
                float temp[QK_K];
                dequantize_q4_K_block(&blocks[bi], temp);
                int64_t elems = (bi == bpr - 1) ? (ne00 - bi * QK_K) : QK_K;
                for (int64_t k = 0; k < elems; k++) dst_row[bi * QK_K + k] = temp[k];
            }
        }
    }
}

// ========================================================================
// OP: SET_ROWS  —  inverse of GET_ROWS (F32 src -> F32/F16 dst)
// ========================================================================
static void ggml_et_op_set_rows(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const float * src0_data = (const float *)(uintptr_t)m->src0.data;
    const int64_t * src1_data = (const int64_t *)(uintptr_t)m->src1.data;
    void * dst_data = (void *)(uintptr_t)m->dst.data;
    if (!src0_data || !src1_data || !dst_data) return;

    const int64_t ne00 = m->src0.ne[0];
    const int64_t ne01 = m->src0.ne[1];
    const int64_t ne02 = m->src0.ne[2];
    const int64_t ne03 = m->src0.ne[3];
    const size_t nb01 = (size_t)m->src0.nb[1];
    const size_t nb02 = (size_t)m->src0.nb[2];
    const size_t nb03 = (size_t)m->src0.nb[3];
    const size_t nb1 = (size_t)m->dst.nb[1];
    const size_t nb2 = (size_t)m->dst.nb[2];
    const size_t nb3 = (size_t)m->dst.nb[3];
    const int64_t ne10 = m->src1.ne[0];
    const int64_t ne11 = m->src1.ne[1];
    const int64_t ne12 = m->src1.ne[2];
    const size_t nb10 = (size_t)m->src1.nb[0];
    const size_t nb11 = (size_t)m->src1.nb[1];
    const size_t nb12 = (size_t)m->src1.nb[2];
    const int dst_type = m->dst.type;

    const int64_t total_rows = ne01 * ne02 * ne03;

    for (int64_t rf = tid; rf < total_rows; rf += nth) {
        int64_t i01 = rf % ne01;
        int64_t tmp = rf / ne01;
        int64_t i02 = tmp % ne02;
        int64_t i03 = tmp / ne02;

        int64_t i12 = i03 % ne12;
        int64_t i11 = i02 % ne11;
        int64_t idx_off = i01 * nb10 + i11 * nb11 + i12 * nb12;
        int64_t dst_row_idx = *(int64_t *)((char *)src1_data + idx_off);

        const float * src_row = (const float *)((const char *)src0_data + i01*nb01 + i02*nb02 + i03*nb03);
        char * dst_row_base = (char *)dst_data + dst_row_idx*nb1 + i02*nb2 + i03*nb3;

        if (dst_type == GGML_TYPE_F32) {
            float * dp = (float *)dst_row_base;
            for (int64_t k = 0; k < ne00; k++) {
                atomic_store_f32((volatile float *)&dp[k], src_row[k]);
            }
        } else {
            volatile uint16_t * dp = (volatile uint16_t *)dst_row_base;
            for (int64_t k = 0; k < ne00; k++) {
                atomic_store_f16(&dp[k], fp32_to_fp16(src_row[k]));
            }
        }
    }
}

// ========================================================================
// OP: CONT  —  make contiguous (F32 and F16)
// ========================================================================
static void ggml_et_op_cont(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const void * src_data = (const void *)(uintptr_t)m->src0.data;
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

    const int64_t total_rows = ne01 * ne02 * ne03;
    const int elem_size = (m->src0.type == GGML_TYPE_F16) ? 2 : 4;

    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t i01 = row % ne01;
        int64_t i02 = (row / ne01) % ne02;
        int64_t i03 = row / (ne01 * ne02);

        const char * sp = (const char *)src_data + i01*nb01 + i02*nb02 + i03*nb03;
        char * dp = (char *)dst_data + row * ne00 * elem_size;

        if (nb00 == (size_t)elem_size) {
            // Row is contiguous in source: bulk copy
            for (int64_t i = 0; i < ne00 * elem_size; i++) dp[i] = sp[i];
        } else {
            // Non-contiguous
            for (int64_t i00 = 0; i00 < ne00; i00++) {
                const char * s = sp + i00 * nb00;
                char * d = dp + i00 * elem_size;
                for (int k = 0; k < elem_size; k++) d[k] = s[k];
            }
        }
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
        ggml_et_op_cont(env, m);
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
#define FA_DV_MAX_CG 128

static void ggml_et_op_flash_attn_ext(void * env, struct ggml_node_meta_et * m) {
    int tid, nth;
    if (cg_thread_setup(env, &tid, &nth)) return;

    const char * q_data = (const char *)(uintptr_t)m->src0.data;
    const char * k_data = (const char *)(uintptr_t)m->src1.data;
    const char * v_data = (const char *)(uintptr_t)m->src2.data;
    char * dst_data     = (char *)(uintptr_t)m->dst.data;
    if (!q_data || !k_data || !v_data || !dst_data) return;

    float scale;
    memcpy(&scale, &m->op_params[0], sizeof(float));

    const int64_t dk  = m->src0.ne[0];
    const int64_t nq  = m->src0.ne[1];
    const int64_t nhq = m->src0.ne[2];
    const int64_t no  = m->src0.ne[3];
    const int64_t nk  = m->src1.ne[1];
    const int64_t nhk = m->src1.ne[2];
    const int64_t dv  = m->src2.ne[0];

    if (dv > FA_DV_MAX_CG) return;

    const int k_type = m->src1.type;
    const int v_type = m->src2.type;
    const int64_t gqa_ratio = nhq / nhk;
    const int64_t total_rows = nq * nhq * no;

    for (int64_t row = tid; row < total_rows; row += nth) {
        int64_t iq3 = row / (nhq * nq);
        int64_t rem = row % (nhq * nq);
        int64_t iq2 = rem / nq;
        int64_t iq1 = rem % nq;
        int64_t ik2 = iq2 / gqa_ratio;

        const float * pq = (const float *)(q_data +
            iq1*(size_t)m->src0.nb[1] + iq2*(size_t)m->src0.nb[2] + iq3*(size_t)m->src0.nb[3]);
        float * out = (float *)(dst_data +
            iq2*(size_t)m->dst.nb[1] + iq1*(size_t)m->dst.nb[2] + iq3*(size_t)m->dst.nb[3]);

        int64_t kv_base_k = ik2*(size_t)m->src1.nb[2] + iq3*(size_t)m->src1.nb[3];
        int64_t kv_base_v = ik2*(size_t)m->src2.nb[2] + iq3*(size_t)m->src2.nb[3];

        float acc[FA_DV_MAX_CG];
        for (int64_t d = 0; d < dv; d++) acc[d] = 0.0f;
        float M = -3.402823466e+38f;
        float S = 0.0f;

        for (int64_t ik1 = 0; ik1 < nk; ik1++) {
            const char * pk = k_data + ik1*(size_t)m->src1.nb[1] + kv_base_k;
            const char * pv = v_data + ik1*(size_t)m->src2.nb[1] + kv_base_v;

            // QK dot
            float s = 0.0f;
            if (k_type == GGML_TYPE_F32) {
                const float * kf = (const float *)pk;
                for (int64_t d = 0; d < dk; d++) s += pq[d] * kf[d];
            } else {
                for (int64_t d = 0; d < dk; d++)
                    s += pq[d] * fp16_to_fp32(*(const uint16_t *)(pk + d*(size_t)m->src1.nb[0]));
            }
            s = s * scale;

            float Mold = M;
            float ms = 1.0f, vs = 1.0f;
            if (s > M) {
                M = s;
                ms = et_expf(Mold - M);
                for (int64_t d = 0; d < dv; d++) acc[d] *= ms;
            } else {
                vs = et_expf(s - M);
            }

            if (v_type == GGML_TYPE_F32) {
                const float * vf = (const float *)pv;
                for (int64_t d = 0; d < dv; d++) acc[d] += vf[d] * vs;
            } else {
                for (int64_t d = 0; d < dv; d++)
                    acc[d] += fp16_to_fp32(*(const uint16_t *)(pv + d*(size_t)m->src2.nb[0])) * vs;
            }
            S = S * ms + vs;
        }

        float S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
        for (int64_t d = 0; d < dv; d++) {
            out[d] = acc[d] * S_inv;
        }
    }
}

// ========================================================================
// Entry point — graph execution loop
// ========================================================================
int entry_point(struct ggml_cgraph_et * cg, void * env) {
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *)cg->data;
    uint8_t * node_op = (uint8_t *)(node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    // device_barrier(32);

    for (int i = 0; i < n_nodes; i++) {
        const int op = node_op[i];

        if (op == GGML_OP_NONE) {
            continue;
        }

        // // Fusion: RMS_NORM + MUL -> fused RMS_NORM_MUL
        // if (op == GGML_OP_RMS_NORM &&
        //     ggml_et_can_fuse(cg, i, node_op, n_nodes,
        //                      (enum ggml_op[]){ GGML_OP_RMS_NORM, GGML_OP_MUL }, 2)) {
        //     ggml_et_op_rms_norm_mul(env, &node_meta[i], &node_meta[i + 1]);
        //     i++;
        //     device_barrier(32);
        //     continue;
        // }

        switch (op) {
            case GGML_OP_SQR:            ggml_et_op_sqr(env, &node_meta[i]); break;
            case GGML_OP_UNARY:          ggml_et_op_unary(env, &node_meta[i]); break;
            case GGML_OP_SUM_ROWS:       ggml_et_op_sum_rows(env, &node_meta[i]); break;
            case GGML_OP_MUL:            ggml_et_op_mul(env, &node_meta[i]); break;
            case GGML_OP_ADD:            ggml_et_op_add(env, &node_meta[i]); break;
            case GGML_OP_SUB:            ggml_et_op_sub(env, &node_meta[i]); break;
            case GGML_OP_CUMSUM:         ggml_et_op_cumsum(env, &node_meta[i]); break;
            case GGML_OP_MUL_MAT:        ggml_et_op_mul_mat(env, &node_meta[i]); break;
            case GGML_OP_MUL_MAT_ID:     ggml_et_op_mul_mat_id(env, &node_meta[i]); break;
            case GGML_OP_ROPE:           ggml_et_op_rope(env, &node_meta[i]); break;
            case GGML_OP_RMS_NORM:       ggml_et_op_rms_norm(env, &node_meta[i]); break;
            case GGML_OP_NORM:           ggml_et_op_norm(env, &node_meta[i]); break;
            case GGML_OP_L2_NORM:        ggml_et_op_l2_norm(env, &node_meta[i]); break;
            case GGML_OP_SCALE:          ggml_et_op_scale(env, &node_meta[i]); break;
            case GGML_OP_GLU:            ggml_et_op_glu(env, &node_meta[i]); break;
            case GGML_OP_SOFT_MAX:       ggml_et_op_softmax(env, &node_meta[i]); break;
            case GGML_OP_FLASH_ATTN_EXT: ggml_et_op_flash_attn_ext(env, &node_meta[i]); break;
            case GGML_OP_GET_ROWS:       ggml_et_op_get_rows(env, &node_meta[i]); break;
            case GGML_OP_SET_ROWS:       ggml_et_op_set_rows(env, &node_meta[i]); break;
            case GGML_OP_CONT:           ggml_et_op_cont(env, &node_meta[i]); break;
            case GGML_OP_CPY:            ggml_et_op_cpy(env, &node_meta[i]); break;
            case GGML_OP_CONCAT:         ggml_et_op_concat(env, &node_meta[i]); break;
            case GGML_OP_REPEAT:         ggml_et_op_repeat(env, &node_meta[i]); break;
            case GGML_OP_PAD:            ggml_et_op_pad(env, &node_meta[i]); break;
            case GGML_OP_SET:            ggml_et_op_set(env, &node_meta[i]); break;
            case GGML_OP_FILL:           ggml_et_op_fill(env, &node_meta[i]); break;
            case GGML_OP_DIAG:           ggml_et_op_diag(env, &node_meta[i]); break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                return -1;
        }

        if (op != GGML_OP_RESHAPE &&
            op != GGML_OP_VIEW    &&
            op != GGML_OP_PERMUTE &&
            op != GGML_OP_TRANSPOSE) {
            device_barrier(32);
        }
    }

    return 0;
}
