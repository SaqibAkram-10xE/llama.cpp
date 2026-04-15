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

// ROPE type defines for monolithic path
#define GGML_ROPE_TYPE_NEOX_CG   2
#define GGML_ROPE_TYPE_MROPE_CG  8
#define GGML_ROPE_TYPE_IMROPE_CG 40

#define MAX_ROPE_HALF_DIMS 128
#define ROPE_VEC_WIDTH 8

#define ROPE_PI         3.14159265358979323846f
#define ROPE_TWO_PI     6.28318530717958647693f
#define ROPE_PI_OVER_2  1.57079632679489661923f
#define ROPE_INV_TWO_PI 0.15915494309189533577f

// Helper function for thread setup
static inline int cg_thread_setup(void * env, int * out_tid, int * out_nth) {
    kernel_environment_t * ke = (kernel_environment_t *)env;
    if (!ke) return -1;
    *out_tid = get_relative_thread_id(ke->shire_mask);
    *out_nth = get_num_threads(ke->shire_mask);
    if (*out_tid < 0) return -1;
    return 0;
}

//------------------------------------------------------------------------------
// ROPE helper functions for YaRN and IMROPE support
//------------------------------------------------------------------------------

static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
    float denom = high - low;
    if (denom < 0.001f) denom = 0.001f;
    const float y = et_fdiv((float)(i0 / 2) - low, denom);
    const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - clamped;
}

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
        : : "r"(old_mask)
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
    float max_bias;
    float logit_softcap;
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

        // device_barrier(32);

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
                    int tid, nth;
                    if (cg_thread_setup(env, &tid, &nth)) break;

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

            case GGML_OP_ROPE:
                {
                    int tid, nth;
                    if (cg_thread_setup(env, &tid, &nth)) break;

                    const float * src0_data   = (const float *)(uintptr_t)node_meta[i].src0.data;
                    const int32_t * src1_data = (const int32_t *)(uintptr_t)node_meta[i].src1.data;
                    const float * freq_factors = node_meta[i].src2.data ? (const float *)(uintptr_t)node_meta[i].src2.data : 0;
                    float * dst_data          = (float *)(uintptr_t)node_meta[i].dst.data;
                    if (!src0_data || !src1_data || !dst_data) break;

                    const int64_t head_dim = node_meta[i].src0.ne[0];
                    const int64_t heads    = node_meta[i].src0.ne[1];
                    const int64_t seq_len  = node_meta[i].src0.ne[2];
                    const int64_t batch    = node_meta[i].src0.ne[3];

                    int32_t n_past     = node_meta[i].op_params[0];
                    int32_t n_dims     = node_meta[i].op_params[1];
                    int32_t mode       = node_meta[i].op_params[2];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                    int32_t n_ctx_orig = node_meta[i].op_params[4];
                    int32_t sections[4];
                    memcpy(&freq_base,   &node_meta[i].op_params[5],  sizeof(float));
                    memcpy(&freq_scale,  &node_meta[i].op_params[6],  sizeof(float));
                    memcpy(&ext_factor,  &node_meta[i].op_params[7],  sizeof(float));
                    memcpy(&attn_factor, &node_meta[i].op_params[8],  sizeof(float));
                    memcpy(&beta_fast,   &node_meta[i].op_params[9],  sizeof(float));
                    memcpy(&beta_slow,   &node_meta[i].op_params[10], sizeof(float));
                    for (int j = 0; j < 4; j++) {
                        memcpy(&sections[j], &node_meta[i].op_params[11 + j], sizeof(int32_t));
                    }

                    if (n_dims <= 0 || n_dims > head_dim) break;

                    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));
                    const int32_t half_dims = n_dims / 2;
                    const int is_neox = (mode & GGML_ROPE_TYPE_NEOX_CG) != 0;
                    const int is_imrope = (mode == GGML_ROPE_TYPE_IMROPE_CG);
                    const int use_neox_rotation = is_neox || is_imrope;

                    // YaRN correction dimensions
                    float corr_dims[2] = {0.0f, 0.0f};
                    if (n_ctx_orig > 0 && beta_fast > 0.0f) {
                        float cd_s = (float)n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta_fast) * 2.0f);
                        float cd_e = (float)n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta_slow) * 2.0f);
                        corr_dims[0] = cd_s > 0.0f ? cd_s : 0.0f;
                        corr_dims[1] = cd_e < (float)(n_dims - 1) ? cd_e : (float)(n_dims - 1);
                    }

                    // Cache buffers
                    float cos_cache[MAX_ROPE_HALF_DIMS] __attribute__((aligned(32)));
                    float sin_cache[MAX_ROPE_HALF_DIMS] __attribute__((aligned(32)));

                    const int64_t total_heads = batch * seq_len * heads;

                    // For IMROPE position cache invalidation: track all 4 channels
                    int32_t last_pos = -1;
                    int32_t last_pos_h = -1;
                    int32_t last_pos_w = -1;
                    int32_t last_pos_e = -1;

                    for (int64_t wu = tid; wu < total_heads; wu += nth) {
                        const int64_t h = wu % heads;
                        const int64_t s = (wu / heads) % seq_len;
                        const int64_t b = wu / (heads * seq_len);

                        if (is_imrope) {
                            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
                            const int32_t pt = src1_data[s]              + n_past;
                            const int32_t ph = src1_data[s + seq_len]    + n_past;
                            const int32_t pw = src1_data[s + seq_len * 2] + n_past;
                            const int32_t pe = src1_data[s + seq_len * 3] + n_past;

                            if (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                                compute_imrope_cache(
                                    cos_cache, sin_cache,
                                    n_dims, theta_scale,
                                    pt, ph, pw, pe,
                                    sections,
                                    freq_factors, freq_scale,
                                    corr_dims, ext_factor, attn_factor
                                );
                                last_pos = pt;
                                last_pos_h = ph;
                                last_pos_w = pw;
                                last_pos_e = pe;
                            }
                        } else {
                            const int32_t pos = src1_data[s] + n_past;

                            if (pos != last_pos) {
                                compute_rope_cache(
                                    cos_cache, sin_cache,
                                    n_dims, theta_scale, pos,
                                    freq_factors, freq_scale,
                                    corr_dims, ext_factor, attn_factor
                                );
                                last_pos = pos;
                            }
                        }

                        const float * head_src = (const float *)((const char *)src0_data +
                            b * (size_t)node_meta[i].src0.nb[3] + s * (size_t)node_meta[i].src0.nb[2] + h * (size_t)node_meta[i].src0.nb[1]);
                        float * head_dst = (float *)((char *)dst_data +
                            b * (size_t)node_meta[i].dst.nb[3] + s * (size_t)node_meta[i].dst.nb[2] + h * (size_t)node_meta[i].dst.nb[1]);

                        // Copy dimensions beyond n_dims unchanged
                        for (int64_t d = n_dims; d < head_dim; ++d) {
                            head_dst[d] = head_src[d];
                        }

                        if (use_neox_rotation) {
                            // NEOX/IMROPE: pairs at (i, i+half_dims) - use vectorized path
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
            FENCE; // drain all stores to L1 before barrier
            device_barrier(32);
        }

        // device_barrier(32);
        
    }


    return 0;
}
