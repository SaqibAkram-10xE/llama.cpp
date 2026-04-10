// //******************************************************************************
// // ROPE (Rotary Position Encoding) Kernel
// // Applies rotary position encoding:
// //   f32[head_dim, heads, seq_len] x i32 -> f32[head_dim, heads, seq_len]
// //******************************************************************************

// #include <stdint.h>
// #include <stdbool.h>
// #include "ggml_tensor.h"
// #include "platform.h"
// #include "math_fp.h"

// // ROPE constants (matching GGML definitions)
// #define GGML_ROPE_TYPE_NEOX 2
// #define CACHE_LINE_SIZE_F32 16

// // ROPE operation parameters structure (matches ggml-et-ops.h)
// typedef struct {
//     int32_t n_past;
//     int32_t n_dims;        // Number of dimensions to apply ROPE to (must be even)
//     int32_t mode;          // ROPE mode (0=normal, 2=neox)
//     int32_t n_ctx;
//     int32_t n_ctx_orig;
//     float   freq_base;     // Base frequency (usually 10000.0f)
//     float   freq_scale;    // Frequency scaling factor
//     float   ext_factor;    // Extension factor for YaRN
//     float   attn_factor;   // Attention factor for YaRN
//     float   beta_fast;     // Fast beta for YaRN
//     float   beta_slow;     // Slow beta for YaRN
//     int32_t sections[4];   // Sections for multi-modal ROPE
// } rope_params_t;

// // ROPE kernel parameters structure (matches ggml_et_rope_params)
// struct ggml_et_rope_params {
//     struct ggml_tensor src0;  // F32 input tensor
//     struct ggml_tensor src1;  // I32 position tensor
//     struct ggml_tensor src2;  // F32 frequency factors (optional)
//     struct ggml_tensor dst;   // F32 output tensor
//     rope_params_t rope_params;
// };

// // YaRN helper functions
// static inline float rope_yarn_ramp(const float low, const float high, const int i0) {
//     float denom = high - low;
//     if (denom < 0.001f) denom = 0.001f;  // MAX(0.001f, high - low)

//     const float y = et_fdiv((float)(i0 / 2) - low, denom);
//     const float clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);  // MIN(1, MAX(0, y))
//     return 1.0f - clamped;
// }

// static inline float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float beta, float freq_base) {
//     return n_dims * et_fdiv(et_logf(et_fdiv((float)n_ctx_orig, freq_base)), et_logf(beta) * 2.0f);
// }

// static inline void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
//                                        float beta_fast, float beta_slow, float dims[2]) {
//     float start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
//     float end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);

//     dims[0] = start > 0.0f ? start : 0.0f;
//     dims[1] = end < (float)(n_dims - 1) ? end : (float)(n_dims - 1);
// }

// // YaRN algorithm (MIT licensed, Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng)
// static inline void rope_yarn(float theta_extrap, float freq_scale, const float corr_dims[2],
//                              int64_t i0, float ext_factor, float mscale,
//                              float* cos_theta, float* sin_theta) {
//     // theta_interp uses frequency scaling
//     float theta_interp = freq_scale * theta_extrap;
//     float theta = theta_interp;

//     if (ext_factor != 0.0f) {
//         // Mix between interpolated and extrapolated based on dimension
//         float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
//         theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;

//         // Magnitude scaling correction for interpolation
//         mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
//     }

//     *cos_theta = et_cosf(theta) * mscale;
//     *sin_theta = et_sinf(theta) * mscale;
// }

// #ifdef ENABLE_MONOLITHIC_COMPUTE
// #define ROPE_F32_FUNC rope_f32_impl
// #else
// #define ROPE_F32_FUNC entry_point
// #endif

// int ROPE_F32_FUNC(struct ggml_et_rope_params* params, void* env) {
//     kernel_environment_t* kernel_env = (kernel_environment_t*)env;

//     if (!kernel_env) {
//         return -1;
//     }

//     int thread_id = get_relative_thread_id(kernel_env->shire_mask);
//     int num_threads = get_num_threads(kernel_env->shire_mask);

//     if (thread_id < 0) {
//         return -1;
//     }

//     if (params == 0 || ((uint64_t)params & 0x7) != 0) {
//         return -1; // Invalid pointer
//     }

//     struct ggml_tensor* src0 = &params->src0; // F32 input tensor [head_dim, heads, seq_len, batch]
//     struct ggml_tensor* src1 = &params->src1; // I32 position tensor [seq_len]
//     struct ggml_tensor* src2 = &params->src2; // F32 frequency factors (optional)
//     struct ggml_tensor* dst = &params->dst;   // F32 output tensor [head_dim, heads, seq_len, batch]

//     if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
//         return -1; // Unsupported type combination
//     }

//     const float* src0_data = (const float*)src0->data;    // F32 input activations
//     const int32_t* src1_data = (const int32_t*)src1->data; // I32 positions
//     const float* freq_factors = NULL;                      // Optional frequency factors
//     if (src2 && src2->data) {
//         freq_factors = (const float*)src2->data;
//     }
//     float* dst_data = (float*)dst->data;                  // F32 output

//     if (!src0_data || !src1_data || !dst_data) {
//         return -1; // Null data pointer
//     }

//     // Get tensor dimensions
//     // src0: [head_dim, heads, seq_len, batch]
//     // src1: [seq_len] (positions)
//     // dst:  [head_dim, heads, seq_len, batch]
//     const int64_t head_dim = src0->ne[0];   // Head dimension (e.g., 128)
//     const int64_t heads = src0->ne[1];      // Number of heads (e.g., 32 for Q, 8 for K/V)
//     const int64_t seq_len = src0->ne[2];    // Sequence length (e.g., 512)
//     const int64_t batch = src0->ne[3];      // Batch size

//     const rope_params_t* rope_params = &params->rope_params;
//     const int32_t n_dims = rope_params->n_dims;           // Dimensions to apply ROPE to
//     const float freq_base = rope_params->freq_base;       // Base frequency (10000.0)
//     const float freq_scale = rope_params->freq_scale;     // Frequency scaling
//     const int32_t mode = rope_params->mode;               // ROPE mode

//     if (n_dims <= 0 || n_dims > head_dim || n_dims % 2 != 0) {
//         return -1; // Invalid ROPE dimensions
//     }

//     const int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float

//     // Calculate YaRN correction dimensions
//     float corr_dims[2];
//     rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base,
//                        rope_params->beta_fast, rope_params->beta_slow, corr_dims);

//     if (mode & GGML_ROPE_TYPE_NEOX) {
//         // NeoX Mode: Work on complete heads to handle split pattern
//         const int64_t total_work_units = batch * seq_len * heads;

//         // Distribute work units across threads
//         int64_t units_per_thread = total_work_units / num_threads;
//         int64_t start_unit = thread_id * units_per_thread;
//         int64_t end_unit = (thread_id == num_threads - 1) ?
//                            total_work_units :
//                            start_unit + units_per_thread;

//         const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));

//         const bool is_inplace = (src0_data == dst_data);

//         for (int64_t unit = start_unit; unit < end_unit; unit++) {
//             // Map work unit back to coordinates
//             int64_t h = unit % heads;
//             int64_t s = (unit / heads) % seq_len;
//             int64_t b = unit / (heads * seq_len);

//             const float* head_src = (const float*)((char*)src0_data +
//                 b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

//             float* head_dst = (float*)((char*)dst_data +
//                 b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

//             const int32_t pos = src1_data[s] + rope_params->n_past;

//             // Process cache lines within this head
//             int64_t cachelines_in_head = (head_dim + elements_per_cacheline - 1) / elements_per_cacheline;

//             if (is_inplace) {
//                 // Inplace: Process each cacheline, writing both halves together
//                 for (int64_t cl = 0; cl < cachelines_in_head; cl++) {
//                     float* cacheline = head_dst + cl * elements_per_cacheline;

//                     // Copy source to destination first (for unmodified elements)
//                     for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
//                         cacheline[elem] = head_src[cl * elements_per_cacheline + elem];
//                     }

//                     // Process elements in this cache line
//                     for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
//                         int64_t dim_idx = cl * elements_per_cacheline + elem;

//                         if (dim_idx < n_dims / 2) {
//                             // First half - read before any writes, then write both halves
//                             float x0 = head_src[dim_idx];
//                             float x1 = head_src[dim_idx + n_dims/2];

//                             // Calculate theta for this dimension pair
//                             float theta = 1.0f;
//                             for (int64_t j = 0; j < dim_idx; j++) {
//                                 theta *= theta_scale;
//                             }

//                             const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
//                             const float theta_base = (float)pos * theta;

//                             float cos_theta, sin_theta;
//                             rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2,
//                                      rope_params->ext_factor, rope_params->attn_factor,
//                                      &cos_theta, &sin_theta);

//                             // Write both halves of the pair (safe because we read both first)
//                             head_dst[dim_idx] = x0 * cos_theta - x1 * sin_theta;
//                             head_dst[dim_idx + n_dims/2] = x0 * sin_theta + x1 * cos_theta;
//                         }
//                     }
//                 }
//             } else {
//                 // Non-inplace: Copy all cachelines first, then process rotations
//                 for (int64_t cl = 0; cl < cachelines_in_head; cl++) {
//                     float* cacheline = head_dst + cl * elements_per_cacheline;
//                     for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < head_dim; elem++) {
//                         cacheline[elem] = head_src[cl * elements_per_cacheline + elem];
//                     }
//                 }

//                 // Now process rotations, writing both halves together
//                 for (int64_t dim_idx = 0; dim_idx < n_dims / 2; dim_idx++) {
//                     float x0 = head_dst[dim_idx];
//                     float x1 = head_dst[dim_idx + n_dims/2];

//                     // Calculate theta for this dimension pair
//                     float theta = 1.0f;
//                     for (int64_t j = 0; j < dim_idx; j++) {
//                         theta *= theta_scale;
//                     }

//                     const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;
//                     const float theta_base = (float)pos * theta;

//                     float cos_theta, sin_theta;
//                     rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2,
//                              rope_params->ext_factor, rope_params->attn_factor,
//                              &cos_theta, &sin_theta);

//                     // Write both halves of the pair
//                     head_dst[dim_idx] = x0 * cos_theta - x1 * sin_theta;
//                     head_dst[dim_idx + n_dims/2] = x0 * sin_theta + x1 * cos_theta;
//                 }
//             }
//         }
//     } else {
//         // Standard Mode: Process cache lines directly across all data
//         const int64_t total_elements = batch * seq_len * heads * head_dim;
//         const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

//         // Distribute cache lines across threads
//         int64_t cachelines_per_thread = total_cachelines / num_threads;
//         int64_t start_cacheline = thread_id * cachelines_per_thread;
//         int64_t end_cacheline = (thread_id == num_threads - 1) ?
//                                 total_cachelines :
//                                 start_cacheline + cachelines_per_thread;

//         // Pre-calculate theta_scale
//         const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));

//         for (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
//             float* cacheline_dst = dst_data + cl * elements_per_cacheline;
//             const float* cacheline_src = src0_data + cl * elements_per_cacheline;

//             // Copy source to destination first
//             for (int64_t elem = 0; elem < elements_per_cacheline && (cl * elements_per_cacheline + elem) < total_elements; elem++) {
//                 cacheline_dst[elem] = cacheline_src[elem];
//             }

//             // Process pairs in this cache line
//             const int64_t pairs_per_cacheline = elements_per_cacheline / 2;

//             for (int64_t local_pair = 0; local_pair < pairs_per_cacheline; local_pair++) {
//                 int64_t global_element_idx = cl * elements_per_cacheline + local_pair * 2;

//                 if (global_element_idx + 1 >= total_elements) break;

//                 // Map back to [batch, seq, head, dim] coordinates
//                 int64_t dim_in_head = global_element_idx % head_dim;
//                 int64_t h = (global_element_idx / head_dim) % heads;
//                 int64_t s = (global_element_idx / (head_dim * heads)) % seq_len;
//                 int64_t b = global_element_idx / (head_dim * heads * seq_len);

//                 if (dim_in_head < n_dims && dim_in_head % 2 == 0) {
//                     // Calculate position and theta
//                     const int32_t pos = src1_data[s] + rope_params->n_past;
//                     const int64_t pair_idx = dim_in_head / 2;

//                     float theta = 1.0f;
//                     for (int64_t j = 0; j < pair_idx; j++) {
//                         theta *= theta_scale;
//                     }

//                     const float ff = freq_factors ? freq_factors[pair_idx] : 1.0f;
//                     const float theta_base = (float)pos * theta;

//                     // Apply rotation to this pair using YaRN
//                     float cos_theta, sin_theta;
//                     rope_yarn(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_in_head,
//                              rope_params->ext_factor, rope_params->attn_factor,
//                              &cos_theta, &sin_theta);

//                     float x0 = cacheline_dst[local_pair * 2];
//                     float x1 = cacheline_dst[local_pair * 2 + 1];

//                     cacheline_dst[local_pair * 2]     = x0 * cos_theta - x1 * sin_theta;
//                     cacheline_dst[local_pair * 2 + 1] = x0 * sin_theta + x1 * cos_theta;
//                 }
//             }
//         }
//     }

//     return 0;
// }











//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_IMROPE 40
#define MAX_ROPE_HALF_DIMS 256  // supports up to n_dims=512

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

// Compact work descriptor — avoids putting 4 full ggml_tensor copies (~1400 B)
// on every hart's stack in monolithic mode.  Only the fields the ROPE
// computation actually touches are kept here (~172 B).
typedef struct {
    const float*   src0_data;
    const int32_t* src1_data;
    const float*   freq_factors;  // NULL if not present
    float*         dst_data;
    int64_t        ne[4];         // head_dim, heads, seq_len, batch (from src0)
    int64_t        src0_nb1, src0_nb2, src0_nb3;
    int64_t        dst_nb1, dst_nb2, dst_nb3;
    rope_params_t  rp;
} rope_f32_work_t;

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

//------------------------------------------------------------------------------
// Core computation — works from the compact rope_f32_work_t descriptor.
// Keeps the heavy cos/sin caches on THIS frame only (~2 KB), while the
// caller avoids putting 4 full ggml_tensor copies on the stack (~1.4 KB
// saved in monolithic mode).
//------------------------------------------------------------------------------

static int rope_f32_compute(const rope_f32_work_t* w,
                            int thread_id, int num_threads) {
    const int64_t head_dim = w->ne[0];
    const int64_t heads    = w->ne[1];
    const int64_t seq_len  = w->ne[2];
    const int64_t batch    = w->ne[3];

    const rope_params_t* rope_params = &w->rp;
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
            const int32_t pt = w->src1_data[s]              + rope_params->n_past;
            const int32_t ph = w->src1_data[s + seq_len]    + rope_params->n_past;
            const int32_t pw = w->src1_data[s + seq_len * 2] + rope_params->n_past;
            const int32_t pe = w->src1_data[s + seq_len * 3] + rope_params->n_past;

            if (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(
                    cos_cache, sin_cache,
                    n_dims, theta_scale,
                    pt, ph, pw, pe,
                    rope_params->sections,
                    w->freq_factors, freq_scale,
                    corr_dims, rope_params->ext_factor, rope_params->attn_factor
                );
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } else {
            const int32_t pos = w->src1_data[s] + rope_params->n_past;

            if (pos != last_pos) {
                compute_rope_cache(
                    cos_cache, sin_cache,
                    n_dims, theta_scale, pos,
                    w->freq_factors, freq_scale,
                    corr_dims, rope_params->ext_factor, rope_params->attn_factor
                );
                last_pos = pos;
            }
        }

        const float* head_src = (const float*)((const char*)w->src0_data +
            b * w->src0_nb3 + s * w->src0_nb2 + h * w->src0_nb1);

        float* head_dst = (float*)((char*)w->dst_data +
            b * w->dst_nb3 + s * w->dst_nb2 + h * w->dst_nb1);

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

//------------------------------------------------------------------------------
// Entry point — standalone kernel API (thin wrapper around rope_f32_compute)
//------------------------------------------------------------------------------

#ifdef ENABLE_MONOLITHIC_COMPUTE
#define ROPE_F32_FUNC rope_f32_impl
#else
#define ROPE_F32_FUNC entry_point
#endif

int ROPE_F32_FUNC(struct ggml_et_rope_params* params, void* env) {
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

    if (!src0->data || !src1->data || !dst->data) {
        return -1;
    }

    rope_f32_work_t w;
    w.src0_data    = (const float*)src0->data;
    w.src1_data    = (const int32_t*)src1->data;
    w.freq_factors = (src2 && src2->data) ? (const float*)src2->data : NULL;
    w.dst_data     = (float*)dst->data;
    for (int j = 0; j < 4; j++) w.ne[j] = src0->ne[j];
    w.src0_nb1 = (int64_t)src0->nb[1];
    w.src0_nb2 = (int64_t)src0->nb[2];
    w.src0_nb3 = (int64_t)src0->nb[3];
    w.dst_nb1  = (int64_t)dst->nb[1];
    w.dst_nb2  = (int64_t)dst->nb[2];
    w.dst_nb3  = (int64_t)dst->nb[3];
    w.rp       = params->rope_params;

    return rope_f32_compute(&w, thread_id, num_threads);
}
