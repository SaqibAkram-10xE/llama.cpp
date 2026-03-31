//******************************************************************************
// Element-wise Map F32 Kernel
// Element-wise operations: dst[i] = src0[i] op src1[i]
// Supports: MUL, ADD (more operations to be added later)
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "ggml_tensor.h"
#include "platform.h"

// Host-side ggml op codes (ggml.h) used for serialized node_op[] in graph mode.
// These numeric values must match the host enum, not the device-side ggml_tensor.h enum.
#define HOST_GGML_OP_NONE      0
#define HOST_GGML_OP_ADD       2
#define HOST_GGML_OP_MUL       7
#define HOST_GGML_OP_RMS_NORM  25
#define HOST_GGML_OP_MUL_MAT   29
#define HOST_GGML_OP_CONT      35
#define HOST_GGML_OP_RESHAPE   36
#define HOST_GGML_OP_VIEW      37
#define HOST_GGML_OP_PERMUTE   38
#define HOST_GGML_OP_TRANSPOSE 39
#define HOST_GGML_OP_GET_ROWS  40
#define HOST_GGML_OP_SET_ROWS  42
#define HOST_GGML_OP_SOFT_MAX  46
#define HOST_GGML_OP_ROPE      48
#define HOST_GGML_OP_GLU       94
#define HOST_GGML_OP_MUL_MAT_ID 30

// #include "mul_mat_Q8_0.c"
#include "quants.h"
#include "math_fp.h"
#include "block_ops.h"
// #include "et_backend/utils.h"
// #include "et_backend/esr_defines.h"

// enum ggml_op {
//     GGML_OP_NONE = 0,

//     GGML_OP_DUP = 1,
//     GGML_OP_ADD = 2,
//     GGML_OP_ADD_ID = 3,
//     GGML_OP_ADD1 = 4,
//     GGML_OP_ACC = 5,
//     GGML_OP_SUB = 6,
//     GGML_OP_MUL = 7,
//     GGML_OP_DIV = 8,
//     GGML_OP_SQR = 9,
//     GGML_OP_SQRT = 10,
//     GGML_OP_LOG = 11,
//     GGML_OP_SIN = 12,
//     GGML_OP_COS = 13,
//     GGML_OP_SUM = 14,
//     GGML_OP_SUM_ROWS = 15,
//     GGML_OP_CUMSUM = 16,
//     GGML_OP_MEAN = 17,
//     GGML_OP_ARGMAX = 18,
//     GGML_OP_COUNT_EQUAL = 19,
//     GGML_OP_REPEAT = 20,
//     GGML_OP_REPEAT_BACK = 21,
//     GGML_OP_CONCAT = 22,
//     GGML_OP_SILU_BACK = 23,
//     GGML_OP_NORM = 24,
//     GGML_OP_RMS_NORM = 25,
//     GGML_OP_RMS_NORM_BACK = 26,
//     GGML_OP_GROUP_NORM = 27,
//     GGML_OP_L2_NORM = 28,

//     GGML_OP_MUL_MAT = 28,
//     GGML_OP_MUL_MAT_ID = 29,
//     GGML_OP_OUT_PROD = 30,

//     GGML_OP_SCALE = 31,
//     GGML_OP_SET = 32,
//     GGML_OP_CPY = 33,
//     GGML_OP_CONT = 34,
//     GGML_OP_RESHAPE = 35,
//     GGML_OP_VIEW = 36,
//     GGML_OP_PERMUTE = 37,
//     GGML_OP_TRANSPOSE = 38,
//     GGML_OP_GET_ROWS = 39,
//     GGML_OP_GET_ROWS_BACK = 40,
//     GGML_OP_SET_ROWS = 41,
//     GGML_OP_DIAG = 42,
//     GGML_OP_DIAG_MASK_INF = 43,
//     GGML_OP_DIAG_MASK_ZERO = 44,
//     GGML_OP_SOFT_MAX = 45,
//     GGML_OP_SOFT_MAX_BACK = 46,
//     GGML_OP_ROPE = 47,
//     GGML_OP_ROPE_BACK = 48,
//     GGML_OP_CLAMP = 49,
//     GGML_OP_CONV_TRANSPOSE_1D = 50,
//     GGML_OP_IM2COL = 51,
//     GGML_OP_IM2COL_BACK = 52,
//     GGML_OP_IM2COL_3D = 53,
//     GGML_OP_CONV_2D = 54,
//     GGML_OP_CONV_3D = 55,
//     GGML_OP_CONV_2D_DW = 56,
//     GGML_OP_CONV_TRANSPOSE_2D = 57,
//     GGML_OP_POOL_1D = 58,
//     GGML_OP_POOL_2D = 59,
//     GGML_OP_POOL_2D_BACK = 60,
//     GGML_OP_UPSCALE = 61,
//     GGML_OP_PAD = 62,
//     GGML_OP_PAD_REFLECT_1D = 63,
//     GGML_OP_ROLL = 64,
//     GGML_OP_ARANGE = 65,
//     GGML_OP_TIMESTEP_EMBEDDING = 66,
//     GGML_OP_ARGSORT = 67,
//     GGML_OP_TOP_K = 68,
//     GGML_OP_LEAKY_RELU = 69,
//     GGML_OP_TRI = 70,
//     GGML_OP_FILL = 71,

//     GGML_OP_FLASH_ATTN_EXT = 72,
//     GGML_OP_FLASH_ATTN_BACK = 73,
//     GGML_OP_SSM_CONV = 74,
//     GGML_OP_SSM_SCAN = 75,
//     GGML_OP_WIN_PART = 76,
//     GGML_OP_WIN_UNPART = 77,
//     GGML_OP_GET_REL_POS = 78,
//     GGML_OP_ADD_REL_POS = 79,
//     GGML_OP_RWKV_WKV6 = 80,
//     GGML_OP_GATED_LINEAR_ATTN = 81,
//     GGML_OP_RWKV_WKV7 = 82,
//     GGML_OP_SOLVE_TRI = 83,

//     GGML_OP_UNARY = 84,

//     GGML_OP_MAP_CUSTOM1 = 85,
//     GGML_OP_MAP_CUSTOM2 = 86,
//     GGML_OP_MAP_CUSTOM3 = 87,

//     GGML_OP_CUSTOM = 88,

//     GGML_OP_CROSS_ENTROPY_LOSS = 89,
//     GGML_OP_CROSS_ENTROPY_LOSS_BACK = 90,
//     GGML_OP_OPT_STEP_ADAMW = 91,
//     GGML_OP_OPT_STEP_SGD = 92,

//     GGML_OP_GLU = 93,

//     GGML_OP_COUNT = 94,
// };

// const char * ggml_op_name(enum ggml_op op) {
//     return GGML_OP_NAME[op];
// }

// GLU operation types (from ggml.h)
enum ggml_glu_op {
    GGML_GLU_OP_REGLU = 0,
    GGML_GLU_OP_GEGLU = 1,
    GGML_GLU_OP_SWIGLU = 2,
    GGML_GLU_OP_GEGLU_ERF = 3,
    GGML_GLU_OP_GEGLU_QUICK = 4
};

// Operation identifiers for memops kernel
enum ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Parameter structures for different operations
struct ggml_et_elmap_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

struct ggml_et_rms_norm_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
    float eps;                // Epsilon parameter for numerical stability
};

struct ggml_et_glu_params {
    struct ggml_tensor src0;     // F32 input tensor A (or combined tensor if src1 is null)
    struct ggml_tensor src1;     // F32 input tensor B (null for single tensor mode)
    struct ggml_tensor dst;      // F32 output tensor (n/2 columns)
    int32_t glu_op_type;         // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t swapped;             // Whether gate and value are swapped
};

struct ggml_et_softmax_params {
    struct ggml_tensor src0;     // F32 input tensor
    struct ggml_tensor src1;     // F32 mask tensor (optional, may be zeroed if not used)
    struct ggml_tensor src2;     // F32 sinks tensor (optional, may be zeroed if not used)
    struct ggml_tensor dst;      // F32 output tensor
    float scale;                 // Scale factor (temperature scaling)
    float max_bias;              // Max bias for ALiBi (0.0f if not used)
};

struct ggml_et_get_rows_params {
    struct ggml_tensor src0;     // Data tensor (F32 or Q8_0)
    struct ggml_tensor src1;     // Row indices tensor (I32)
    struct ggml_tensor dst;      // Output tensor (F32)
};

struct ggml_et_set_rows_params {
    struct ggml_tensor src0;     // F32 source data tensor
    struct ggml_tensor src1;     // I64 row indices tensor
    struct ggml_tensor dst;      // F32/F16 destination tensor
};

struct ggml_et_cont_params {
    struct ggml_tensor src0;     // F32 input tensor (non-contiguous)
    struct ggml_tensor dst;      // F32 output tensor (contiguous)
};

struct memset_params {
    uint32_t op_type;      // GGML_ET_MEMOP_MEMSET
    uint32_t value;        // Value to set (extended to uint32_t for alignment)
    void* dst_ptr;         // Destination device pointer
    size_t size;           // Number of bytes to set
};

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

struct ggml_et_rope_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor src1;  // I32 position tensor
    struct ggml_tensor src2;  // F32 frequency factors (optional)
    struct ggml_tensor dst;   // F32 output tensor
    rope_params_t rope_params;
};

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX 2
#define CACHE_LINE_SIZE_F32 16
#define CACHE_LINE_SIZE_BYTES 64
#define CACHE_LINE_F32_ELEMS 16
#define CACHE_LINE_F16_ELEMS 32
#define LOG2E_F 1.4426950408889634f
#define MAX_ROPE_HALF_DIMS 128
#define ROPE_VEC_WIDTH 8
#define ROPE_PI 3.14159265358979323846f
#define ROPE_TWO_PI 6.28318530717958647693f
#define ROPE_PI_OVER_2 1.57079632679489661923f
#define ROPE_INV_TWO_PI 0.15915494309189533577f

// struct ggml_et_binary_params {
//     struct ggml_tensor src0;
//     struct ggml_tensor src1;
//     struct ggml_tensor dst;
// };

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

void delay(unsigned long count) {
    volatile unsigned long i;
    for (i = 0; i < count; i++) {
        // empty
    }
}
#define FENCE __asm__ __volatile__ ("fence\n");

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

static inline void softmax_update_scalar(softmax_params_t * params, float x) {
    if (!softmax_lane_is_valid(x)) {
        return;
    }

    if (!params->valid_mask) {
        params->max_val = x;
        params->sum_val = 1.0f;
        params->valid_mask = 1;
        return;
    }

    if (x > params->max_val) {
        params->sum_val = params->sum_val * et_expf(params->max_val - x) + 1.0f;
        params->max_val = x;
    } else {
        params->sum_val += et_expf(x - params->max_val);
    }
}

static inline void chunk_transform_ps_8_branchless_mask(float * tmp8, const float * src, const float * mask, float scale, float slope) {
    unsigned long ms;
    const float zero = 0.0f;
    const unsigned long mask_load_m0 = (mask != NULL) ? 0xFFul : 0x00ul;
    const float * mp = (mask != NULL) ? mask : &zero;

    __asm__ volatile (
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_scale])   \n\t"
        "fbc.ps    f11, 0(%[p_slope])   \n\t"
        "fbc.ps    f1, 0(%[p_zero])     \n\t"
        "mov.m.x   m0, %[maskm0], 0     \n\t"
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

static inline softmax_params_t softmax_pass1_range(const float * src, const float * mask, int begin, int end, float scale, float slope) {
    __attribute__((aligned(32))) float lane_max[8];
    __attribute__((aligned(32))) float lane_sum[8];
    __attribute__((aligned(32))) float tmp[8];
    uint8_t valid_mask = 0;
    const float one_f = 1.0f;
    const float zero_f = 0.0f;
    const float neg_inf = -INFINITY;
    const float log2e = LOG2E_F;
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
    for (; i + 8 <= end; i += 8) {
        chunk_transform_ps_8_branchless_mask(tmp, src + i, mask ? (mask + i) : NULL, scale, slope);

        uint8_t cur_mask = 0;
        for (int j = 0; j < 8; ++j) {
            if (softmax_lane_is_valid(tmp[j])) {
                cur_mask |= (uint8_t)(1u << j);
            }
        }

        const uint8_t init_mask = (uint8_t)(cur_mask & ~valid_mask);
        const uint8_t upd_mask = (uint8_t)(cur_mask & valid_mask);

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
        if ((valid_mask & (1u << k)) && (out.max_val == -INFINITY || lane_max[k] > out.max_val)) {
            out.max_val = lane_max[k];
        }
    }

    if (out.max_val != -INFINITY) {
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

    for (; i < end; ++i) {
        float x = src[i] * scale;
        if (mask != NULL) {
            x += mask[i] * slope;
        }
        softmax_update_scalar(&out, x);
    }

    return out;
}

static inline void softmax_pass2_range(float * dst, const float * src, const float * mask, int begin, int end, float scale, float slope, softmax_params_t params) {
    const float s2 = scale * LOG2E_F;
    const float sl2 = slope * LOG2E_F;
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

    int c = begin;
    if (mask != NULL) {
        __asm__ volatile (
            "fbc.ps    f11, 0(%[p_sl2]) \n\t"
            :
            : [p_sl2] "r"(&sl2)
            : "f11"
        );

        for (; c + 8 <= end; c += 8) {
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
        for (; c + 8 <= end; c += 8) {
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

    for (; c < end; ++c) {
        float x = src[c] * scale - params.max_val;
        if (mask != NULL) {
            x += mask[c] * slope;
        }
        dst[c] = et_expf(x) * inv_sum;
    }
}

static void compute_softmax_row(float * dst, const float * src, const float * mask, int cols, float scale, float slope, float sink_value, bool use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, mask, 0, cols, scale, slope);

    if (use_sinks) {
        float max_val = params.max_val;
        if (sink_value > max_val) {
            max_val = sink_value;
        }

        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float x = src[i] * scale;
            if (mask != NULL) {
                x += mask[i] * slope;
            }
            sum += et_expf(x - max_val);
        }
        sum += et_expf(sink_value - max_val);

        float inv_sum = et_fdiv(1.0f, sum);
        for (int i = 0; i < cols; ++i) {
            float x = src[i] * scale;
            if (mask != NULL) {
                x += mask[i] * slope;
            }
            dst[i] = et_expf(x - max_val) * inv_sum;
        }
    } else {
        if (!params.valid_mask) {
            return;
        }
        softmax_pass2_range(dst, src, mask, 0, cols, scale, slope, params);
    }
}

// Helper functions from different kernels
static inline float silu_f32(float x) {
    if (x > 20.0f) {
        return x;
    } else if (x < -20.0f) {
        return 0.0f;
    } else {
        float exp_neg_x = et_expf(-x);
        float denominator = 1.0f + exp_neg_x;
        return et_fdiv(x, denominator);
    }
}

static inline float gelu_f32(float x) {
    const float coef_a_const = 0.044715f;
    const float sqrt2pi_const = 0.79788456080286535587989211986876f;
    const float z = sqrt2pi_const * x * (1.0f + coef_a_const * x * x);
    const float exp_2z = et_expf(2.0f * z);
    return x * (1.0f - et_fdiv(1.0f, exp_2z + 1.0f));
}

static void copy_f32_row(float* dst, const float* src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = src[i];
    }
}

static void copy_q8_0_row(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;

        float temp_buffer[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

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

static void copy_row_cache_align(float* dst, const float* src, int64_t n_bytes) {
    int num_f32_elem = n_bytes / sizeof(float);

    __asm__ volatile (
        "1: \n\t"
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"
        "addi %[src], %[src], 64 \n\t"
        "addi %[dst], %[dst], 64 \n\t"
        "addi %[n], %[n], -16 \n\t"
        "bge %[n], %[stride_count], 1b \n\t"
        : [dst] "+r" (dst), [src] "+r" (src), [n] "+r" (num_f32_elem)
        : [stride_count] "r" (16L)
        : "f0", "f1", "memory"
    );
}

static void dequantize_q8_0_block_cache_aligned(const block_q8_0* block, float* dst) {
    const int8_t* qs_ptr = block->qs;

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    const int32_t __attribute__((aligned(32))) vec_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float scale = fp16_to_fp32(block->d);
    __asm__ volatile (
        "fbcx.ps     f0, %0       \n\t"
        "flq2        f1, 0(%1)    \n\t"
        :: "r"(scale), "r"(vec_indices)
        : "f0", "f1"
    );

    for (int i = 0; i < 4; i++) {
        __asm__ volatile (
            "fgb.ps      f2, f1(%0)   \n\t"
            "fcvt.ps.pw  f2, f2, rne  \n\t"
            "fmul.ps     f2, f2, f0   \n\t"
            "fsq2        f2, 0(%1)    \n\t"
            :: "r"(qs_ptr), "r"(dst)
            : "f2", "memory"
        );
        qs_ptr += 8;
        dst += 8;
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static void copy_q8_0_row_cache_aligned(float* dst, const block_q8_0* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK8_0 - 1) / QK8_0;

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int64_t elements_in_block = (block_idx == num_blocks - 1) ?
            (num_elements - block_idx * QK8_0) : QK8_0;

        float temp_buffer[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], temp_buffer);

        for (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

static void copy_q4_0_row_cache_aligned(float* dst, const block_q4_0* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK4_0 - 1) / QK4_0;

    const int32_t __attribute__((aligned(32))) scatter_offsets[8] = {
        0*4, 16*4, 1*4, 17*4, 2*4, 18*4, 3*4, 19*4
    };
    const int32_t __attribute__((aligned(32))) gather_indices[8] = {0, 0, 1, 1, 2, 2, 3, 3};

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    __asm__ volatile (
        "flq2        f4, 0(%0)    \n\t"
        "flq2        f1, 0(%1)    \n\t"
        :: "r"(scatter_offsets), "r"(gather_indices)
        : "f1", "f4"
    );

    for (int64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_0* block = &src_blocks[block_idx];
        const uint8_t* qs = block->qs;
        float* block_dst = dst + block_idx * QK4_0;

        float scale = fp16_to_fp32(block->d);
        float bias = -8.0f * scale;

        __asm__ volatile (
            "fbcx.ps     f0, %0       \n\t"
            "fbcx.ps     f3, %1       \n\t"
            :: "r"(scale), "r"(bias)
            : "f0", "f3"
        );

        for (int i = 0; i < 4; i++) {
            __asm__ volatile (
                "fgb.ps      f2, f1(%0)    \n\t"
                "mov.m.x     m0, x0, 0xAA  \n\t"
                "fsrli.pi    f2, f2, 4     \n\t"
                "mov.m.x     m0, x0, 0xFF  \n\t"
                "fslli.pi    f2, f2, 28    \n\t"
                "fsrli.pi    f2, f2, 28    \n\t"
                "fcvt.ps.pw  f2, f2, rne   \n\t"
                "fmul.ps     f2, f2, f0    \n\t"
                "fadd.ps     f2, f2, f3    \n\t"
                "fscw.ps     f2, f4(%1)    \n\t"
                :: "r"(qs), "r"(block_dst)
                : "f2", "memory"
            );
            qs += 4;
            block_dst += 4;
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static void copy_q4_K_row_cache_aligned(float* dst, const block_q4_K* src_blocks, int64_t num_elements) {
    const int64_t num_blocks = (num_elements + QK_K - 1) / QK_K;

    const int32_t __attribute__((aligned(32))) gather_indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    __asm__ volatile (
        "flq2        f1, 0(%0)    \n\t"
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
            uint8_t sc, m;
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            const float d1 = d * sc;
            const float neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            const float d2 = d * sc;
            const float neg_m2 = -(min * m);

            __asm__ volatile (
                "fbcx.ps     f0, %0       \n\t"
                "fbcx.ps     f3, %1       \n\t"
                :: "r"(d1), "r"(neg_m1)
                : "f0", "f3"
            );

            const uint8_t* qs_lo = qs;
            float* dst_lo = block_dst + j;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile (
                    "fgb.ps      f2, f1(%0)   \n\t"
                    "fandi.pi    f2, f2, 0xF   \n\t"
                    "fcvt.ps.pw  f2, f2, rne   \n\t"
                    "fmadd.ps    f2, f2, f0, f3\n\t"
                    "fsq2        f2, 0(%1)     \n\t"
                    :: "r"(qs_lo), "r"(dst_lo)
                    : "f2", "memory"
                );
                qs_lo += 8;
                dst_lo += 8;
            }

            __asm__ volatile (
                "fbcx.ps     f0, %0       \n\t"
                "fbcx.ps     f3, %1       \n\t"
                :: "r"(d2), "r"(neg_m2)
                : "f0", "f3"
            );

            const uint8_t* qs_hi = qs;
            float* dst_hi = block_dst + j + 32;
            for (int k = 0; k < 4; k++) {
                __asm__ volatile (
                    "fgb.ps      f2, f1(%0)   \n\t"
                    "fsrli.pi    f2, f2, 4     \n\t"
                    "fandi.pi    f2, f2, 0xF   \n\t"
                    "fcvt.ps.pw  f2, f2, rne   \n\t"
                    "fmadd.ps    f2, f2, f0, f3\n\t"
                    "fsq2        f2, 0(%1)     \n\t"
                    :: "r"(qs_hi), "r"(dst_hi)
                    : "f2", "memory"
                );
                qs_hi += 8;
                dst_hi += 8;
            }

            qs += 32;
            is += 2;
        }
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static int get_row_f32_mc_row_cache_aligned(struct ggml_et_get_rows_params* params, void* env)
{
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    for (int64_t i = thread_id; i < total_rows_to_extract; i+=num_threads) {
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        void* src0_data = src0->data;
        int32_t* src1_data = (int32_t*)src1->data;
        float* dst_data = (float*)dst->data;
        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                    i12_idx * ne11 * ne10 +
                                    i11_idx * ne10 +
                                    i10_idx;
        const int32_t row_index = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) {
            return -1;
        }

        const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                     i12_idx * ne02 * ne01 * ne00 +
                                     i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t dst_offset = i;

        if (src0->type == GGML_TYPE_F32) {
            const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_row_cache_align(dst_row, src_row, ne00 * sizeof(float));
        }
        else if (src0->type == GGML_TYPE_Q8_0) {
            const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, ne00);
        }
        else if (src0->type == GGML_TYPE_Q4_0) {
            const int64_t blocks_per_row = (ne00 + QK4_0 - 1) / QK4_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q4_0* src_blocks = (const block_q4_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q4_0_row_cache_aligned(dst_row, src_blocks, ne00);
        }
        else if (src0->type == GGML_TYPE_Q4_K) {
            const int64_t blocks_per_row = (ne00 + QK_K - 1) / QK_K;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q4_K* src_blocks = (const block_q4_K*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q4_K_row_cache_aligned(dst_row, src_blocks, ne00);
        }
    }

    return 0;
}

static void copy_f32_to_f16_row(uint16_t* dst, const float* src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

static void copy_cache_aligned_f32(float* dst, const float* src) {
    __asm__ volatile (
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"
        :
        : [src] "r"(src), [dst] "r"(dst)
        : "f0", "f1", "memory"
    );
}

static void copy_cache_aligned_f16(uint16_t* dst, const float* src) {
    unsigned long mask_temp;
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

// YaRN helper functions
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
    float end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (float)(n_dims - 1) ? end : (float)(n_dims - 1);
}

static const float rope_ps_one[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
};

static const float rope_ps_c3[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f, 1.0f/6.0f
};

static const float rope_ps_c5[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f, 1.0f/120.0f
};

static const float rope_ps_c7[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f, 1.0f/5040.0f
};

static const float rope_ps_c9[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f, 1.0f/362880.0f
};

static const float rope_ps_c11[ROPE_VEC_WIDTH] __attribute__((aligned(32))) = {
    1.0f/39916800.0f, 1.0f/39916800.0f, 1.0f/39916800.0f, 1.0f/39916800.0f,
    1.0f/39916800.0f, 1.0f/39916800.0f, 1.0f/39916800.0f, 1.0f/39916800.0f
};

static inline uint64_t rope_ps_enter_fullmask(void) {
    uint64_t old_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(old_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    return old_mask;
}

static inline void rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ volatile("mova.m.x %0" :: "r"(old_mask) : "memory");
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
        : "f0", "f1", "f2", "f3", "f4", "memory"
    );
}

static inline void rope_sincos_block8(float * sin8, float * cos8, const float * theta8) {
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
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory"
        );

        rope_ps_leave_fullmask(saved_mask);
    }
}

static inline void rope_yarn_scalar(float theta_extrap, float freq_scale, const float corr_dims[2],
                                    int64_t i0, float ext_factor, float mscale,
                                    float * cos_theta, float * sin_theta) {
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

static inline void compute_rope_cache(float * cos_cache, float * sin_cache,
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

static inline void block_mul(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t i = 0;
    unsigned long original_mask;
    
    // 1. Save current mask: mova.x.m is the correct instruction to read the mask into a register
    __asm__ volatile("mova.x.m %0" : "=r"(original_mask));

    // 2. Process main blocks of 8 elements
    if (elements >= 8) {
        int32_t vec_end = (elements / 8) * 8;
        // Use an immediate for 0xFF
        __asm__ volatile("mov.m.x m0, x0, 0xFF"); 

        for (; i < vec_end; i += 8) {
            __asm__ volatile(
                "flw.ps f10, %1\n"
                "flw.ps f11, %2\n"
                "fmul.ps f12, f10, f11\n"
                "fsw.ps f12, %0\n"
                : "=m"(*(float(*)[8])&dst_block[i])
                : "m"(*(const float(*)[8])&src0_block[i]),
                  "m"(*(const float(*)[8])&src1_block[i])
                : "f10", "f11", "f12", "memory"
            );
        }
    }

    // 3. Handle Tail Elements (1 to 7)
    int32_t rem = elements - i;
    if (rem > 0) {
        uint32_t tail_mask = (1U << rem) - 1;
        // Correct instruction to move a register value into mask m0
        __asm__ volatile(
            "mov.m.x m0, %0, 0" 
            : 
            : "r"(tail_mask)
        );

        __asm__ volatile(
            "flw.ps f10, %1\n"
            "flw.ps f11, %2\n"
            "fmul.ps f12, f10, f11\n"
            "fsw.ps f12, %0\n"
            : "=m"(*(float(*)[8])&dst_block[i])
            : "m"(*(const float(*)[8])&src0_block[i]),
              "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12", "memory"
        );
    }

    // 4. Restore original mask using mov.m.x (not mova.m.x)
    __asm__ volatile("mov.m.x m0, %0, 0" :: "r"(original_mask));
}

static inline void block_add(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t i = 0;
    unsigned long original_mask;
    
    // Save current mask
    __asm__ volatile("mova.x.m %0" : "=r"(original_mask));

    // 1. Process main blocks of 8 elements
    if (elements >= 8) {
        int32_t vec_end = (elements / 8) * 8;
        // Use an immediate for 0xFF
        __asm__ volatile("mov.m.x m0, x0, 0xFF"); 

        for (; i < vec_end; i += 8) {
            __asm__ volatile(
                "flw.ps f10, %1\n"
                "flw.ps f11, %2\n"
                "fadd.ps f12, f10, f11\n"
                "fsw.ps f12, %0\n"
                : "=m"(*(float(*)[8])&dst_block[i])
                : "m"(*(const float(*)[8])&src0_block[i]),
                  "m"(*(const float(*)[8])&src1_block[i])
                : "f10", "f11", "f12", "memory"
            );
        }
    }

    // 2. Handle Tail Elements (1 to 7)
    int32_t rem = elements - i;
    if (rem > 0) {
        uint32_t tail_mask = (1U << rem) - 1;
        // Correct instruction to move a register value into mask m0
        __asm__ volatile(
            "mov.m.x m0, %0, 0" 
            : 
            : "r"(tail_mask)
        );

        __asm__ volatile(
            "flw.ps f10, %1\n"
            "flw.ps f11, %2\n"
            "fadd.ps f12, f10, f11\n"
            "fsw.ps f12, %0\n"
            : "=m"(*(float(*)[8])&dst_block[i])
            : "m"(*(const float(*)[8])&src0_block[i]),
              "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12", "memory"
        );
    }

    // Restore original mask
    __asm__ volatile("mov.m.x m0, %0, 0" :: "r"(original_mask));
}

static inline void block_geglu(float* dst_block, const float* x_block, const float* g_block, int elements) {
    int32_t vec_end = (elements / 8) * 8;
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    float one_const = 1.0f;
    float coef_a_const = 0.044715f;
    float sqrt2pi_const = 0.79788456080286535587989211986876f;
    float two_log2e_const = 2.8853900817779268f;

    for (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[x_vec]\n"
            "flw.ps f11, %[g_vec]\n"
            "fbc.ps f20, %[one_ptr]\n"
            "fbc.ps f22, %[coef_ptr]\n"
            "fbc.ps f23, %[sqrt2pi_ptr]\n"
            "fbc.ps f24, %[two_log2e_ptr]\n"
            "fmul.ps f12, f10, f10\n"
            "fmadd.ps f13, f22, f12, f20\n"
            "fmul.ps f14, f23, f10\n"
            "fmul.ps f14, f14, f13\n"
            "fmul.ps f15, f14, f24\n"
            "fexp.ps f15, f15\n"
            "fadd.ps f16, f15, f20\n"
            "frcp.ps f16, f16\n"
            "fsub.ps f16, f20, f16\n"
            "fmul.ps f16, f10, f16\n"
            "fmul.ps f18, f16, f11\n"
            "fsw.ps f18, %[dst_out]\n"
            : [dst_out] "=m"(*(float(*)[8])&dst_block[i])
            : [x_vec] "m"(*(const float(*)[8])&x_block[i]),
              [g_vec] "m"(*(const float(*)[8])&g_block[i]),
              [one_ptr] "m"(one_const),
              [coef_ptr] "m"(coef_a_const),
              [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18",
              "f20", "f22", "f23", "f24", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    for (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = gelu_f32(x_block[i]) * g_block[i];
    }
}

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
              [zero_ptr] "m"(zero_const),
              [one_ptr] "m"(one_const),
              [log2e_ptr] "m"(log2e_const)
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

// KERNEL_TRAMPOLINE();

// Individual operation implementations
int rms_norm_f32_impl(struct ggml_et_rms_norm_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* dst = &params->dst;
    float eps = params->eps;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;
    if (!src0_data || !dst_data) return -1;

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    if (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) return -1;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {
                const float* src_ptr = (const float*)((const char*)src0_data + i3*nb03 + i2*nb02 + i1*nb01);
                float* dst_ptr = (float*)((char*)dst_data + i3*nb3 + i2*nb2 + i1*nb1);

                float sum = 0.0f;
                int32_t vec_end = (int32_t)((ne0 / 8) * 8);

                if (vec_end > 0) {
                    float zero = 0.0f;
                    __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    for (int32_t i0 = 0; i0 < vec_end; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                            : "f10", "f11"
                        );
                    }

                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f" (sum)
                        :
                        : "t0", "f1", "f2", "f3", "f4", "f5"
                    );
                }

                for (int32_t i0 = vec_end; i0 < (int32_t)ne0; i0++) {
                    const float x = src_ptr[i0];
                    sum += x * x;
                }

                const float mean = et_fdiv(sum, (float)(int32_t)ne0);
                const float scale = et_powf(mean + eps, -0.5f);

                if (!(scale > 0.0f)) return -1;

                for (int32_t i0 = 0; i0 < vec_end; i0 += 8) {
                    __asm__ volatile(
                        "flw.ps f12, %[x_vec]\n"
                        "fbc.ps f13, %[scale_ptr]\n"
                        "fmul.ps f14, f12, f13\n"
                        "fsw.ps f14, %[result]\n"
                        : [result] "=m"(*(float(*)[8])&dst_ptr[i0])
                        : [x_vec] "m"(*(const float(*)[8])&src_ptr[i0]),
                          [scale_ptr] "m"(scale)
                        : "f12", "f13", "f14"
                    );
                }

                for (int32_t i0 = vec_end; i0 < (int32_t)ne0; i0++) {
                    dst_ptr[i0] = src_ptr[i0] * scale;
                }
            }
        }
    }

    return 0;
}

int glu_f32_impl(struct ggml_et_glu_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) return 0;
    if (thread_id != 0) return 0; // Single-threaded for now

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;
    if (params->glu_op_type != GGML_GLU_OP_SWIGLU &&
        params->glu_op_type != GGML_GLU_OP_GEGLU) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;
    int32_t swapped = params->swapped;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;
    if (src1 && src1->type != GGML_TYPE_F32) return -1;

    float* src0_data = (float*)src0->data;
    float* src1_data = src1 ? (float*)src1->data : src0_data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !dst_data) return -1;

    const int64_t nc = dst->ne[0];
    const int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];

    const size_t src0_stride = src0->nb[1];
    const size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];
    const size_t dst_stride = dst->nb[1];

    if (src1) {
        if (src0->ne[0] != nc || src1->ne[0] != nc) return -1;
    } else {
        if (src0->ne[0] != 2 * nc) return -1;
    }

    const int64_t elements_per_cacheline = 16;
    const int64_t total_elements = nr * nc;
    const int64_t total_cachelines = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    int64_t start_cacheline = 0;
    int64_t end_cacheline = total_cachelines;

    for (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
        int64_t global_element_start = cl * elements_per_cacheline;
        int64_t row = global_element_start / nc;
        int64_t col = global_element_start % nc;

        if (global_element_start >= total_elements) break;

        int64_t elements_remaining = total_elements - global_element_start;
        int elements_this_block = (int)((elements_remaining < elements_per_cacheline) ?
                                       elements_remaining : elements_per_cacheline);

        int64_t elements_processed = 0;
        while (elements_processed < elements_this_block && row < nr) {
            int64_t elements_in_row = nc - col;
            int64_t elements_to_process = elements_this_block - elements_processed;
            if (elements_to_process > elements_in_row) {
                elements_to_process = elements_in_row;
            }

            float* dst_ptr = (float*)((char*)dst_data + row * dst_stride) + col;

            float* x_ptr;
            float* g_ptr;

            if (src1) {
                x_ptr = (float*)((char*)src0_data + row * src0_stride) + col;
                g_ptr = (float*)((char*)src1_data + row * src1_stride) + col;
            } else {
                float* src0_row = (float*)((char*)src0_data + row * src0_stride);
                if (swapped) {
                    g_ptr = src0_row + col;
                    x_ptr = src0_row + nc + col;
                } else {
                    x_ptr = src0_row + col;
                    g_ptr = src0_row + nc + col;
                }
            }

            if (params->glu_op_type == GGML_GLU_OP_GEGLU) {
                block_geglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);
            } else {
                block_swiglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);
            }

            elements_processed += elements_to_process;
            col += elements_to_process;

            if (col >= nc) {
                row++;
                col = 0;
            }
        }
    }

    return 0;
}

int softmax_f32_impl(struct ggml_et_softmax_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* src2 = &params->src2;
    struct ggml_tensor* dst = &params->dst;
    float scale = params->scale;
    float max_bias = params->max_bias;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    bool use_mask = (src1->data != NULL && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));
    bool use_sinks = (src2->data != NULL && src2->type == GGML_TYPE_F32);

    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;
    float* mask_data = use_mask ? (float*)src1->data : NULL;
    float* sinks_data = use_sinks ? (float*)src2->data : NULL;

    if (!src0_data || !dst_data) return -1;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = use_mask ? src1->ne[0] : 0;
    const int64_t ne11 = use_mask ? src1->ne[1] : 0;
    const int64_t ne12 = use_mask ? src1->ne[2] : 0;
    const int64_t ne13 = use_mask ? src1->ne[3] : 0;

    if (use_mask) {
        if (ne10 != ne00 || ne11 < ne01 || 
            (ne12 > 0 && ne02 % ne12 != 0) || 
            (ne13 > 0 && ne03 % ne13 != 0)) {
            return -1;
        }
    }

    const uint32_t n_head = (uint32_t)ne02;
    uint32_t n_head_log2 = 0;
    float m0 = 1.0f;
    float m1 = 1.0f;

    if (max_bias > 0.0f) {
        n_head_log2 = 1;
        while (n_head_log2 < n_head) {
            n_head_log2 <<= 1;
        }
        if (n_head_log2 > n_head) {
            n_head_log2 >>= 1;
        }

        float inv_n_head_log2 = et_fdiv(1.0f, (float)n_head_log2);
        m0 = et_expf(-max_bias * 0.69314718f * inv_n_head_log2);
        m1 = et_expf(-max_bias * 0.69314718f * inv_n_head_log2 * 0.5f);
    }

    const int64_t rows_per_i03 = ne02 * ne01;
    const int64_t total_rows = ne03 * rows_per_i03;

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        const int64_t i03 = row / rows_per_i03;
        const int64_t rem = row % rows_per_i03;
        const int64_t i02 = rem / ne01;
        const int64_t i01 = rem % ne01;

        float slope = 1.0f;
        if (max_bias > 0.0f) {
            const uint32_t h = (uint32_t)i02;
            if (h < n_head_log2) {
                slope = m0;
                for (uint32_t i = 0; i < h; i++) {
                    slope *= m0;
                }
            } else {
                const uint32_t exp = 2 * (h - n_head_log2) + 1;
                slope = m1;
                for (uint32_t i = 1; i < exp; i++) {
                    slope *= m1;
                }
            }
        }

        float sink_value = 0.0f;
        if (use_sinks && sinks_data) {
            sink_value = sinks_data[i02];
        }

        const int64_t src_offset = i03 * ne02 * ne01 * ne00 +
                                  i02 * ne01 * ne00 +
                                  i01 * ne00;

        const float* src_row = src0_data + src_offset;
        float* dst_row = dst_data + src_offset;
        const float* mask_row = NULL;

        if (use_mask && mask_data) {
            const int64_t mask_i03 = (ne13 > 0) ? i03 % ne13 : 0;
            const int64_t mask_i02 = (ne12 > 0) ? i02 % ne12 : 0;
            const int64_t mask_i01 = i01;

            const int64_t mask_offset = mask_i03 * ne12 * ne11 * ne10 +
                                       mask_i02 * ne11 * ne10 +
                                       mask_i01 * ne10;

            mask_row = mask_data + mask_offset;
        }

        compute_softmax_row(dst_row, src_row, mask_row, (int)ne00, scale, slope, sink_value, use_sinks);
    }

    return 0;
}

int get_rows_f32_impl(struct ggml_et_get_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    // Fast path - we know how to deal with them multi-core
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32
        && dst->ne[0] % (64 / sizeof(float)) == 0) {
        return get_row_f32_mc_row_cache_aligned(params, env);
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) return 0;
    if (thread_id != 0) return 0; // Single-threaded for now

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) return -1;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) return -1;

    void* src0_data = src0->data;
    int32_t* src1_data = (int32_t*)src1->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) return -1;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    for (int64_t i = 0; i < total_rows_to_extract; i++) {
        const int64_t i13_idx = i / (ne12 * ne11 * ne10);
        const int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        const int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        const int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        const int64_t index_offset = i13_idx * ne12 * ne11 * ne10 +
                                    i12_idx * ne11 * ne10 +
                                    i11_idx * ne10 +
                                    i10_idx;
        const int32_t row_index = src1_data[index_offset];

        if (row_index < 0 || row_index >= ne01) return -1;

        const int64_t batch_offset = i11_idx * ne01 * ne00 +
                                     i12_idx * ne02 * ne01 * ne00 +
                                     i13_idx * ne03 * ne02 * ne01 * ne00;

        const int64_t dst_offset = i;

        if (src0->type == GGML_TYPE_F32) {
            const float* src_row = (const float*)src0_data + row_index * ne00 + batch_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_f32_row(dst_row, src_row, ne00);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            const int64_t blocks_per_row = (ne00 + QK8_0 - 1) / QK8_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q8_0* src_blocks = (const block_q8_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q8_0_row(dst_row, src_blocks, ne00);
        } else if (src0->type == GGML_TYPE_Q4_0) {
            const int64_t blocks_per_row = (ne00 + QK4_0 - 1) / QK4_0;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q4_0* src_blocks = (const block_q4_0*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q4_0_row(dst_row, src_blocks, ne00);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            const int64_t blocks_per_row = (ne00 + QK_K - 1) / QK_K;
            const int64_t src_block_offset = (row_index * blocks_per_row) +
                                           (batch_offset / ne00) * blocks_per_row;
            const block_q4_K* src_blocks = (const block_q4_K*)src0_data + src_block_offset;
            float* dst_row = dst_data + dst_offset * ne00;
            copy_q4_K_row(dst_row, src_blocks, ne00);
        }
    }

    return 0;
}

int set_rows_f32_impl(struct ggml_et_set_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) return -1;
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) return -1;

    float* src0_data = (float*)src0->data;
    int64_t* src1_data = (int64_t*)src1->data;
    void* dst_data = dst->data;

    if (!src0_data || !src1_data || !dst_data) return -1;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    const int64_t ne_dst1 = dst->ne[1];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    if (ne10 != ne01) return -1;

    const int64_t total_rows = ne01 * ne02 * ne03;
    const int64_t dst_cl_elems = (dst->type == GGML_TYPE_F16) ? CACHE_LINE_F16_ELEMS : CACHE_LINE_F32_ELEMS;
    const bool row_cache_aligned = (ne00 >= dst_cl_elems) && (ne00 % dst_cl_elems == 0);

    if (row_cache_aligned) {
        const int64_t cls_per_row = ne00 / dst_cl_elems;
        const int64_t total_cls = total_rows * cls_per_row;
        const int64_t cls_per_thread = (total_cls + num_threads - 1) / num_threads;
        const int64_t my_start = thread_id * cls_per_thread;
        int64_t my_end = my_start + cls_per_thread;
        if (my_end > total_cls) my_end = total_cls;
        if (my_start >= total_cls) return 0;

        for (int64_t cl = my_start; cl < my_end; cl++) {
            const int64_t row_flat = cl / cls_per_row;
            const int64_t cl_in_row = cl % cls_per_row;
            const int64_t i01 = row_flat % ne01;
            const int64_t tmp = row_flat / ne01;
            const int64_t i02 = tmp % ne02;
            const int64_t i03 = tmp / ne02;
            const int64_t i12 = i03 % ne12;
            const int64_t i11 = i02 % ne11;
            const int64_t i10 = i01;
            const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
            const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

            if (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                return -1;
            }

            const int64_t elem_offset = cl_in_row * dst_cl_elems;
            const float* src_ptr = (const float*)((char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03) + elem_offset;
            char* dst_row_base = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

            if (dst->type == GGML_TYPE_F32) {
                float* dst_ptr = (float*)dst_row_base + elem_offset;
                copy_cache_aligned_f32(dst_ptr, src_ptr);
            } else {
                uint16_t* dst_ptr = (uint16_t*)dst_row_base + elem_offset;
                copy_cache_aligned_f16(dst_ptr, src_ptr);
            }
        }
    } else {
        for (int64_t row_flat = thread_id; row_flat < total_rows; row_flat += num_threads) {
            const int64_t i01 = row_flat % ne01;
            const int64_t tmp = row_flat / ne01;
            const int64_t i02 = tmp % ne02;
            const int64_t i03 = tmp / ne02;
            const int64_t i12 = i03 % ne12;
            const int64_t i11 = i02 % ne11;
            const int64_t i10 = i01;
            const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
            const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

            if (dst_row_index < 0 || dst_row_index >= ne_dst1) {
                return -1;
            }

            const float* src_row = (const float*)((char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03);
            char* dst_row_base = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

            if (dst->type == GGML_TYPE_F32) {
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

    return 0;
}

int cont_f32_impl(struct ggml_et_cont_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    float* src0_data = (float*)src0->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !dst_data) return -1;

    const int64_t src_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t dst_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (src_elements != dst_elements) return -1;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t total_rows = ne01;
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    if (start_row >= total_rows) return 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const int64_t dst_linear_base = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00;

            for (int64_t i01 = start_row; i01 < end_row; i01++) {
                const int64_t dst_linear_row_base = dst_linear_base + i01 * ne00;

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const int64_t src_offset_bytes = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                    const float* src_ptr = (const float*)((const char*)src0_data + src_offset_bytes);

                    const int64_t dst_linear_idx = dst_linear_row_base + i00;

                    atomic_store_f32((volatile float*)&dst_data[dst_linear_idx], *src_ptr);
                }
            }
        }
    }

    return 0;
}

int memops_impl(struct memset_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id != 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;
    if (params->op_type != GGML_ET_MEMOP_MEMSET) return -1;

    uint8_t* dst = (uint8_t*)params->dst_ptr;
    uint8_t value = (uint8_t)params->value;
    size_t size = params->size;

    if (!dst || size == 0) return -1;

    while (size > 0 && ((uint64_t)dst & 0x7) != 0) {
        *dst++ = value;
        size--;
    }

    uint64_t pattern = value;
    pattern |= pattern << 8;
    pattern |= pattern << 16;
    pattern |= pattern << 32;

    uint64_t* dst64 = (uint64_t*)dst;
    while (size >= 8) {
        *dst64++ = pattern;
        size -= 8;
    }

    dst = (uint8_t*)dst64;
    while (size > 0) {
        *dst++ = value;
        size--;
    }

    return 0;
}

int rope_f32_impl(struct ggml_et_rope_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return -1;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* src2 = &params->src2;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) return -1;

    const float* src0_data = (const float*)src0->data;
    const int32_t* src1_data = (const int32_t*)src1->data;
    const float* freq_factors = NULL;
    if (src2 && src2->data) {
        freq_factors = (const float*)src2->data;
    }
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) return -1;

    const int64_t head_dim = src0->ne[0];
    const int64_t heads = src0->ne[1];
    const int64_t seq_len = src0->ne[2];
    const int64_t batch = src0->ne[3];

    const rope_params_t* rope_params = &params->rope_params;
    const int32_t n_dims = rope_params->n_dims;
    const float freq_base = rope_params->freq_base;
    const float freq_scale = rope_params->freq_scale;
    const int32_t mode = rope_params->mode;

    if (n_dims <= 0 || n_dims > head_dim || n_dims % 2 != 0) return -1;

    if (n_dims / 2 > MAX_ROPE_HALF_DIMS) return -1;

    float cos_cache[MAX_ROPE_HALF_DIMS];
    float sin_cache[MAX_ROPE_HALF_DIMS];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base,
                       rope_params->beta_fast, rope_params->beta_slow, corr_dims);

    const int64_t total_heads = batch * seq_len * heads;
    const int64_t start_wu = (total_heads * thread_id) / num_threads;
    const int64_t end_wu = (total_heads * (thread_id + 1)) / num_threads;

    if (start_wu >= end_wu) return 0;

    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));
    const int32_t half_dims = n_dims / 2;
    const int is_neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    int32_t last_pos = -1;

    for (int64_t wu = start_wu; wu < end_wu; ++wu) {
        const int64_t h = wu % heads;
        const int64_t s = (wu / heads) % seq_len;
        const int64_t b = wu / (heads * seq_len);
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

        const float* head_src = (const float*)((const char*)src0_data +
            b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        float* head_dst = (float*)((char*)dst_data +
            b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        for (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        if (is_neox) {
            uint64_t temp_mask;
            __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
            __asm__ volatile("mov.m.x m0, x0, 0xFF");

            int32_t dim_idx = 0;
            for (; dim_idx + 8 <= half_dims; dim_idx += 8) {
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

            for (; dim_idx < half_dims; ++dim_idx) {
                const float x0 = head_src[dim_idx];
                const float x1 = head_src[dim_idx + half_dims];
                head_dst[dim_idx] = x0 * cos_cache[dim_idx] - x1 * sin_cache[dim_idx];
                head_dst[dim_idx + half_dims] = x0 * sin_cache[dim_idx] + x1 * cos_cache[dim_idx];
            }
        } else {
            for (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                const int32_t dim_in_head = pair_idx * 2;
                const float x0 = head_src[dim_in_head];
                const float x1 = head_src[dim_in_head + 1];

                head_dst[dim_in_head] = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    return 0;
}

// Helper function to compute GCD for cache line alignment
static inline int64_t gcd_i64(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

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


int el_map_f32(struct ggml_et_elmap_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination
    }

    float* src0_data = (float*)src0->data;
    float* src1_data = (float*)src1->data;
    float* dst_data = (float*)dst->data;

    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    enum ggml_op operation = dst->op;

    if (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) {
        return -1; // Unsupported operation
    }

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Calculate total number of rows (flatten dimensions 1,2,3)
    const int64_t total_rows = ne1 * ne2 * ne3;

    // Distribute rows across threads using ceiling division to handle remainder
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    if (start_row >= total_rows) {
        return 0;
    }

    bool cache_aligned = (dst->ne[0] % 16 == 0);
    if(!cache_aligned) {
        return 1;
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
        float* dst_ptr = (float*)((char*)dst_data + i03*nb3 + i02*nb2 + i01*nb1);
        const float* src0_ptr = (const float*)((const char*)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
        const float* src1_ptr = (const float*)((const char*)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

        // Broadcasting in dimension 0: src1 repeats across src0
        const int64_t nr0 = ne0 / ne10;  // How many times src1 is repeated in dimension 0

        for (int64_t r = 0; r < nr0; r++) {
            // Process ne10 elements at a time using block functions
            const float* src0_block = src0_ptr + r * ne10;
            float* dst_block = dst_ptr + r * ne10;

            switch (operation) {
                case GGML_OP_MUL:
                    block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                    break;
                case GGML_OP_ADD:
                    block_add_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                    break;
                case GGML_OP_SUB:
                    block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10);
                    break;
                default:
                    return 1;
            }
        }
    }

    return 0;
}

int mul_mat_f16(struct ggml_et_binary_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;
    
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1;
    }

    // Thread coordination
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0 || (thread_id & 1)) {
        return 0; // Skip odd threads to avoid resource contention
    }

    int effective_thread_id = thread_id / 2;
    int effective_num_threads = (num_threads + 1) / 2;

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A (F16)
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B (F32)
    struct ggml_tensor* dst  = &params->dst;  // Output matrix C (F32)

    // Strictly validate: src0 is F16, others are F32
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; 
    }

    const uint16_t* src0_data = (const uint16_t*)src0->data;
    const float* src1_data = (const float*)src1->data;
    float* dst_data  = (float*)dst->data;

    // Dimensions and Strides
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne2  = dst->ne[2],  ne3  = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    // F16 specific block size (Usually QK_F16)
    const int block_size = QK_F16; 
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
            volatile float* c_element = (volatile float*)((char*)dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    return 0;
}

int mul_mat_f32(struct ggml_et_binary_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    // Thread coordination
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    if (thread_id < 0 || (thread_id & 1)) {
        return 0; // Skip odd threads to avoid resource contention
    }

    int effective_thread_id = thread_id / 2;
    int effective_num_threads = (num_threads + 1) / 2;

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B
    struct ggml_tensor* dst  = &params->dst;  // Output matrix C

    // Strictly validate for F32
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; 
    }

    const float* src0_data = (const float*)src0->data;
    const float* src1_data = (const float*)src1->data;
    float* dst_data       = (float*)dst->data;

    // Dimensions and Strides
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const int64_t ne2  = dst->ne[2],  ne3  = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

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
            volatile float* c_element = (volatile float*)((char*)dst_data + m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    return 0;
}

int mul_mat_Q8_0(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    const int64_t stride_m = 2048;

    // Matrix dimensions
    const int64_t K    = params->src0.ne[0];
    const int64_t M    = params->src0.ne[1];
    const int64_t N    = params->src1.ne[1];
    const int64_t ne02 = params->src0.ne[2];
    const int64_t ne03 = params->src0.ne[3];
    const int64_t ne12 = params->src1.ne[2];
    const int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    const size_t nb01 = params->src0.nb[1];
    const size_t nb02 = params->src0.nb[2];
    const size_t nb03 = params->src0.nb[3];

    const size_t nb11 = params->src1.nb[1];
    const size_t nb12 = params->src1.nb[2];
    const size_t nb13 = params->src1.nb[3];

    const size_t nbd1 = params->dst.nb[1];
    const size_t nbd2 = params->dst.nb[2];
    const size_t nbd3 = params->dst.nb[3];

    // Q8_0 block size is 32
    const int64_t K_blocks = K / 32;

    // Broadcasting ratios
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        const char* src0_ptr3 = (const char*)params->src0.data + i03 * nb03;
        const char* src1_ptr3 = (const char*)params->src1.data + i3 * nb13;
        char* dst_ptr3       = (char*)params->dst.data + i3 * nbd3;

        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char* src0_ptr2 = src0_ptr3 + i02 * nb02;
            const char* src1_ptr2 = src1_ptr3 + i2 * nb12;
            char* dst_ptr2       = dst_ptr3 + i2 * nbd2;

            for (int64_t n = 0; n < N; n++) {
                // src1 is F32, so column pointer moves by nb11
                const float* b_col_base = (const float*)(src1_ptr2 + n * nb11);

                for (int64_t m = hart_id; m < M; m += stride_m) {
                    // src0 is Q8_0 blocks, row pointer moves by nb01
                    const block_q8_0* q_row = (const block_q8_0*)(src0_ptr2 + m * nb01);
                    float sum = 0.0f;

                    for (int64_t kb = 0; kb < K_blocks; kb++) {
                        // q_row is a pointer to blocks, so + kb moves by sizeof(block_q8_0)
                        // b_col is float*, so we move 32 elements (kb << 5)
                        sum += compute_block_dot_product_q8_0(q_row + kb, b_col_base + (kb << 5));
                    }

                    // Store result in dst[m, n, i2, i3]
                    float* dst_entry = (float*)(dst_ptr2 + n * nbd1 + m * sizeof(float));
                    atomic_store_f32((volatile float*)dst_entry, sum);
                }
            }
        }
    }
    return 0;
}

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

// // static int once = 0;
#define NOP   __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n" ::: "memory");
#define WFI   __asm__ __volatile__ ("wfi\n");
#define THREAD_0 0
#define THREAD_1 1
#define FCC_0    0
#define FCC_1    1
#define MASTER_SHIRE 32

//******************************************************************************
// Atomic Operations
//******************************************************************************

// Global AMO primitives — ET custom 'g' suffix instructions that go through
// the NoC coherence fabric for chip-wide atomicity.

// Atomic swap (word), returns previous value.
static inline uint32_t __attribute__((always_inline))
et_global_swap_w(volatile void *addr, uint32_t val)
{
    uint32_t ret;
    __asm__ __volatile__(
        "amoswapg.w %0, %1, (%2)"
        : "=r"(ret) : "r"(val), "r"(addr) : "memory"
    );
    return ret;
}

// Atomic add (word), returns previous value.
static inline uint32_t __attribute__((always_inline))
et_global_add_w(volatile void *addr, uint32_t val)
{
    uint32_t ret;
    __asm__ __volatile__(
        "amoaddg.w %0, %1, (%2)"
        : "=r"(ret) : "r"(val), "r"(addr) : "memory"
    );
    return ret;
}

// Atomic store (halfword, global). Address must be 16-bit aligned.
static inline void __attribute__((always_inline))
et_global_store_hw(volatile void *addr, uint16_t val)
{
    __asm__ __volatile__(
        "shg %0, (%1)"
        : : "r"(val), "r"(addr) : "memory"
    );
}

// // Convenience wrappers — float types, fire-and-forget (old value discarded).
// ========================================================================
// Synchronization primitives for ET multi-shire graph execution
// Based on the proven gp-sdk sync.h barrier<Scope::device> pattern.
// ========================================================================

// FCC consume - blocks until a credit is available on the specified FCC register
inline __attribute__((always_inline)) void fcc_consume(uint64_t fcc_reg)
{
    __asm__ __volatile__("csrw fcc, %0\n" : : "r"(fcc_reg));
}

// ESR address construction for FCC credit increment registers
#define ESR_SHIRE_REGION       0x0100340000ULL
#define ESR_REGION_SHIRE_SHIFT 22
#define ESR_SHIRE(shire, name) \
    (ESR_SHIRE_REGION | ((uint64_t)(shire) << ESR_REGION_SHIRE_SHIFT) | (uint64_t)(ESR_##name))
#define ESR_FCC_CREDINC_0 0xC0

// Send FCC credit to specified shire/thread/register targeting specific minions
inline __attribute__((always_inline))
void fcc_send(uint32_t shire, uint32_t thread, uint32_t fcc_reg, uint64_t hart_mask)
{
    volatile uint64_t* fcc_credinc_addr =
        (uint64_t*)ESR_SHIRE(shire, FCC_CREDINC_0) +
        ((thread << 1) | fcc_reg);
    *fcc_credinc_addr = hart_mask;
}

// Fast Local Barrier - per-shire hardware barrier
// Returns 1 if this hart was the last to arrive, 0 otherwise
inline __attribute__((always_inline))
uint64_t flbarrier(uint64_t barrier_num, uint64_t match)
{
    uint64_t ret;
    uint64_t flb_arg = (match << 5) | (barrier_num & 0x1F);
    __asm__ __volatile__("csrrw %0, 0x820, %1" : "=r"(ret) : "r"(flb_arg));
    return ret;
}

// Evict whole shire L1+L2 via firmware syscall
static inline void __attribute__((always_inline)) flush_shire_l1_l2(void) {
    register uint64_t a0 __asm__("a0") = 11; // SYSCALL_CACHE_OPS_EVICT_WHOLE_L1_L2 11
    register uint64_t a1 __asm__("a1") = 0;
    register uint64_t a2 __asm__("a2") = 0;
    register uint64_t a3 __asm__("a3") = 0;
    __asm__ __volatile__("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3) : "memory");
}

// ========================================================================
// Device barrier: synchronizes ALL harts across ALL 32 compute shires.
// Ported from gp-sdk sync.h barrier<Scope::device>.
//
// Protocol (hierarchical, using master shire 0 as coordinator):
//   1. Intra-shire barrier (FLB 0 + FCC 0) — all 64 harts per shire sync
//   2. Shire leader flushes L1/L2 so writes are globally visible
//   3. Worker shires: minion 0 sends FCC 1 to master shire (to minion[shire_id])
//   4. Master shire: collector minions 1-31 each consume one FCC 1, then FLB
//   5. Last collector broadcasts FCC 1 to all master shire harts
//   6. All master shire harts consume FCC 1
//   7. Collector minions 1-31 each send FCC 1 to wake their assigned worker shire
//   8. Worker shire harts consume FCC 1 and proceed
// ========================================================================
#define SHIRE_OWN 0xFF
#define ALL_MINIONS_MASK 0xFFFFFFFFULL

static inline void __attribute__((always_inline))
device_barrier(uint32_t num_shires)
{
    const uint64_t hart_id   = get_hart_id();
    const uint32_t shire_id  = (uint32_t)(hart_id >> 6);
    const uint32_t local_id  = (uint32_t)((hart_id >> 1) & 0x1F); // minion id within shire (0-31)
    const uint32_t thread    = (uint32_t)(hart_id & 0x1);          // thread 0 or 1

    // --- Step 1: Intra-shire barrier (FLB 0, FCC 0) ---
    if (flbarrier(0, 63)) {
        // Last hart: flush cache, then wake all local harts
        flush_shire_l1_l2();
        fcc_send(SHIRE_OWN, 0, 0, ALL_MINIONS_MASK);
        fcc_send(SHIRE_OWN, 1, 0, ALL_MINIONS_MASK);
    }
    fcc_consume(0);

    // --- Step 2: Cross-shire sync (FCC 1) ---
    // Master shire = shire 0.  Uses minions 1..(num_shires-1) as collectors.
    if (num_shires <= 1) return; // single shire, nothing to do

    if (shire_id == 0) {
        // MASTER SHIRE
        if (local_id > 0 && local_id < num_shires) {
            // Collector minion: wait for credit from worker shire[local_id]
            fcc_consume(1);
            // FLB among all collectors: (num_shires-1) minions × 2 threads
            uint64_t flb_count = (uint64_t)((num_shires - 1) * 2 - 1);
            if (flbarrier(0, flb_count)) {
                // Last collector: broadcast release to entire master shire
                fcc_send(0, 0, 1, ALL_MINIONS_MASK);
                fcc_send(0, 1, 1, ALL_MINIONS_MASK);
            }
        }
        // ALL master shire harts wait for release
        fcc_consume(1);

        // Master shire wakes worker shires: minion[i] wakes shire[i]
        if (local_id > 0 && local_id < num_shires) {
            fcc_send(local_id, thread, 1, ALL_MINIONS_MASK);
        }
    } else if (shire_id < num_shires) {
        // WORKER SHIRE
        // Minion 0 sends arrival credit to master shire, targeting minion[shire_id]
        if (local_id == 0) {
            uint64_t target_minion = 1ULL << shire_id;
            fcc_send(0, thread, 1, target_minion);
        }
        // ALL worker shire harts wait for wake-up from master
        fcc_consume(1);
    }


}



// ========================================================================
// Entry point — graph execution loop
// ========================================================================
int entry_point(struct ggml_cgraph_et* cg, void* env) {
    int64_t hart_id = get_hart_id();

    // Reconstruct pointers on device side
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *) cg->data;
    uint8_t * node_op = (uint8_t *) (node_meta + cg->n_nodes);
    const int n_nodes = cg->n_nodes;

    for (int i = 0; i < n_nodes; i++)
    {
        const int node_op_val = node_op[i];
        if (node_op_val == HOST_GGML_OP_NONE) continue;

        switch (node_op_val) {
            case HOST_GGML_OP_MUL:
            case HOST_GGML_OP_ADD:
                {
                    struct ggml_et_elmap_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    const enum ggml_op el_op = (node_op_val == HOST_GGML_OP_MUL) ? GGML_OP_MUL : GGML_OP_ADD;
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, el_op);

                    if (params.dst.type != GGML_TYPE_F32 ||
                        params.src0.type != GGML_TYPE_F32 ||
                        params.src1.type != GGML_TYPE_F32) {
                        break;
                    }
                    el_map_f32(&params, env);
                }
                break;

            case HOST_GGML_OP_MUL_MAT:
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

            case HOST_GGML_OP_MUL_MAT_ID:
                break;

            case HOST_GGML_OP_RMS_NORM:
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

            case HOST_GGML_OP_GLU:
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

            case HOST_GGML_OP_SOFT_MAX:
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

            case HOST_GGML_OP_GET_ROWS:
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

            case HOST_GGML_OP_SET_ROWS:
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

            case HOST_GGML_OP_CONT:
                {
                    struct ggml_et_cont_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, GGML_OP_CONT);
                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
                        cont_f32_impl(&params, env);
                    }
                }
                break;

            case HOST_GGML_OP_ROPE:
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

            case HOST_GGML_OP_RESHAPE:
            case HOST_GGML_OP_VIEW:
            case HOST_GGML_OP_PERMUTE:
            case HOST_GGML_OP_TRANSPOSE:
                break;

            default:
                break;
        }

        // Synchronize all harts across all shires after each compute node.
        // Skip barrier for metadata-only ops (RESHAPE/VIEW/PERMUTE/TRANSPOSE)
        // since they don't touch data.
        if (node_op_val != HOST_GGML_OP_RESHAPE &&
            node_op_val != HOST_GGML_OP_VIEW &&
            node_op_val != HOST_GGML_OP_PERMUTE &&
            node_op_val != HOST_GGML_OP_TRANSPOSE &&
            node_op_val != HOST_GGML_OP_NONE) {
            // device_barrier_refined(32);
            device_barrier(32);
            
        }
    }

    return 0;
}
