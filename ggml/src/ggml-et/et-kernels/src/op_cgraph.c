//******************************************************************************
// Element-wise Map F32 Kernel
// Element-wise operations: dst[i] = src0[i] op src1[i]
// Supports: MUL, ADD (more operations to be added later)
//******************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "ggml_tensor.h"
#include "platform.h"

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

// struct ggml_et_binary_params {
//     struct ggml_tensor src0;
//     struct ggml_tensor src1;
//     struct ggml_tensor dst;
// };

struct ggml_tensor_et {
    int64_t ne[4];      // dimensions
    size_t nb[4];       // strides
    enum ggml_type type;
    void* data;         // Device pointer if needed
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

    uint8_t data[];   // flexible array at end
};

void delay(unsigned long count) {
    volatile unsigned long i;
    for (i = 0; i < count; i++) {
        // empty
    }
}
#define FENCE __asm__ __volatile__ ("fence\n");

static float find_max_f32(const float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    return max_val;
}

static void compute_softmax_row(
    float* dst,           // Output row
    const float* src,     // Input row
    const float* mask,    // Mask row (can be NULL)
    int ne00,             // Input row length
    int ne10,             // Mask row length (guaranteed equal to ne00 in ggml)
    float scale,          // Scale factor
    float slope,          // ALiBi slope factor
    float sink_value,     // Sink value for this head (or -INFINITY if no sinks)
    bool use_sinks)       // Whether sinks are enabled
{
    // Step 1: Apply scaling and masking/bias to input
    for (int i = 0; i < ne00; i++) {
        dst[i] = src[i] * scale;
    }

    // Add mask/bias if present
    if (mask != NULL) {
        for (int i = 0; i < ne00; i++) {
            dst[i] += slope * mask[i];
        }
    }

    // Step 2: Find maximum for numerical stability
    float max_val = find_max_f32(dst, ne00);

    if (use_sinks) {
        if (sink_value > max_val) {
            max_val = sink_value;
        }
    }

    // Step 3: Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < ne00; i++) {
        float exp_val = et_expf(dst[i] - max_val);
        dst[i] = exp_val;
        sum += exp_val;
    }

    if (use_sinks) {
        sum += et_expf(sink_value - max_val);
    }

    // Step 4: Normalize by sum to get probabilities
    if (sum > 0.0f) {
        float inv_sum = et_fdiv(1.0f, sum);
        for (int i = 0; i < ne00; i++) {
            dst[i] *= inv_sum;
        }
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

static void copy_f32_to_f16_row(uint16_t* dst, const float* src, int64_t num_elements) {
    for (int64_t i = 0; i < num_elements; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
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

static inline void rope_yarn(float theta_extrap, float freq_scale, const float corr_dims[2],
                             int64_t i0, float ext_factor, float mscale,
                             float* cos_theta, float* sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
    }

    *cos_theta = et_cosf(theta) * mscale;
    *sin_theta = et_sinf(theta) * mscale;
}
/*
// Block operation implementations using ET vector instructions
static inline void block_mul(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    // Process 8 elements at a time using vector multiplication
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Compute results into temporary buffer
        float temp_result[8];
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
            "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
            "fmul.ps f12, f10, f11\n"          // dst = src0 * src1 (8-wide)
            "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

            : [dst_vec] "=m"(*(float(*)[8])temp_result)
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );

        // Use atomic stores to write results to global memory
        for (int32_t j = 0; j < 8; j++) {
            atomic_store_f32((volatile float*)&dst_block[i + j], temp_result[j]);
        }
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Handle remaining elements (< 8) with scalar operations and atomic stores
    for (int32_t i = vec_end; i < elements; i++) {
        float result = src0_block[i] * src1_block[i];
        atomic_store_f32((volatile float*)&dst_block[i], result);
    }
}
*/

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

// static inline void block_add(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
//     // Process 8 elements at a time using vector addition
//     int32_t vec_end = (elements / 8) * 8;

//     // Set mask register to enable all 8 vector elements
//     unsigned long temp_mask;
//     __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
//     __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

//     for (int32_t i = 0; i < vec_end; i += 8) {
//         // Compute results into temporary buffer
//         float temp_result[8];
//         __asm__ volatile(
//             "flw.ps f10, %[src0_vec]\n"        // Load 8 src0 values
//             "flw.ps f11, %[src1_vec]\n"        // Load 8 src1 values
//             "fadd.ps f12, f10, f11\n"          // dst = src0 + src1 (8-wide)
//             "fsw.ps f12, %[dst_vec]\n"         // Store 8 results to temp buffer

//             : [dst_vec] "=m"(*(float(*)[8])temp_result)
//             : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
//               [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
//             : "f10", "f11", "f12"
//         );

//         // Use atomic stores to write results to global memory
//         for (int32_t j = 0; j < 8; j++) {
//             atomic_store_f32((volatile float*)&dst_block[i + j], temp_result[j]);
//         }
//     }

//     // Restore original mask
//     __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

//     // Handle remaining elements (< 8) with scalar operations and atomic stores
//     for (int32_t i = vec_end; i < elements; i++) {
//         float result = src0_block[i] + src1_block[i];
//         atomic_store_f32((volatile float*)&dst_block[i], result);
//     }
// }


/*
static inline void block_add(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements (m0 is the mask register)
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           

    for (int32_t i = 0; i < vec_end; i += 8) {
        // Direct vector addition and ordinary store to global memory
        __asm__ volatile(
            "flw.ps f10, %1\n"        // Load 8 src0 values
            "flw.ps f11, %2\n"        // Load 8 src1 values
            "fadd.ps f12, f10, f11\n" // dst = src0 + src1 (8-wide)
            "fsw.ps f12, %0\n"        // Ordinary store of 8 results to dst_block
            : "=m"(*(float(*)[8])&dst_block[i])
            : "m"(*(const float(*)[8])&src0_block[i]),
              "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12", "memory"
        );
    }

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Handle remaining elements (< 8) with standard scalar stores
    for (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = src0_block[i] + src1_block[i];
    }
}
*/

static inline void block_add(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t i = 0;
    unsigned long original_mask;
    
    // Save current mask
    __asm__ volatile("mova.x.m %0" : "=r"(original_mask));

    // 1. Process main blocks of 8 elements
    if (elements >= 8) {
        int32_t vec_end = (elements / 8) * 8;
        // Use an immediate for 0xFF since it's a constant
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
        // FIX: Move the variable tail_mask into a register (%0) 
        // then move that register into m0.
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
                    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

                    for (int32_t i0 = 0; i0 < vec_end; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            "fsw.ps f10, %[result]\n"
                            : [result] "=m"(*(float(*)[8])acc_vec)
                            : [acc] "m"(*(const float(*)[8])acc_vec),
                              [x_vec] "m"(*(const float(*)[8])&src_ptr[i0])
                            : "f10", "f11"
                        );
                    }

                    for (int i = 0; i < 8; i++) {
                        sum += acc_vec[i];
                    }
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
    if (params->glu_op_type != GGML_GLU_OP_SWIGLU) return -1;

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

            block_swiglu(dst_ptr, x_ptr, g_ptr, (int)elements_to_process);

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
    if (thread_id < 0) return 0;
    if (thread_id != 0) return 0; // Single-threaded for now

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

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
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

            for (int64_t i01 = 0; i01 < ne01; i01++) {
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

                compute_softmax_row(dst_row, src_row, mask_row, (int)ne00, (int)ne10, scale, slope, sink_value, use_sinks);
            }
        }
    }

    return 0;
}

int get_rows_f32_impl(struct ggml_et_get_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) return 0;
    if (thread_id != 0) return 0; // Single-threaded for now

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst = &params->dst;

    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) return -1;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_Q8_0) return -1;

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
        }
    }

    return 0;
}

int set_rows_f32_impl(struct ggml_et_set_rows_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    if (thread_id < 0) return 0;
    if (thread_id != 0) return 0; // Single-threaded for now

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

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                const int64_t i10 = i01;

                const int64_t index_byte_offset = i10*nb10 + i11*nb11 + i12*nb12;
                const int64_t dst_row_index = *(int64_t*)((char*)src1_data + index_byte_offset);

                if (dst_row_index < 0 || dst_row_index >= ne_dst1) return -1;

                const char* src_row_ptr = (char*)src0_data + i01*nb01 + i02*nb02 + i03*nb03;
                const float* src_row = (const float*)src_row_ptr;

                char* dst_row_ptr = (char*)dst_data + dst_row_index*nb1 + i02*nb2 + i03*nb3;

                if (dst->type == GGML_TYPE_F32) {
                    float* dst_row = (float*)dst_row_ptr;
                    copy_f32_row(dst_row, src_row, ne00);
                } else if (dst->type == GGML_TYPE_F16) {
                    uint16_t* dst_row = (uint16_t*)dst_row_ptr;
                    copy_f32_to_f16_row(dst_row, src_row, ne00);
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

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base,
                       rope_params->beta_fast, rope_params->beta_slow, corr_dims);

    const int64_t total_work_units = batch * seq_len * heads;
    int64_t units_per_thread = total_work_units / num_threads;
    int64_t start_unit = thread_id * units_per_thread;
    int64_t end_unit = (thread_id == num_threads - 1) ? total_work_units : start_unit + units_per_thread;

    const float theta_scale = et_powf(freq_base, et_fdiv(-2.0f, (float)n_dims));

    if (mode & GGML_ROPE_TYPE_NEOX) {
        // NeoX mode: split pairs (k, k + n_dims/2)
        for (int64_t unit = start_unit; unit < end_unit; unit++) {
            int64_t h = unit % heads;
            int64_t s = (unit / heads) % seq_len;
            int64_t b = unit / (heads * seq_len);

            const float* head_src = (const float*)((char*)src0_data +
                b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);
            float* head_dst = (float*)((char*)dst_data +
                b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

            const int32_t pos = src1_data[s];

            // Copy entire head first (skip for inplace, avoids overwriting rotation results)
            if (head_src != head_dst) {
                for (int64_t i0 = 0; i0 < head_dim; i0++) {
                    head_dst[i0] = head_src[i0];
                }
            }

            // Apply NeoX rotations using accumulated theta
            float theta = (float)pos;
            for (int64_t dim_idx = 0; dim_idx < n_dims / 2; dim_idx++) {
                const float ff = freq_factors ? freq_factors[dim_idx] : 1.0f;

                float cos_theta, sin_theta;
                rope_yarn(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2,
                         rope_params->ext_factor, rope_params->attn_factor,
                         &cos_theta, &sin_theta);

                const float x0 = head_src[dim_idx];
                const float x1 = head_src[dim_idx + n_dims/2];

                head_dst[dim_idx]            = x0 * cos_theta - x1 * sin_theta;
                head_dst[dim_idx + n_dims/2] = x0 * sin_theta + x1 * cos_theta;

                theta *= theta_scale;
            }
        }
    } else {
        // Standard mode (mode=0): consecutive pairs (2k, 2k+1)
        for (int64_t unit = start_unit; unit < end_unit; unit++) {
            int64_t h = unit % heads;
            int64_t s = (unit / heads) % seq_len;
            int64_t b = unit / (heads * seq_len);

            const float* head_src = (const float*)((char*)src0_data +
                b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);
            float* head_dst = (float*)((char*)dst_data +
                b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

            const int32_t pos = src1_data[s];

            // Copy entire head first (handles non-rotated elements beyond n_dims)
            if (head_src != head_dst) {
                for (int64_t i0 = 0; i0 < head_dim; i0++) {
                    head_dst[i0] = head_src[i0];
                }
            }

            // Apply standard rotations: consecutive pairs (2k, 2k+1)
            float theta = (float)pos;
            for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;

                float cos_theta, sin_theta;
                rope_yarn(et_fdiv(theta, ff), freq_scale, corr_dims, i0,
                         rope_params->ext_factor, rope_params->attn_factor,
                         &cos_theta, &sin_theta);

                const float x0 = head_src[i0];
                const float x1 = head_src[i0 + 1];

                head_dst[i0]     = x0 * cos_theta - x1 * sin_theta;
                head_dst[i0 + 1] = x0 * sin_theta + x1 * cos_theta;

                theta *= theta_scale;
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

int el_map_f32(struct ggml_et_elmap_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

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
    struct ggml_tensor* dst  = &params->dst;

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

    if (operation != GGML_OP_MUL && operation != GGML_OP_ADD) {
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

    // Cache line alignment: prevent false sharing between threads
    // Each cache line is 64 bytes = 16 floats
    const int64_t CACHE_LINE_BYTES = 64;
    const int64_t row_bytes = ne0 * sizeof(float);
    const int64_t row_gcd = gcd_i64(row_bytes, CACHE_LINE_BYTES);
    const int64_t rows_per_cache_group = CACHE_LINE_BYTES / row_gcd;

    // Distribute rows across threads, rounding up to cache line boundaries
    int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    
    // Round rows_per_thread up to nearest multiple of rows_per_cache_group
    // This ensures each thread's memory region starts at a cache line boundary
    if (rows_per_cache_group > 1) {
        rows_per_thread = ((rows_per_thread + rows_per_cache_group - 1) / rows_per_cache_group) * rows_per_cache_group;
    }

    const int64_t start_row = thread_id * rows_per_thread;
    const int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

    if (start_row >= total_rows) {
        return 0;
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
                    block_mul(dst_block, src0_block, src1_ptr, (int)ne10);
                    break;
                case GGML_OP_ADD:
                    block_add(dst_block, src0_block, src1_ptr, (int)ne10);
                    break;
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
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;
       
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    
    if (thread_id < 0) return 0;

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

                for (int64_t m = thread_id; m < M; m += num_threads) {
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
    dst->data = src->data;
    dst->op   = op;
    for(int j = 0; j < 4; j++){
        dst->ne[j] = src->ne[j];
        dst->nb[j] = src->nb[j];
    }
}

// // static int once = 0;

// /*! \fn inline uint64_t shire_barrier(uint64_t flb, uint64_t fcc, uint64_t thread_count, uint64_t minion_mask_t0, uint64_t minion_mask_t1)
//     \brief Shire-only barrier using FLBs and FCCs
//     \param flb FLbarrier value
//     \param fcc  FCC value
//     \param thread_count active thread count
//     \param minion_mask_t0 Mask of active thread0 minions
//     \param minion_mask_t1 Mask of active thread1 minions
//     \return last thread to reach barrier
//     \syncops Implementation of shire_barrier api
// */

// // Please read te files:
// //home/saqib/Documents/Prj/P1/ET_platform/et-platform/et-common-libs/include/etsoc/isa

// inline uint64_t __attribute__((always_inline)) shire_barrier(uint64_t flb, uint64_t fcc,
//     uint64_t thread_count, uint64_t minion_mask_t0, uint64_t minion_mask_t1)
// {
//     uint64_t last = flbarrier(flb, thread_count - 1);

//     if (last)
//     {
//         fcc_send(SHIRE_OWN, THREAD_0, fcc, minion_mask_t0);
//         fcc_send(SHIRE_OWN, THREAD_1, fcc, minion_mask_t1);
//     }
//     fcc_consume(fcc);

//     return last;
// }
#define MCACHE_CONTROL 0x7CA // Machine-mode CSR
#define UCACHE_CONTROL 0x801 // User-mode shadow CSR

void bulk_invalidate_l1() {
    // 1. Memory Fence: Ensure all previous memory ops are complete
    __asm__ volatile ("fence" ::: "memory");

    // 2. Toggle the D1Split bit (Bit 0)
    // Switching from Shared to Split (or vice-versa) invalidates the entire L1.
    // We read the current state, flip bit 0, and write it back.
    unsigned long current_ctrl;
    __asm__ volatile ("csrr %0, %1" : "=r"(current_ctrl) : "i"(UCACHE_CONTROL));
    
    unsigned long toggled_ctrl = current_ctrl ^ 0x1; 
    __asm__ volatile ("csrw %0, %1" :: "i"(UCACHE_CONTROL), "r"(toggled_ctrl));

    // 3. Return to original state (Optional, but usually desired)
    __asm__ volatile ("csrw %0, %1" :: "i"(UCACHE_CONTROL), "r"(current_ctrl));

    // 4. Sync: Wait for the hardware FSM to finish the invalidation/zeroing
    // Documentation suggests a TensorWait or similar sync after cacheops
    __asm__ volatile ("fence" ::: "memory");
}

inline void __attribute__((always_inline))
cache_invalidate(uint64_t inval_instr_cache, uint64_t inval_TLBs_and_PTW)
{
    uint64_t csr_enc = (inval_TLBs_and_PTW & 1) | ((inval_instr_cache & 1) << 1);

    __asm__ __volatile__("csrw 0x7d0, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

int entry_point(struct ggml_cgraph_et* cg, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;
    
    int64_t hart_id = get_hart_id();
    // static int once = 0;
    
    // Reconstruct pointers on device side
    struct ggml_node_meta_et * node_meta = (struct ggml_node_meta_et *) cg->data;
    uint8_t * node_op = (uint8_t *) (node_meta + cg->n_nodes);
    
    // if(once == 0){
    //     hart_id == 0 ? et_printf("***DEV***: Hart %d starting execution\n", hart_id) : et_printf("");
    //     hart_id == 0 ? et_printf("***DEV***: Computing graph with %d nodes\n", cg->n_nodes) : et_printf("");
    // }
    
    const uint8_t n_nodes = cg->n_nodes;
    // hart_id == 0 ? et_printf("***DEV***: cgraph->nodes[0]->src[0]->type: %d\n", cgraph->nodes[0]->src[0]->type) : et_printf("");
    // We can access cgraph->nodes[i] pointer (host memory mapped)

    // if(once == 0)
    // {
    //     hart_id == 0 ? et_printf("***DEV***: cgraph->size: %d\n", cg->size) : et_printf("");
    //     hart_id == 0 ? et_printf("***DEV***: cgraph->nleafs: %d\n", cg->n_leafs) : et_printf("");
    //     hart_id == 0 ? et_printf("***DEV***: node_op[0] %d\n", node_op[0]) : et_printf("");
    //     hart_id == 0 ? et_printf("***DEV***: n_nodes %d\n", n_nodes) : et_printf("");
    //     once = 1;
    // }

    int num_threads = get_num_threads(kernel_env->shire_mask);
    
    for (int i = 0; i < n_nodes; i++)
    {

        // cache_invalidate(1,1);

        // bulk_invalidate_l1();
        // shire_barrier(0, 0, num_threads, 0xFFFFFFFF, 0xFFFFFFFF);
        
        const int node_op_val = node_op[i];
        if (node_op_val == GGML_OP_NONE) continue;
        
        switch (node_op_val) {
            case GGML_OP_MUL:
            case GGML_OP_ADD:
                {
                    struct ggml_et_elmap_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);

                    // Type validation
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
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);

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
                // hart_id == 0 ? et_printf("***DEV***: Executed GGML_OP_MUL_MAT_ID \n") : et_printf("");

                // ggml_et_op_mul(dev_ctx, node);
                // ggml_et_op_mul_mat_id(dev_ctx, node);
                break;

            case GGML_OP_RMS_NORM:
                {
                    struct ggml_et_rms_norm_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);
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
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);
                    memcpy(&params.glu_op_type, &node_meta[i].op_params[0], sizeof(int32_t));
                    memcpy(&params.swapped, &node_meta[i].op_params[1], sizeof(int32_t));

                    if (params.dst.type == GGML_TYPE_F32 && params.src0.type == GGML_TYPE_F32) {
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
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);
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
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);

                    if (params.dst.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_I32) {
                        get_rows_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_SET_ROWS:
                {
                    struct ggml_et_set_rows_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.src1, &node_meta[i].src1, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);

                    if (params.src0.type == GGML_TYPE_F32 && params.src1.type == GGML_TYPE_I64) {
                        set_rows_f32_impl(&params, env);
                    }
                }
                break;

            case GGML_OP_CONT:
                {
                    struct ggml_et_cont_params params;
                    convert_to_ggml_tensor(&params.src0, &node_meta[i].src0, GGML_OP_NONE);
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);

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
                    convert_to_ggml_tensor(&params.dst, &node_meta[i].dst, (enum ggml_op)node_op_val);
                    
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

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                {
                    // hart_id == 0 ? et_printf("***DEV***: Executed GGML_OP_RESHAPE/VIEW/PERMUTE/TRANSPOSE \n") : et_printf("");
                    // These are metadata-only operations that require no computation
                    // GGML_LOG_DEBUG("ET: No-op metadata operation: %s\n", ggml_op_name(node->op));
                }
                break;

            default:
                {
                    // hart_id == 0 ? et_printf("***DEV***: Executed DEFAULT/UNSUPPORTED OP %d \n", node_op_val) : et_printf("");
                    // GGML_LOG_ERROR("ET: Unsupported operation in graph: %s\n", ggml_op_name(node->op));
                }
                break; //GGML_STATUS_FAILED;
        }
    
    
    }
    

   
        
    return 0;
}
