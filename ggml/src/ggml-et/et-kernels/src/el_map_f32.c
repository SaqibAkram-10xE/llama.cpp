//******************************************************************************
// Element-wise Map F32 Kernel
// Element-wise operations: dst[i] = src0[i] op src1[i]
// Supports: MUL, ADD, SUB
//
// When ENABLE_MONOLITHIC_COMPUTE is defined (included by op_cgraph_2.c),
// the function is compiled as el_map_f32. Otherwise it compiles as entry_point.
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"

// ============================================================
// Block helpers: vectorised ET SIMD operations
// ============================================================

static inline void block_mul_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t vec_end = (elements / 8) * 8;
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    for (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "flw.ps f11, %[src1_vec]\n"
            "fmul.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_add_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t vec_end = (elements / 8) * 8;
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    for (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "flw.ps f11, %[src1_vec]\n"
            "fadd.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

static inline void block_sub_cache_aligned(float* dst_block, const float* src0_block, const float* src1_block, int elements) {
    int32_t vec_end = (elements / 8) * 8;
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    for (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ volatile(
            "flw.ps f10, %[src0_vec]\n"
            "flw.ps f11, %[src1_vec]\n"
            "fsub.ps f12, f10, f11\n"
            "fsw.ps f12, %[dst_vec]\n"
            : [dst_vec] "=m"(*(float(*)[8])&dst_block[i])
            : [src0_vec] "m"(*(const float(*)[8])&src0_block[i]),
              [src1_vec] "m"(*(const float(*)[8])&src1_block[i])
            : "f10", "f11", "f12"
        );
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
}

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

// ============================================================
// Main implementation — function name controlled by macro
// ============================================================

#ifdef ENABLE_MONOLITHIC_COMPUTE
#define EL_MAP_F32_FUNC el_map_f32
#else
#define EL_MAP_F32_FUNC entry_point
#endif

int EL_MAP_F32_FUNC(struct ggml_et_binary_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;
    if (!kernel_env) return -1;

    int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    if (params == 0 || ((uint64_t)params & 0x7) != 0) return -1;

    struct ggml_tensor* src0 = &params->src0;
    struct ggml_tensor* src1 = &params->src1;
    struct ggml_tensor* dst  = &params->dst;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    float* src0_data = (float*)src0->data;
    float* src1_data = (float*)src1->data;
    float* dst_data  = (float*)dst->data;
    if (!src0_data || !src1_data || !dst_data) return -1;

    enum ggml_op operation = dst->op;
    if (operation != GGML_OP_MUL && operation != GGML_OP_ADD && operation != GGML_OP_SUB) return -1;

    const int64_t ne0  = dst->ne[0],  ne1  = dst->ne[1],  ne2  = dst->ne[2],  ne3  = dst->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    if (dst->ne[0] % 16 != 0) return 1;

    const bool no_broadcast   = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    const bool all_contiguous = (nb0 == 4 && nb00 == 4 && nb10 == 4 &&
                                 nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    if (no_broadcast && all_contiguous) {
        const int64_t total_elements       = ne0 * ne1 * ne2 * ne3;
        const int64_t elements_per_cacheline = 16;
        const int64_t total_cachelines     = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;
        const int64_t cl_per_thread        = (total_cachelines + num_threads - 1) / num_threads;
        const int64_t cl_start             = thread_id * cl_per_thread;
        int64_t       cl_end               = cl_start + cl_per_thread;
        if (cl_end > total_cachelines) cl_end = total_cachelines;
        if (cl_start >= total_cachelines) return 0;

        const int64_t elem_start = cl_start * elements_per_cacheline;
        int64_t       elem_end   = cl_end   * elements_per_cacheline;
        if (elem_end > total_elements) elem_end = total_elements;
        const int32_t count = (int32_t)(elem_end - elem_start);

        switch (operation) {
            case GGML_OP_MUL: block_mul_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, count); break;
            case GGML_OP_ADD: block_add_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, count); break;
            case GGML_OP_SUB: block_sub_cache_aligned(dst_data + elem_start, src0_data + elem_start, src1_data + elem_start, count); break;
            default: return 1;
        }
        return 0;
    }

    const int64_t total_rows      = ne1 * ne2 * ne3;
    const int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
    const int64_t start_row       = thread_id * rows_per_thread;
    const int64_t end_row         = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;
    if (start_row >= total_rows) return 0;

    for (int64_t ir = start_row; ir < end_row; ir++) {
        const int64_t i03 = ir / (ne2 * ne1);
        const int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        const int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);
        const int64_t i13 = i03 % ne13, i12 = i02 % ne12, i11 = i01 % ne11;

        float*       dst_ptr  = (float*)((char*)dst_data       + i03*nb3  + i02*nb2  + i01*nb1);
        const float* src0_ptr = (const float*)((const char*)src0_data + i03*nb03 + i02*nb02 + i01*nb01);
        const float* src1_ptr = (const float*)((const char*)src1_data + i13*nb13 + i12*nb12 + i11*nb11);

        if (ne10 == 1) {
            float scalar = src1_ptr[0];
            switch (operation) {
                case GGML_OP_MUL: block_mul_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0); break;
                case GGML_OP_ADD: block_add_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0); break;
                case GGML_OP_SUB: block_sub_broadcast(dst_ptr, src0_ptr, scalar, (int)ne0); break;
                default: return 1;
            }
        } else {
            const int64_t nr0 = ne0 / ne10;
            for (int64_t r = 0; r < nr0; r++) {
                const float* src0_block = src0_ptr + r * ne10;
                float*       dst_block  = dst_ptr  + r * ne10;
                switch (operation) {
                    case GGML_OP_MUL: block_mul_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10); break;
                    case GGML_OP_ADD: block_add_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10); break;
                    case GGML_OP_SUB: block_sub_cache_aligned(dst_block, src0_block, src1_ptr, (int)ne10); break;
                    default: return 1;
                }
            }
        }
    }
    return 0;
}
