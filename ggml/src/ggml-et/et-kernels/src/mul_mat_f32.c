//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

KERNEL_TRAMPOLINE();




/*! \def QUANT_LAST_TRANS
    \brief Tensor Quant instruction: Do not perform any more transformations.
*/
#define QUANT_LAST_TRANS 0
/*! \def QUANT_INT32_TO_FP32
    \brief Tensor Quant instruction: Convert all elements of A from 32-bit signed integer values to single-precision
    floating-point values.*/
#define QUANT_INT32_TO_FP32 1
/*! \def QUANT_FP32_TO_INT32
    \brief Tensor Quant instruction: Convert all elements of A from single-precision floating-point values to 32-
    bit signed integer values.*/
#define QUANT_FP32_TO_INT32 2
/*! \def QUANT_RELU
    \brief Tensor Quant instruction: Convert all negative INT32 values in A to 0*/
#define QUANT_RELU 3
/*! \def QUANT_INT32_ADD_ROW
    \brief Tensor Quant instruction: Read the low-order COLS+1 32-bit signed integer values from an L1
    scratchpad line, and add this vector to every row of the 32-bit signed integer
    matrix A.*/
#define QUANT_INT32_ADD_ROW 4
/*! \def QUANT_INT32_ADD_COL
    \brief Tensor Quant instruction: Read the low-order ROWS+1 32-bit signed integer values from an L1
    scratchpad line, and add this vector to every column of the 32-bit signed
    integer matrix A.*/
#define QUANT_INT32_ADD_COL 5
/*! \def QUANT_FP32_MUL_ROW
    \brief Tensor Quant instruction: Read the low-order COLS+1 single-precision floating-point values from an
    L1 scratchpad line, and multiply the single-precision elements of each row
    of matrix A element-wise by this vector.*/
#define QUANT_FP32_MUL_ROW 6
/*! \def QUANT_FP32_MUL_COL
    \brief Tensor Quant instruction: Read the low-order ROWS+1 single-precision floating-point values from an
    L1 scratchpad line, and multiply the single-precision elements of each col-
    umn of matrix A element-wise by this vector.*/
#define QUANT_FP32_MUL_COL 7
/*! \def QUANT_SATINT8
    \brief Tensor Quant instruction: Clamp all 32-bit signed integer values in A to the range [-128, 127]. 
    The values are written in bits 7:0 of each element, with bits 31:8 set to zero.*/
#define QUANT_SATINT8 8
/*! \def QUANT_SATUINT8
    \brief Tensor Quant instruction: Clamp all 32-bit signed integer values in A to the range [0, 255]. The values
    are written in bits 7:0 of each element, with bits 31:8 set to zero.*/
#define QUANT_SATUINT8 9
/*! \def QUANT_PACK_128B
    \brief Tensor Quant instruction: Copy the low-order byte of the n-th 32-bit value in each row of A to the n-th
    byte of the row.*/
#define QUANT_PACK_128B 10
/*! \def TENSOR_REDUCE_OP_FADD
    \brief Tensor Reduce instruction: The result is the addition of the incoming single-precision floating-point data
    and the single-precision floating-point values in the vector register file.*/
#define TENSOR_REDUCE_OP_FADD 0
// #define TENSOR_REDUCE_OP_FSUB 1 -- Not supported
/*! \def TENSOR_REDUCE_OP_FMAX
    \brief Tensor Reduce instruction: The result is the maximum of the incoming single-precision floating-point data
and the single-precision floating-point values in the vector register file.*/
#define TENSOR_REDUCE_OP_FMAX 2
/*! \def TENSOR_REDUCE_OP_FMIN
    \brief Tensor Reduce instruction: The result is the minimum of the incoming single-precision floating-point data
and the single-precision floating-point values in the vector register file..*/
#define TENSOR_REDUCE_OP_FMIN 3
/*! \def TENSOR_REDUCE_OP_IADD
    \brief Tensor Reduce instruction: The result is the addition of the incoming 32-bit integer data and the 32-bit inte-
ger values in the vector register file.*/
#define TENSOR_REDUCE_OP_IADD 4
// #define TENSOR_REDUCE_OP_ISUB 5 -- Not supported
/*! \def TENSOR_REDUCE_OP_IMAX
    \brief Tensor Reduce instruction: The result is the maximum of the incoming 32-bit signed integer data and the
32-bit signed integer values in the vector register file.*/
#define TENSOR_REDUCE_OP_IMAX 6
/*! \def TENSOR_REDUCE_OP_IMIN
    \brief Tensor Reduce instruction: The result is the minimum of the incoming 32-bit signed integer data and the
32-bit signed integer values in the vector register file.*/
#define TENSOR_REDUCE_OP_IMIN 7
/*! \def TENSOR_REDUCE_OP_FGET
    \brief Tensor Reduce instruction get function to be performed*/
#define TENSOR_REDUCE_OP_FGET 8
/*! \def TENSOR_LOAD_WAIT_0
    \brief Tensor load to L1 Scratchpad with ID = 0 is complete.*/
#define TENSOR_LOAD_WAIT_0 0
/*! \def TENSOR_LOAD_WAIT_1
    \brief Tensor load to L1 Scratchpad with ID = 1 is complete.*/
#define TENSOR_LOAD_WAIT_1 1
/*! \def TENSOR_FMA_WAIT
    \brief All previous tensor matrix multiplication instructions are complete.*/
#define TENSOR_FMA_WAIT 7
/*! \def TENSOR_STORE_WAIT
    \brief All previous tensor store instructions are complete.*/
#define TENSOR_STORE_WAIT 8
/*! \def TENSOR_REDUCE_WAIT
    \brief All previous tensor reduction instructions are complete*/
#define TENSOR_REDUCE_WAIT 9
/*! \def TENSOR_QUANT_WAIT
    \brief TensorQuant is complete*/
#define TENSOR_QUANT_WAIT 10
/*! \def TENSOR_ERROR_LOAD_TRANSFORM
    \brief Define for tensor load transform error.*/
#define TENSOR_ERROR_LOAD_TRANSFORM 1
/*! \def TENSOR_ERROR_FCC_OVERFLOW
    \brief Define for tensor fcc overflow error.*/
#define TENSOR_ERROR_FCC_OVERFLOW 3
/*! \def TENSOR_ERROR_SCP_DISABLED
    \brief Define for tensor scp disabled error.*/
#define TENSOR_ERROR_SCP_DISABLED 4
/*! \def TENSOR_ERROR_LOCKSW
    \brief Define for tensor locksw error.*/
#define TENSOR_ERROR_LOCKSW 5
/*! \def TENSOR_ERROR_TL1_FMA
    \brief Define for L1 FMA error.*/
#define TENSOR_ERROR_TL1_FMA 6
/*! \def TENSOR_ERROR_MEM_FAULT
    \brief Define for Memory fault error.*/
#define TENSOR_ERROR_MEM_FAULT 7
/*! \def TENSOR_ERROR_STORE_COOP
    \brief Define for store coop error.*/
#define TENSOR_ERROR_STORE_COOP 8
/*! \def TENSOR_ERROR_REDUCE
    \brief Define for tensor reduce error.*/
#define TENSOR_ERROR_REDUCE 9
/*! \fn inline void tensor_wait(long id)
    \brief Tensor wait instruction, Tensor Wait can be used to stall execution until
    a previously issued tensor instruction completes.
    \param id tensor ID
    \return none
    \tensorops Implementation of tensor_wait api
*/
inline __attribute__((always_inline)) void tensor_wait(long id)
{
    __asm__ __volatile__(" csrw 0x830, %[id]\n" : : [id] "r"(id) : "memory");
}

/*! \fn inline void tensor_load (tensor_load *conf)
    \brief Tensor load instruction, it loads data from memory (bypass-ing the L1 cache) 
    into the L1 scratchpad. Input parameter defines the configuration to tensor load.
    \param use_tmask the tensor_mask register is used for this operation
    \param use_coop the operation is a cooperative tensor load.
    \param dst_start L1 Scratchpad starting cache line
    \param transformation These bits, along with bit 52, decodes the type of tensor operation.
    \param use_tenb This bit, along with transformation, decodes the type of tensor operation.
    \param addr tensor load address
    \param offset tensor load address offset
    \param num_lines tensor load number of cache lines
    \param stride tensor load stride value
    \param id tensor load id  
    \return none
    \tensorops Implementation of tensor_load api
*/
inline void __attribute__((always_inline)) tensor_load(uint8_t use_tmask, uint8_t use_coop,
    uint64_t dst_start, uint64_t transformation, uint64_t use_tenb, uint64_t addr, uint64_t offset,
    uint64_t num_lines, uint64_t stride, uint64_t id)
{
    uint64_t csr_enc = (((uint64_t)use_tmask & 1) << 63) | (((uint64_t)use_coop & 1) << 62) |
                       ((transformation & 0x7) << 59) | ((dst_start & 0x3F) << 53) |
                       ((use_tenb & 0x1) << 52) | ((addr & 0xFFFFFFFFFFC0ULL)) |
                       ((offset & 0x3) << 4) | ((num_lines & 0xF));
    register uint64_t x31_enc __asm__("x31") = (stride & 0xFFFFFFFFFFC0ULL) | (id & 0x1);

    __asm__ __volatile__("csrw 0x83f, %[csr_enc]\n"
                         :
                         : [x31_enc] "r"(x31_enc), [csr_enc] "r"(csr_enc)

    );
}


/*! \fn inline void tensor_store_scp(uint64_t entry_stride,
                                     uint64_t start_scp_entry,
                                     uint64_t Arows,
                                     uint64_t addr,
                                     uint64_t stride)
   \brief Tensor Store writes a series of 64-byte blocks of data from the L1 scratchpad into memory.  
   A matrix X can have up to 16 rows, and each row can be up to 64B in size (the number of columns depends on the type of elements of X).
   \param entry_stride Register stride
   \param start_scp_entry Start register
   \param Arows A matrix row size
   \param addr Virtual Address 
   \param stride This value is the distance in bytes between consecutive tensor rows in memory
   \return none
   \tensorops Implementation of tensor_store_scp api
*/
inline void __attribute__((always_inline)) tensor_store_scp(
    uint64_t entry_stride, uint64_t start_scp_entry, uint64_t Arows, uint64_t addr, uint64_t stride)
{
    uint64_t csr_enc = ((entry_stride & 0x3) << 62) | ((start_scp_entry & 0x3F) << 56) |
                       ((addr & 0xFFFFFFFFFFC0ULL)) | ((Arows & 0xF) << 51) | (((uint64_t)1) << 48);
    register uint64_t x31_enc __asm__("x31") = (stride & 0xFFFFFFFFFFC0UL);

    __asm__ __volatile__("csrw 0x87f, %[csr_enc]\n"
                         :
                         : [x31_enc] "r"(x31_enc), [csr_enc] "r"(csr_enc));
}

/*! \fn inline void tensor_store(uint64_t reg_stride,
                                 uint64_t start_reg,
                                 uint64_t cols,
                                 uint64_t Arows,
                                 uint64_t addr,
                                 uint64_t coop_store,
                                 uint64_t stride)
   \brief The Tensor store instruction reads a tensor from the vector register files and writes it to memory,
   bypassing the L1 data cache and the L2 cache. For the purposes of this instruction the tensor has ROWS+1 rows,
   and each row is 16*SIZE+16 bytes in size.
   \param reg_stride Register stride
   \param start_reg start register address
   \param cols  matrix row size.
   \param Arows  matrix row size
   \param addr Virtual Address 
   \param coop_store Number of minions to cooperate with
   \param stride This value is the distance in bytes between consecutive tensor rows in memory
   \return none
   \tensorops Implementation of tensor_store api
*/
inline void __attribute__((always_inline)) tensor_store(uint64_t reg_stride, uint64_t start_reg,
    uint64_t cols, uint64_t Arows, uint64_t addr, uint64_t coop_store, uint64_t stride)
{
    uint64_t warl = 0;
    uint64_t csr_enc = ((reg_stride & 0x3) << 62) | ((start_reg & 0x1F) << 57) |
                       ((cols & 0x3) << 55) | ((addr & 0xFFFFFFFFFFF0)) | ((Arows & 0xF) << 51) |
                       ((coop_store & 0x3) << 49) | ((warl & 0xF));

    register uint64_t x31_enc __asm__("x31") = (stride & 0xFFFFFFFFFF0UL);

    __asm__ __volatile__("csrw 0x87f, %[csr_enc]\n"
                         :
                         : [x31_enc] "r"(x31_enc), [csr_enc] "r"(csr_enc));
}

/*! \fn inline void tensor_fma(bool use_tmask,
                               uint64_t b_num_col,
                               uint64_t a_num_rows,
                               uint64_t a_num_cols, 
                               uint64_t offset, 
                               bool tenc_loc, 
                               bool tenb_unsigned, 
                               bool tena_unsigned, 
                               bool tenb_loc, 
                               uint64_t scp_loc_b, 
                               uint64_t scp_loc_a, 
                               uint64_t opcode, 
                               bool first_pass)
   \brief The Tensor FMA instruction multiplies two matrices A and B, optionally adds the resulting matrix
   to a third matrix C, and writes the result back onto matrix C
   \param use_tmask Use tensor_mask CSR to skip operations in an A row granularity.
   \param b_num_col B matrix number of columns
   \param a_num_rows A matrix number of rows
   \param a_num_cols A matrix number of columns
   \param offset A matrix starting column for the operation.
   \param tenc_loc Location of matrix C (0 = L1 scratchpad, 1 = memory).   
   \param tenb_unsigned TenB is signed (0) or unsigned (1).
   \param tena_unsigned TenA is signed (0) or unsigned (1).
   \param tenb_loc Location of matrix B (0 = L1 scratchpad, 1 = memory).
   \param scp_loc_b Starting L1 scratchpad cache line where matrix B is stored, ignored when xs[20] = 1.
   \param scp_loc_a Starting L1 scratchpad cache line where matrix A is stored, ignored when xs[20] = 1.
   \param opcode TensorType = 011
   \param first_pass if set to 0 then the initial value of TenC is added to the result
   \return none
   \tensorops Implementation of tensor_fma api
*/
inline void __attribute__((always_inline))
tensor_fma(uint8_t use_tmask, uint64_t b_num_col, uint64_t a_num_rows, uint64_t a_num_cols,
    uint64_t offset, uint8_t tenc_loc, uint8_t tenb_unsigned, uint8_t tena_unsigned, uint8_t tenb_loc,
    uint64_t scp_loc_b, uint64_t scp_loc_a, uint64_t opcode, uint8_t first_pass)
{
    uint64_t csr_enc =
        (((uint64_t)use_tmask & 1) << 63) | ((b_num_col & 0x3) << 55) | ((a_num_rows & 0xF) << 51) |
        ((a_num_cols & 0xF) << 47) | ((offset & 0xF) << 43) | (((uint64_t)tenc_loc & 1) << 23) |
        (((uint64_t)tena_unsigned & 1) << 22) | (((uint64_t)tenb_unsigned & 1) << 21) |
        (((uint64_t)tenb_loc & 1) << 20) | ((scp_loc_b & 0xFF) << 12) | ((scp_loc_a & 0xFF) << 4) |
        ((opcode & 0x7) << 1) | ((uint64_t)first_pass & 1);

    __asm__ __volatile__("csrw 0x801, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) :);
}

/*! \fn inline uint32_t tensor_reduce_uint32(uint32_t value, uint64_t operation, uint64_t partnerID, uint64_t action)
   \brief Tensor reduce allows a group of harts to communicate values held in floating-point registers to collectively calculate a reduction
   function. 
   \param value Register stride
   \param operation Function to be performed.
   \param partnerID Receiver minionID.
   \param action action value
   \return uint32_t value after reduction
   \tensorops Implementation of tensor_reduce_uint32 api
*/
inline uint32_t __attribute__((always_inline))
tensor_reduce_uint32(uint32_t value, uint64_t operation, uint64_t partnerID, uint64_t action)
{
    uint64_t warl = 0;
    uint32_t out;
    uint64_t csr_enc = ((warl & 0x2) << 62) | ((0ULL & 0x1F) << 57) | ((warl & 0x1FFFFFFF) << 28) |
                       ((operation & 0xF) << 24) | ((1ULL & 0xFF) << 16) |
                       ((partnerID & 0x1FFF) << 3) | ((warl & 0x1) << 2) | ((action & 0x3));

    __asm__ __volatile__("fmv.s.x     f0, %[value]\n"
                         "csrw 0x800, %[csr_enc]\n"
                         "fmv.x.s     %[out], f0\n"
                         : [out] "=r"(out)
                         : [csr_enc] "r"(csr_enc), [value] "r"(value)
                         : "f0");

    return out;
}




// Main entry point for MUL_MAT Q8_0 x F32 kernel
int entry_point(struct ggml_et_binary_params* params, void* env) {
    // Cast env to proper type
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    // Validate environment pointer
    // if (!kernel_env) {
    //     return -1;
    // }
    uint64_t hart_id = get_hart_id();
    uint64_t minion_id = (hart_id >> 1) & 0x1F; // 2 harts per minion
    uint64_t shire_id_orig = hart_id >> 6;
    
    // Get thread coordination info for B row-based parallelization
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

    // Handle invalid thread ID
    // if (thread_id < 0) {
    //     return -1;
    // }

    // HACK: Only use even thread IDs (thread 0, 2, 4, ...) to avoid resource contention
    // Each minion has 2 threads sharing instruction/data cache, NOC to RAM, and FPU
    // Odd threads return immediately to avoid fighting for shared resources
    // if (thread_id & 1) {
    //     return 0;  // Odd thread - skip work
    // }

    // Adjust thread count and ID for even-only threading
    // int effective_thread_id = thread_id / 2;
    // int effective_num_threads = (num_threads + 1) / 2;  // Ceiling division

    // Basic safety check on params
    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    // Extract tensor references
    struct ggml_tensor* src0 = &params->src0; // Weight matrix A (q8_0)
    struct ggml_tensor* src1 = &params->src1; // Activation matrix B (f32)
    struct ggml_tensor* dst = &params->dst;   // Output matrix C (f32)

    // Validate tensor types - src1 and dst must be f32, src0 can be quantized
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return -1; // Unsupported type combination - src1 and dst must be f32
    }

    // // FIXME: temporary, run F16 in single thread.
    // if (src0->type == GGML_TYPE_F16) {
    //     effective_num_threads = 1;
    //     if (effective_thread_id != 0) {
    //         return 0;
    //     }
    // }

    // Get data pointers
    const void* src0_data = src0->data;                          // Quantized weight blocks
    const float* src1_data = (const float*)src1->data;           // F32 activations
    float* dst_data = (float*)dst->data;                         // F32 output

    // Validate data pointers
    if (!src0_data || !src1_data || !dst_data) {
        return -1; // Null data pointer
    }

    // Determine block size based on src0 type
    int block_size;
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            block_size = QK8_0;
            break;
        case GGML_TYPE_F16:
            block_size = QK_F16;
            break;
        case GGML_TYPE_F32:
            block_size = QK_F32;
            break;
        default:
            return -1; // Unsupported src0 type
    }

    // Get matrix dimensions (following GGML convention)
    // src0 (A): [ne00=K, ne01=M, ne02, ne03]
    // src1 (B): [ne10=K, ne11=N, ne12, ne13]
    // dst (C):  [ne0=M, ne1=N, ne2, ne3]
    const int64_t K  = src0->ne[0];  // Hidden dimension
    const int64_t M  = src0->ne[1];  // Output features (rows in result)
    const int64_t N  = src1->ne[1];  // Sequence length (cols in result)

    // Higher dimensions for batch processing
    const int64_t ne02 = src0->ne[2];  // src0 batch dim 2
    const int64_t ne03 = src0->ne[3];  // src0 batch dim 3
    const int64_t ne12 = src1->ne[2];  // src1 batch dim 2
    const int64_t ne13 = src1->ne[3];  // src1 batch dim 3
    const int64_t ne2  = dst->ne[2];   // dst batch dim 2
    const int64_t ne3  = dst->ne[3];   // dst batch dim 3

    // Strides (in bytes)
    const size_t nb01 = src0->nb[1];   // src0 row stride
    const size_t nb02 = src0->nb[2];   // src0 batch stride 2
    const size_t nb03 = src0->nb[3];   // src0 batch stride 3
    const size_t nb11 = src1->nb[1];   // src1 row stride
    const size_t nb12 = src1->nb[2];   // src1 batch stride 2
    const size_t nb13 = src1->nb[3];   // src1 batch stride 3
    const size_t nb1  = dst->nb[1];    // dst row stride
    const size_t nb2  = dst->nb[2];    // dst batch stride 2
    const size_t nb3  = dst->nb[3];    // dst batch stride 3

    // Verify K dimension alignment for quantization
    // Q8_0 requires strict alignment (quantized data must be block-aligned)
    // F32 and F16 can handle partial blocks with scalar remainders
    if (src0->type == GGML_TYPE_Q8_0 && K % block_size != 0) {
        return -1; // Q8_0 requires K to be multiple of block_size
    }

    // Verify first dimension is contiguous (required assumption)
    size_t expected_element_size_src0;
    if (src0->type == GGML_TYPE_Q8_0) {
        expected_element_size_src0 = sizeof(block_q8_0);  // Q8_0 blocks
    } else if (src0->type == GGML_TYPE_F16) {
        expected_element_size_src0 = sizeof(uint16_t);    // F16 elements
    } else if (src0->type == GGML_TYPE_F32) {
        expected_element_size_src0 = sizeof(float);       // F32 elements
    } else {
        return -1; // Unsupported type
    }

    const size_t expected_element_size_src1 = sizeof(float);
    const size_t expected_element_size_dst = sizeof(float);

    if (src0->nb[0] != expected_element_size_src0 ||
        src1->nb[0] != expected_element_size_src1 ||
        dst->nb[0] != expected_element_size_dst) {
        return -1; // First dimension must be contiguous
    }

    const int64_t K_blocks = K / block_size;  // Number of quantized blocks per row

    // Threading: distribute output elements across threads
    const uint64_t total_elements = M * N * ne2 * ne3;

    // Cache line is 64 bytes = 16 floats
    // Cache line is 64 bytes = 16 floats
    const uint64_t per_thread = 1;  // Process 16 elements per thread per iteration
    int effective_num_threads = 1 << 10; //1024; 
    int effective_thread_id = hart_id >> 1; // x minion_id x
    
    const uint64_t threads_stride = per_thread * effective_num_threads;

    // Early exit if this thread is beyond the data
    if (effective_thread_id * per_thread >= total_elements) {
        return 0;
    }

    // Broadcasting support (exactly like GGML CPU backend)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

// Configuration for Tensor Operations (derived from user example and headers)
    // TL0 (Tensor Load) Configs
    const uint8_t TL0_USE_TMASK = 0;
    const uint8_t TL0_USE_COOP = 0;
    const uint8_t TL0_USE_TENB = 0;
    const uint64_t TL0_TRANSFORMATION = 0; // Default/No transform
    const uint64_t TL0_STRIDE = 0x40; // 64 bytes
    
    // TFMA (Tensor FMA) Configs
    const uint8_t TFMA_USE_TMASK = 0;
    const uint8_t TFMA_TENC_LOC = 0; // L1 scratchpad
    const uint8_t TFMA_TENB_UNSIGNED = 0;
    const uint8_t TFMA_TENA_UNSIGNED = 0;
    const uint8_t TFMA_TENB_LOC = 0; // L1 scratchpad
    const uint64_t TFMA_OPCODE = 3; // 011 for FMA
    
    // TSTORE (Tensor Store) Configs
    const uint64_t TSTORE_STRIDE = 0x40; // 64 bytes
    const uint64_t TSTORE_COOP = 0;


    // Process elements assigned to this thread
    for (uint64_t base_idx = effective_thread_id * per_thread; base_idx < total_elements; base_idx += threads_stride) {
        // Process up to per_thread elements
        for (uint64_t j = 0; j < per_thread; j++) {
            const uint64_t idx = base_idx + j;

            if (idx >= total_elements) {
                break;
            }

            // Decode linear index to (m, n, i2, i3)
            // Layout: m + M * (n + N * (i2 + ne2 * i3))
            const int64_t i3 = idx / (M * N * ne2);
            const int64_t rem3 = idx % (M * N * ne2);
            const int64_t i2 = rem3 / (M * N);
            const int64_t rem2 = rem3 % (M * N);
            const int64_t n = rem2 / M;
            const int64_t m = rem2 % M;

            // Broadcasting indices for src0
            const int64_t i03 = i3 / r3;
            const int64_t i02 = i2 / r2;

            // Broadcasting indices for src1
            const int64_t i13 = (ne13 > 1) ? i3 : 0;
            const int64_t i12 = (ne12 > 1) ? i2 : 0;

            // Compute dot product: A[m, :] . B[:, n]
            float sum = 0.0f;

            // Process full blocks
            for (int64_t kb = 0; kb < K_blocks; kb++) {
                // Get pointer to B column at row kb*block_size
                const float* b_col_start = (const float*)((const char*)src1_data +
                                                         (kb * block_size) * src1->nb[0] +
                                                         n * nb11 + i12 * nb12 + i13 * nb13);

                // Compute block dot product based on type
                switch (src0->type) {
                    case GGML_TYPE_Q8_0: {
                        const block_q8_0* q8_row = (const block_q8_0*)((const char*)src0_data +
                                                                       m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_q8_0(&q8_row[kb], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F16: {
                        const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data +
                                                                    m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_f16_naive(&f16_row[kb * block_size], b_col_start);
                        break;
                    }
                    case GGML_TYPE_F32: {
                        // const float* f32_row = (const float*)((const char*)src0_data +
                        //                                       m * nb01 + i02 * nb02 + i03 * nb03);
                        // sum += compute_block_dot_product_f32(&f32_row[kb * block_size], b_col_start);
                        // break;
                        const int BLOCK_M = 16;
                        const int BLOCK_N = 4; // 4 columns
                        const int BLOCK_K = 16;

                        for (int m = 0; m < M; m += BLOCK_M) {
                            for (int n = 0; n < N; n += BLOCK_N) {
                                
                                // Initialize Accumulator (C tile) in registers/scratchpad if needed
                                // Or use first_pass=true in FMA
                                
                                for (int k = 0; k < K; k += BLOCK_K) {
                                    // Load Tile A [BLOCK_M, BLOCK_K]
                                    // tensor_load(...) for A
                                    uint64_t addr_a = (uint64_t)src0_data + (m * K + k) * sizeof(float); // Simplified address
                                    tensor_load(TL0_USE_TMASK, TL0_USE_COOP, 0 /*dst_start*/, TL0_TRANSFORMATION, TL0_USE_TENB, 
                                            addr_a, 0 /*offset*/, BLOCK_M-1 /*num_lines*/, TL0_STRIDE, 0 /*id*/);
                                    
                                    tensor_wait(TENSOR_LOAD_WAIT_0);

                                    // Load Tile B [BLOCK_K, BLOCK_N]
                                    // tensor_load(...) for B
                                    // Note: B needs to be loaded carefully depending on layout (transposed?)
                                    // Assuming B is [K, N], we need to load columns. 
                                    // Tensor load typically loads rows. 
                                    // If B is column-major or we need to transpose, that's complex.
                                    // Assuming row-major B for now.
                                    uint64_t addr_b = (uint64_t)src1_data + (k * N + n) * sizeof(float);
                                    tensor_load(TL0_USE_TMASK, TL0_USE_COOP, 16 /*dst_start*/, TL0_TRANSFORMATION, TL0_USE_TENB, 
                                            addr_b, 0 /*offset*/, BLOCK_K-1 /*num_lines*/, TL0_STRIDE, 1 /*id*/);
                                    
                                    tensor_wait(TENSOR_LOAD_WAIT_1);

                                    // Perform FMA
                                    // C += A * B
                                    uint8_t first_pass = (k == 0);
                                    tensor_fma(TFMA_USE_TMASK, BLOCK_N-1 /*b_num_col*/, BLOCK_M-1 /*a_num_rows*/, BLOCK_K-1 /*a_num_cols*/, 
                                            0 /*offset*/, TFMA_TENC_LOC, TFMA_TENB_UNSIGNED, TFMA_TENA_UNSIGNED, TFMA_TENB_LOC, 
                                            16 /*scp_loc_b*/, 0 /*scp_loc_a*/, TFMA_OPCODE, first_pass);
                                    
                                    tensor_wait(TENSOR_FMA_WAIT);
                                }

                                // Store Result C [BLOCK_M, BLOCK_N]
                                uint64_t addr_c = (uint64_t)dst_data + (m * N + n) * sizeof(float);
                                tensor_store(0 /*reg_stride*/, 0 /*start_reg*/, BLOCK_N-1 /*cols*/, BLOCK_M-1 /*rows*/, 
                                            addr_c, TSTORE_COOP, TSTORE_STRIDE);
                                
                                tensor_wait(TENSOR_STORE_WAIT);
                            }
                        }
                    }
                    default:
                        return -1;
                }
            }

            // Handle partial block (remainder) for F32 and F16
            const int64_t K_remainder = K % block_size;
            if (K_remainder > 0 && src0->type != GGML_TYPE_Q8_0) {
                const int64_t remainder_offset = K_blocks * block_size;
                const float* b_col_start = (const float*)((const char*)src1_data +
                                                         remainder_offset * src1->nb[0] +
                                                         n * nb11 + i12 * nb12 + i13 * nb13);

                switch (src0->type) {
                    case GGML_TYPE_F16: {
                        const uint16_t* f16_row = (const uint16_t*)((const char*)src0_data +
                                                                    m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_f16_partial(&f16_row[remainder_offset], b_col_start, K_remainder);
                        break;
                    }
                    case GGML_TYPE_F32: {
                        const float* f32_row = (const float*)((const char*)src0_data +
                                                              m * nb01 + i02 * nb02 + i03 * nb03);
                        sum += compute_block_dot_product_f32_partial(&f32_row[remainder_offset], b_col_start, K_remainder);
                        break;
                    }
                    default:
                        break;
                }
            }

            // Store result using atomic store to avoid cache coherency issues
            // when multiple threads write to the same cache line
            volatile float* c_element = (volatile float*)((char*)dst_data +
                                       m * dst->nb[0] + n * nb1 + i2 * nb2 + i3 * nb3);
            atomic_store_f32(c_element, sum);
        }
    }

    return 0;
}



























// //******************************************************************************
// // MUL_MAT Kernel
// // Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// //******************************************************************************

// #include <stdint.h>
// #include "ggml_tensor.h"
// #include "platform.h"
// #include "math_fp.h"
// #include "quants.h"
// #include "block_ops.h"
// // #include "../../../../../../et-platform/et-common-libs/include/etsoc/isa/tensors.h"
// // #include "../../../../../../et-platform/et-common-libs/include/etsoc/isa/hart.h"

// KERNEL_TRAMPOLINE();



// // Main entry point for MUL_MAT Q8_0 x F32 kernel
// int entry_point(struct ggml_et_binary_params* params, void* env) {
//     // Cast env to proper type
//     kernel_environment_t* kernel_env = (kernel_environment_t*)env;

//     // Validate environment pointer
//     if (!kernel_env) {
//         return -1;
//     }

//     // Basic safety check on params
//     if (params == 0 || ((uint64_t)params & 0x7) != 0) {
//         return -1; // Invalid pointer
//     }

//     // Extract tensor references
//     struct ggml_tensor* src0 = &params->src0; // Weight matrix A (q8_0)
//     struct ggml_tensor* src1 = &params->src1; // Activation matrix B (f32)
//     struct ggml_tensor* dst = &params->dst;   // Output matrix C (f32)

//     // Validate tensor types - src1 and dst must be f32
//     if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
//         return -1; // Unsupported type combination - src1 and dst must be f32
//     }

//     // Get matrix dimensions
//     const int64_t K  = src0->ne[0];
//     const int64_t M  = src0->ne[1];
//     const int64_t N  = src1->ne[1];

//     // Get data pointers
//     const void* src0_data = src0->data;
//     const float* src1_data = (const float*)src1->data;
//     float* dst_data = (float*)dst->data;

//     if (!src0_data || !src1_data || !dst_data) {
//         return -1;
//     }

//     // Tensor operation parameters
//     // Assuming 32x32 tiling or similar based on hardware capability
//     // For now, we'll implement a basic loop structure that can be optimized
    
//     // Configuration for Tensor Operations (derived from user example and headers)
//     // TL0 (Tensor Load) Configs
//     const uint8_t TL0_USE_TMASK = 0;
//     const uint8_t TL0_USE_COOP = 0;
//     const uint8_t TL0_USE_TENB = 0;
//     const uint64_t TL0_TRANSFORMATION = 0; // Default/No transform
//     const uint64_t TL0_STRIDE = 0x40; // 64 bytes
    
//     // TFMA (Tensor FMA) Configs
//     const uint8_t TFMA_USE_TMASK = 0;
//     const uint8_t TFMA_TENC_LOC = 0; // L1 scratchpad
//     const uint8_t TFMA_TENB_UNSIGNED = 0;
//     const uint8_t TFMA_TENA_UNSIGNED = 0;
//     const uint8_t TFMA_TENB_LOC = 0; // L1 scratchpad
//     const uint64_t TFMA_OPCODE = 3; // 011 for FMA
    
//     // TSTORE (Tensor Store) Configs
//     const uint64_t TSTORE_STRIDE = 0x40; // 64 bytes
//     const uint64_t TSTORE_COOP = 0;

//     // Iterate over output tiles
//     // This is a simplified placeholder for the actual tiling logic
//     // Real implementation needs to handle M, N, K blocking
    
//     // For demonstration, we'll use the scalar fallback if not F32 for now
//     // or if dimensions are small
//     if (src0->type != GGML_TYPE_F32) {
//          // Fallback to original scalar implementation for non-F32 src0
//          // (Re-pasting the original scalar logic here would be too long, 
//          //  so I will assume we want to replace it. 
//          //  But since I need to be careful, I will only implement the F32 path with tensors)
//          return -1; // TODO: Re-implement scalar fallback or support Q8_0 tensors
//     }

//     // F32 Implementation using Tensors
//     // We need to load blocks of A and B, compute, and store C.
    
//     // Block sizes (hardware dependent, assuming 16x16 or similar from example)
//     const int BLOCK_M = 16;
//     const int BLOCK_N = 4; // 4 columns
//     const int BLOCK_K = 16;

//     for (int m = 0; m < M; m += BLOCK_M) {
//         for (int n = 0; n < N; n += BLOCK_N) {
            
//             // Initialize Accumulator (C tile) in registers/scratchpad if needed
//             // Or use first_pass=true in FMA
            
//             for (int k = 0; k < K; k += BLOCK_K) {
//                 // Load Tile A [BLOCK_M, BLOCK_K]
//                 // tensor_load(...) for A
//                 uint64_t addr_a = (uint64_t)src0_data + (m * K + k) * sizeof(float); // Simplified address
//                 tensor_load(TL0_USE_TMASK, TL0_USE_COOP, 0 /*dst_start*/, TL0_TRANSFORMATION, TL0_USE_TENB, 
//                            addr_a, 0 /*offset*/, BLOCK_M-1 /*num_lines*/, TL0_STRIDE, 0 /*id*/);
                
//                 tensor_wait(TENSOR_LOAD_WAIT_0);

//                 // Load Tile B [BLOCK_K, BLOCK_N]
//                 // tensor_load(...) for B
//                 // Note: B needs to be loaded carefully depending on layout (transposed?)
//                 // Assuming B is [K, N], we need to load columns. 
//                 // Tensor load typically loads rows. 
//                 // If B is column-major or we need to transpose, that's complex.
//                 // Assuming row-major B for now.
//                 uint64_t addr_b = (uint64_t)src1_data + (k * N + n) * sizeof(float);
//                 tensor_load(TL0_USE_TMASK, TL0_USE_COOP, 16 /*dst_start*/, TL0_TRANSFORMATION, TL0_USE_TENB, 
//                            addr_b, 0 /*offset*/, BLOCK_K-1 /*num_lines*/, TL0_STRIDE, 1 /*id*/);
                
//                 tensor_wait(TENSOR_LOAD_WAIT_1);

//                 // Perform FMA
//                 // C += A * B
//                 uint8_t first_pass = (k == 0);
//                 tensor_fma(TFMA_USE_TMASK, BLOCK_N-1 /*b_num_col*/, BLOCK_M-1 /*a_num_rows*/, BLOCK_K-1 /*a_num_cols*/, 
//                           0 /*offset*/, TFMA_TENC_LOC, TFMA_TENB_UNSIGNED, TFMA_TENA_UNSIGNED, TFMA_TENB_LOC, 
//                           16 /*scp_loc_b*/, 0 /*scp_loc_a*/, TFMA_OPCODE, first_pass);
                
//                 tensor_wait(TENSOR_FMA_WAIT);
//             }

//             // Store Result C [BLOCK_M, BLOCK_N]
//             uint64_t addr_c = (uint64_t)dst_data + (m * N + n) * sizeof(float);
//             tensor_store(0 /*reg_stride*/, 0 /*start_reg*/, BLOCK_N-1 /*cols*/, BLOCK_M-1 /*rows*/, 
//                         addr_c, TSTORE_COOP, TSTORE_STRIDE);
            
//             tensor_wait(TENSOR_STORE_WAIT);
//         }
//     }

//     return 0;
// }

