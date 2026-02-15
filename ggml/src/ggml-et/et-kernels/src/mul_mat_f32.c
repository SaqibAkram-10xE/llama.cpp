#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"




//
// MCACHE_CONTROL
//
inline void __attribute__((always_inline))
mcache_control(uint64_t d1_split, uint64_t scp_en, uint64_t cacheop_rate, uint64_t cacheop_max)
{
    uint64_t csr_enc = ((cacheop_max & 0x1F) << 6) | ((cacheop_rate & 0x7) << 2) |
                       ((scp_en & 0x1) << 1) | ((d1_split & 0x1) << 0);

    __asm__ __volatile__("csrw 0x7e0, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) : "x31");
}

/*
 * L1 SCP
 */
// Dcache configuration
#define L1D_NUM_SETS      16
#define L1D_NUM_WAYS      4
#define L1D_LINE_SIZE     64

#define NOP  __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n");
#define WFI __asm__ __volatile__ ("wfi\n");
#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n" : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n" : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n" : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n" : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n" : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n" : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n" : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n" : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n" : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n" : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n" : : );
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

#define EXCL_MODE(val) __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31"); 
#define MCACHE_CONTROL(x1, x2, x3, x4) __asm__ __volatile__("csrw 0x7e0, %0\n" : : "r"(((x1 & 0x1F) << 6) | ((x2 & 0x7) << 2) | ((x3 & 0x1) << 1) | ((x4 & 0x1) << 0)) : "x31");


static inline void evict_dcache(void)
{
    register uint64_t set asm("a7");
    for(set = 0; set < L1D_NUM_SETS; set++)
    {
        // use_tmask=0, dst=1 (L2/SP_RAM), set=X, way=0, num_lines=15
        __asm__ __volatile__(
            // Wait for previous memory accesses to finish
            "fence\n"
            // Evict L1 Dcache: EvictSW for the 4 ways
            "csrw evict_sw, %0\n"
            "addi %0, %0, 64\n"
            "csrw evict_sw, %0\n"
            "addi %0, %0, 64\n"
            "csrw evict_sw, %0\n"
            "addi %0, %0, 64\n"
            "csrw evict_sw, %0\n"
            "addi %0, %0, 64\n"
            // Wait for the evicts to complete
            "csrwi tensor_wait, 6\n"
            : 
            : "r"((1ull << 58) + ((set & 0xF) << 14) + 15ull)
            : "memory");
    }
	set = 0;
}

void setup_cache_scp(){
    // PRM-8: Cache Control Extension
    EXCL_MODE(1);
    // Evict the whole L1$
    evict_dcache();
    // Shared Mode
    MCACHE_CONTROL(0, 0, 0, 0);
    WAIT_CACHEOPS;
    // D1Split Mode
    MCACHE_CONTROL(0, 0, 0, 1);
    WAIT_CACHEOPS;
    // Scratchpad Mode
    MCACHE_CONTROL(0, 0, 1, 1);
    WAIT_CACHEOPS;
    EXCL_MODE(0);
}

	// evict_dcache();
	// setup_cache_scp();


// Hardware Constants derived from your auto-gen files
#define DESC_ACT_BASE   0x4000000000000000ULL
#define DESC_WGT_BASE   0x210000000000000FULL
#define FMA_CONFIG_INIT 0x187800000100003ULL 
#define SCP_ACT_OFFSET  16392
#define SCP_WGT_OFFSET  16588
#define SWIZZLE_ACT     0x200000000000000ULL
#define SWIZZLE_FMA     0x100ULL

// Address Encoding Helper
static inline uint64_t encode_addr(uint64_t base_desc, uint64_t phys_addr, uint32_t scp_line) {
    uint64_t scp_high = ((uint64_t)scp_line >> 2) << 48;
    uint64_t scp_low  = ((uint64_t)scp_line & 0x3) << 4;
    return base_desc | (phys_addr & 0xFFFFFFFFFFFFULL) | scp_high | scp_low;
}

int entry_point(struct ggml_et_binary_params* params, void* env) {

    evict_dcache();
	setup_cache_scp();
    uint64_t hart_id = get_hart_id();
    if (hart_id & 1) return 0; // Minions only
    
    uint64_t minion_id = (hart_id >> 1) & 0x1F;

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    const float* src0 = (const float*)params->src0.data;
    const float* src1 = (const float*)params->src1.data;
    float* dst        = (float*)params->dst.data;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff");

    // Loop over M (rows) - distributed across minions
    for (int64_t m = minion_id * 16; m < M; m += (32 * 16)) {
        for (int64_t n = 0; n < N; n += 16) {
            
            uint64_t addr_act = encode_addr(DESC_ACT_BASE, (uint64_t)(&src0[m * K]), SCP_ACT_OFFSET);
            uint64_t addr_wgt = encode_addr(DESC_WGT_BASE, (uint64_t)(&src1[n * K]), SCP_WGT_OFFSET);
            uint64_t cmd_fma  = FMA_CONFIG_INIT;

            for (int64_t k = 0; k < K; k += 16) {
                uint64_t cur_act_ptr = (uint64_t)(&src0[m * K + k]);
                uint64_t cur_wgt_ptr = (uint64_t)(&src1[n * K + k]);

                uint64_t load_act = (addr_act & 0xFFFF0000000000F0ULL) | (cur_act_ptr & 0xFFFFFFFFFFF0ULL);
                uint64_t load_wgt = (addr_wgt & 0xFFFF0000000000F0ULL) | (cur_wgt_ptr & 0xFFFFFFFFFFF0ULL);

                // Load Act: pass variables in, use 'mv' to satisfy the hardware requirement for x31
                __asm__ __volatile__ (
                    "li   x31, 0x40\n"
                    "csrw 0x83f, %[desc]" 
                    : : [desc] "r" (load_act) : "x31"
                );

                // Load Wgt
                __asm__ __volatile__ (
                    "li   x31, 0x41\n"
                    "csrwi 0x830, 0\n" 
                    "csrw 0x83f, %[desc]" 
                    : : [desc] "r" (load_wgt) : "x31"
                );

                // FMA
                uint64_t current_fma_cmd = (k == 0) ? (cmd_fma | 1) : (cmd_fma & ~1ULL);
                __asm__ __volatile__ (
                    "csrwi 0x830, 1\n"
                    "csrw 0x801, %[cmd]" 
                    : : [cmd] "r" (current_fma_cmd)
                );

                addr_act ^= SWIZZLE_ACT;
                cmd_fma  ^= SWIZZLE_FMA;
            }

            // Reduction
            __asm__ __volatile__(
                "li x31, 0x20003\n"
                "csrw 0x800, x31\n"
                "addi x31, x31, 8\n"
                "csrw 0x800, x31\n"
                "addi x31, x31, 8\n"
                "csrw 0x800, x31"
                : : : "x31"
            );

            // Store Result (Generic Stride)
            uint64_t store_desc = 0x4080000000000000ULL | (uint64_t)(&dst[n * M + m]);
            uint64_t stride_bytes = M * sizeof(float);

            __asm__ __volatile__(
                "mv   x31, %[stride]\n"    // Compiler will map 'stride_bytes' to a register and we move it to x31
                "csrw 0x87f, %[desc]\n"
                "csrwi 0x830, 8"
                : : [desc] "r" (store_desc), [stride] "r" (stride_bytes) : "x31"
            );
        }
    }
    __asm__ __volatile__("fence" ::: "memory");
    return 0;
}