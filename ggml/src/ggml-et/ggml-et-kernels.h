#pragma once

#include "ggml-et-common.h"
#include <string>
#include <vector>

#define ET_TRACE_BUFFER_SIZE (1024 * 1024 * 8UL)

// Load kernel from file or embedded data and store handle in device context
// Returns true on success, false on failure
//
// Loading strategy:
// - If GGML_ET_KERNELS_PATH env var is set: tries to load from ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf
// - If file not found or env var not set: falls back to embedded kernel data
// - Returns false if kernel cannot be loaded from either source
//
// Kernel is loaded using the device's default stream
bool ggml_et_load_kernel(ggml_backend_et_device_context* dev_ctx,
                         const std::string& kernel_name);

// Launch kernel with parameters on device's default stream
// Performs lazy loading: automatically loads kernel if not already loaded
// Kernel path: ${GGML_ET_KERNELS_PATH}/${kernel_name}.elf (default: /opt/et/ggml/kernels/)
// Returns true on success, false on failure
// Execution is synchronous - waits for completion
bool ggml_et_launch_kernel(ggml_backend_et_device_context* dev_ctx,
                           const std::string& kernel_name,
                           void* params,
                           size_t params_size,
                           uint64_t shire_mask = 0xFFFFFFFF,
                           bool enable_print = true,
                           bool sync_error_check = true);

// Allocate graph buffer in ET device memory and copy graph structure
// Returns device pointer to graph on success, nullptr on failure
// void* ggml_et_allocate_and_copy_graph(ggml_backend_et_device_context* dev_ctx,
//                                       const ggml_cgraph* cgraph);

// Create simple graph parameters for kernel launch
// Returns device pointer to params on success, nullptr on failure
void* ggml_et_create_graph_params(ggml_backend_et_device_context* dev_ctx,
                                  const ggml_cgraph* cgraph);

// Free graph buffer in ET device memory
// void ggml_et_free_graph_buffer(ggml_backend_et_device_context* dev_ctx);

// Unload kernel from device and free resources
// Safe to call even if kernel not loaded
void ggml_et_unload_kernel(ggml_backend_et_device_context* dev_ctx,
                           const std::string& kernel_name);

// Unload all kernels from device context
// Called during device cleanup
void ggml_et_unload_all_kernels(ggml_backend_et_device_context* dev_ctx);
