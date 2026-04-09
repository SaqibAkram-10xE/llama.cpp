//******************************************************************************
// Element-wise Map F32 Kernel — standalone entry point
// Block helpers and el_map_f32 implementation are in el_map_f32.h
//******************************************************************************

#include "el_map_f32.h"

// Entry point for standalone kernel compilation
int entry_point(struct ggml_et_binary_params* params, void* env) {
    return el_map_f32(params, env);
}
