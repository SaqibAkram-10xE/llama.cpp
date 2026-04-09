#pragma once

// Enable monolithic compute path: op_cgraph_2.c includes all kernel implementations
// and ggml-et.cpp uses ggml_et_op_cg() instead of individual kernel dispatch
#define ENABLE_MONOLITHIC_COMPUTE
