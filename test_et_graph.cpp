// #include "ggml.h"
// #include "ggml-backend.h"
// #include "ggml-et.h"
// #include <stdio.h>
// #include <stdlib.h>

// int main() {
//     printf("Testing ET graph transfer...\n");
    
//     // Initialize ET backend
//     ggml_backend_t backend = ggml_backend_et_init(0);
//     if (!backend) {
//         printf("Failed to initialize ET backend\n");
//         return 1;
//     }
    
//     printf("ET backend initialized successfully\n");
    
//     // Create a simple graph with a few nodes
//     struct ggml_init_params params = ggml_init_params_default();
//     ggml_context* ctx = ggml_init(params);
    
//     if (!ctx) {
//         printf("Failed to initialize GGML context\n");
//         ggml_backend_free(backend);
//         return 1;
//     }
    
//     // Create some tensors
//     ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
//     ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
//     ggml_tensor* result = ggml_mul(ctx, a, b);
    
//     // Set tensor names
//     ggml_set_name(a, "tensor_a");
//     ggml_set_name(b, "tensor_b");
//     ggml_set_name(result, "result");
    
//     // Build graph
//     ggml_cgraph* graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
//     ggml_build_forward_expand(graph, result);
    
//     printf("Created graph with %d nodes\n", graph->n_nodes);
    
//     // Test graph computation with ET backend
//     ggml_backend_buffer_type_t buft = ggml_backend_et_get_default_buffer_type(backend);
//     ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(a) + ggml_nbytes(b) + ggml_nbytes(result));
    
//     if (!buffer) {
//         printf("Failed to allocate ET buffer\n");
//         ggml_free(ctx);
//         ggml_backend_free(backend);
//         return 1;
//     }
    
//     // Set tensor buffers
//     ggml_backend_tensor_set(a, ggml_backend_buffer_get_base(buffer), 0, ggml_nbytes(a));
//     ggml_backend_tensor_set(b, ggml_backend_buffer_get_base(buffer), ggml_nbytes(a), ggml_nbytes(b));
//     ggml_backend_tensor_set(result, ggml_backend_buffer_get_base(buffer), ggml_nbytes(a) + ggml_nbytes(b), ggml_nbytes(result));
    
//     printf("Allocated ET buffer of size %zu bytes\n", ggml_backend_buffer_get_size(buffer));
    
//     // Compute graph
//     printf("Computing graph on ET device...\n");
//     ggml_status status = ggml_backend_graph_compute(backend, graph);
    
//     if (status == GGML_STATUS_SUCCESS) {
//         printf("Graph computation successful!\n");
//         printf("Graph was successfully transferred to ET device memory and executed.\n");
//     } else {
//         printf("Graph computation failed with status: %d\n", status);
//     }
    
//     // Cleanup
//     ggml_backend_buffer_free(buffer);
//     ggml_free_graph(graph);
//     ggml_free(ctx);
//     ggml_backend_free(backend);
    
//     printf("ET graph transfer test completed.\n");
//     return 0;
// }
