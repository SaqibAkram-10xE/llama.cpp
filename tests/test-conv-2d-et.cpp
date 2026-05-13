// Direct-launch correctness test for the conv_2d_f32_me ET kernel.
//
// The kernel uses a non-GGML layout contract (channels-innermost everywhere),
// so we can't go through GGML's CONV_2D op yet. Instead we allocate three
// tensors in ET memory shaped to the kernel's contract, upload deterministic
// data, call ggml_et_launch_kernel directly, download the result, and diff it
// against a CPU reference computed in the same layout.
//
// On success: prints "PASS" and exits 0.
// On failure: prints first mismatch and exits 1.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// Internal ET headers — needed for ggml_et_launch_kernel + binary_params.
// We add ggml/src/ggml-et to include path in CMakeLists.
#include "ggml-et-common.h"
#include "ggml-et-kernels.h"
#include "ggml-et-ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int N    = 1;
constexpr int H    = 32;
constexpr int W    = 32;
constexpr int Cin  = 512;
constexpr int Cout = 512;
constexpr int Kh   = 3;
constexpr int Kw   = 3;
constexpr int pad_h = Kh / 2;
constexpr int pad_w = Kw / 2;

// Deterministic but non-trivial fillers (avoid identical values that would
// hide subscript bugs).
float fill_input(int n, int h, int w, int c) {
    int k = ((n * 131 + h) * 137 + w) * 139 + c;
    return ((k % 31) - 15) * 0.05f;
}
float fill_filter(int kh, int kw, int ic, int oc) {
    int k = ((kh * 17 + kw) * 19 + ic) * 23 + oc;
    return ((k % 17) - 8) * 0.07f;
}

// CPU reference. GGML-standard NCHW input/output and OIHW filter (W innermost
// in input/output, Kw innermost in filter). Zero-padding, stride 1.
void cpu_reference(const std::vector<float> & in,
                   const std::vector<float> & flt,
                   std::vector<float> & out) {
    auto in_at  = [&](int n, int c, int h, int w) {
        return in[((n * Cin + c) * H + h) * W + w];
    };
    auto flt_at = [&](int oc, int ic, int kh, int kw) {
        return flt[((oc * Cin + ic) * Kh + kh) * Kw + kw];
    };
    auto out_at = [&](int n, int c, int h, int w) -> float& {
        return out[((n * Cout + c) * H + h) * W + w];
    };

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oh = 0; oh < H; ++oh) {
                for (int ow = 0; ow < W; ++ow) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < Kh; ++kh) {
                        const int ir = oh + kh - pad_h;
                        if (ir < 0 || ir >= H) continue;
                        for (int kw = 0; kw < Kw; ++kw) {
                            const int ic = ow + kw - pad_w;
                            if (ic < 0 || ic >= W) continue;
                            for (int ci = 0; ci < Cin; ++ci) {
                                acc += in_at(n, ci, ir, ic) *
                                       flt_at(oc, ci, kh, kw);
                            }
                        }
                    }
                    out_at(n, oc, oh, ow) = acc;
                }
            }
        }
    }
}

ggml_backend_t find_et_backend() {
    // Per-device names are "ET0", "ET1", ... — match by prefix.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("ET", 0) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

} // anonymous namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_t backend = find_et_backend();
    if (!backend) {
        fprintf(stderr, "ET backend not available — is the ET runtime present?\n");
        return 77;   // CTest "skip" exit code
    }

    // Build a small ggml context. Three tensors + a bit of slack.
    ggml_init_params init_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,    // backend will allocate device-side
    };
    ggml_context * ctx = ggml_init(init_params);

    // GGML ne is innermost-first. Standard CONV_2D layouts:
    //   input  ne = [W, H, Cin, N]      (NCHW with W innermost in memory)
    //   filter ne = [Kw, Kh, Cin, Cout] (OIHW with Kw innermost)
    //   output ne = [W, H, Cout, N]
    int64_t in_ne[4]  = { W,  H,  Cin,  N    };
    int64_t flt_ne[4] = { Kw, Kh, Cin,  Cout };
    int64_t out_ne[4] = { W,  H,  Cout, N    };

    ggml_tensor * t_in  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, in_ne);
    ggml_tensor * t_flt = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, flt_ne);
    ggml_tensor * t_out = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, out_ne);
    ggml_set_name(t_in,  "input");
    ggml_set_name(t_flt, "filter");
    ggml_set_name(t_out, "output");

    // Mirror ggml_conv_2d's op_params layout: [s0, s1, p0, p1, d0, d1].
    t_out->op_params[0] = 1;       // s0
    t_out->op_params[1] = 1;       // s1
    t_out->op_params[2] = pad_w;   // p0
    t_out->op_params[3] = pad_h;   // p1
    t_out->op_params[4] = 1;       // d0
    t_out->op_params[5] = 1;       // d1

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate ET tensors\n");
        return 1;
    }

    // ---- Fill host buffers ----------------------------------------------
    std::vector<float> in_host (N * H * W * Cin);
    std::vector<float> flt_host(Kh * Kw * Cin * Cout);
    std::vector<float> out_host(N * H * W * Cout, 0.0f);

    // NCHW input fill (W innermost).
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < Cin; ++c)
    for (int h = 0; h < H; ++h)
    for (int w = 0; w < W; ++w)
        in_host[((n * Cin + c) * H + h) * W + w] = fill_input(n, h, w, c);

    // OIHW filter fill (Kw innermost).
    for (int oc = 0; oc < Cout; ++oc)
    for (int ic = 0; ic < Cin;  ++ic)
    for (int kh = 0; kh < Kh;   ++kh)
    for (int kw = 0; kw < Kw;   ++kw)
        flt_host[((oc * Cin + ic) * Kh + kh) * Kw + kw] = fill_filter(kh, kw, ic, oc);

    ggml_backend_tensor_set(t_in,  in_host.data(),  0, in_host.size()  * sizeof(float));
    ggml_backend_tensor_set(t_flt, flt_host.data(), 0, flt_host.size() * sizeof(float));

    // Pre-fill the device-side output with a sentinel so we can tell whether
    // the kernel actually wrote it.
    std::vector<float> sentinel(out_host.size(), -987654.0f);
    ggml_backend_tensor_set(t_out, sentinel.data(), 0, sentinel.size() * sizeof(float));

    // ---- Direct kernel launch -------------------------------------------
    auto * dev_ctx =
        (ggml_backend_et_device_context *) backend->device->context;

    ggml_et_binary_params params{};
    params.src0 = *t_flt;
    params.src1 = *t_in;
    params.dst  = *t_out;

    bool ok = ggml_et_launch_kernel(dev_ctx,
                                    "conv_2d_f32_me",
                                    &params, sizeof(params),
                                    /*shire_mask=*/0xFFFFFFFFULL,
                                    /*enable_print=*/false,
                                    /*sync_error_check=*/true);
    if (!ok) {
        fprintf(stderr, "ggml_et_launch_kernel failed for conv_2d_f32_me\n");
        return 1;
    }

    // ---- Read back and compare ------------------------------------------
    ggml_backend_tensor_get(t_out, out_host.data(), 0, out_host.size() * sizeof(float));

    std::vector<float> ref(N * H * W * Cout, 0.0f);
    cpu_reference(in_host, flt_host, ref);

    // Tolerance: this is one F32 GEMM with ~144 multiply-adds per output
    // element (Kh*Kw*Cin = 144). Per-element rounding ~ 144 * eps * |max|.
    // Empirical bound 1e-4 absolute is comfortable.
    const float tol = 1e-4f;
    int    n_bad     = 0;
    int    first_bad = -1;
    float  max_err   = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float e = std::fabs(ref[i] - out_host[i]);
        if (e > max_err) max_err = e;
        if (e > tol) {
            if (first_bad < 0) first_bad = (int) i;
            ++n_bad;
        }
    }

    printf("conv_2d_f32_me: max_err = %g  bad = %d / %zu  (tol = %g)\n",
           max_err, n_bad, ref.size(), tol);

    if (n_bad > 0) {
        // NCHW decode: index = ((n*Cout + c)*H + h)*W + w
        int fi = first_bad;
        int w  = fi % W;    fi /= W;
        int h  = fi % H;    fi /= H;
        int c  = fi % Cout; fi /= Cout;
        int n  = fi;
        printf("  first bad at (n=%d, c=%d, h=%d, w=%d): ref=%g got=%g\n",
               n, c, h, w, ref[first_bad], out_host[first_bad]);
        return 1;
    }

    // Sanity: ensure we didn't accidentally read back the sentinel pattern.
    bool any_nonzero = false;
    for (float v : out_host) if (v != 0.0f) { any_nonzero = true; break; }
    if (!any_nonzero) {
        fprintf(stderr, "output is all zeros — likely the kernel didn't write\n");
        return 1;
    }

    printf("PASS\n");

    // ---- Mini perf bench (best-of-N, single op) -------------------------
    const int n_warm = 2;
    const int n_iter = 10;
    for (int i = 0; i < n_warm; ++i) {
        ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me",
                              &params, sizeof(params),
                              0xFFFFFFFFULL, false, true);
    }
    double best_us = 1e18;
    for (int i = 0; i < n_iter; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_et_launch_kernel(dev_ctx, "conv_2d_f32_me",
                              &params, sizeof(params),
                              0xFFFFFFFFULL, false, true);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (us < best_us) best_us = us;
    }
    // FLOPs: per output element = 2 * Kh*Kw*Cin (mul+add); total elements = N*Cout*H*W.
    const double flops = 2.0 * Kh * Kw * Cin * (double)N * Cout * H * W;
    const double gflops = flops / (best_us * 1e3);  // us → ns; flops/ns = Gflops
    printf("perf: shape (H=%d W=%d Cin=%d Cout=%d Kh=%d Kw=%d) best %.2f us = %.2f GF/s\n",
           H, W, Cin, Cout, Kh, Kw, best_us, gflops);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
