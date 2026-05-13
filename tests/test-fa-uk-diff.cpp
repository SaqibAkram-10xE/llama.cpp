// Diagnostic: compare flash_attn_ext_f16_me dst between standalone and uberkernel
// modes. Run twice — once with GGML_ET_UBERKERNEL=0, once with =1 — saving the
// dst tensor to a file. Then diff the two files. If they differ, FA-ME is
// producing different output inside the uberkernel batch than standalone; if
// they match, the bug is downstream (consumer-side / Q8_0 response to FA's
// output).
//
// Usage:
//   GGML_ET_UBERKERNEL=0 ./test-fa-uk-diff fa_dst_uk0.bin
//   GGML_ET_UBERKERNEL=1 ./test-fa-uk-diff fa_dst_uk1.bin
//   cmp fa_dst_uk0.bin fa_dst_uk1.bin
//   (or: pass two existing files to this binary to get a numeric diff)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Llama-3.2-1B-ish prefill shape. Small enough to be quick; uses GQA.
// Bug from full-graph bisection surfaced in PREFILL, so NQ > 1.
constexpr int DK     = 64;
constexpr int DV     = 64;
constexpr int NH_Q   = 32;
constexpr int NH_KV  = 8;
constexpr int NQ     = 16;    // prefill
constexpr int NK     = 64;    // small context (>= NQ for causal mask)

float fill_q(int d, int h)  { int k = (d * 131 + h * 137) % 1009;       return (k - 504) * 0.001f; }
float fill_k(int d, int t, int h) { int k = ((d * 139 + t) * 149 + h * 151) % 1013; return (k - 506) * 0.001f; }
float fill_v(int d, int t, int h) { int k = ((d * 157 + t) * 163 + h * 167) % 1019; return (k - 509) * 0.001f; }

ggml_backend_t find_et_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("ET", 0) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// f32 → f16 helpers (no GGML include needed for this; use packed-half conversion).
uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v{f};
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp   = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp <= 0)  return (uint16_t)sign;                       // flush subnormals to 0
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);            // ±inf / nan-ish
    return (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
}

int do_diff(const char * pa, const char * pb) {
    FILE * fa = std::fopen(pa, "rb");
    FILE * fb = std::fopen(pb, "rb");
    if (!fa || !fb) { fprintf(stderr, "open failed\n"); return 1; }
    std::fseek(fa, 0, SEEK_END); long la = std::ftell(fa); std::fseek(fa, 0, SEEK_SET);
    std::fseek(fb, 0, SEEK_END); long lb = std::ftell(fb); std::fseek(fb, 0, SEEK_SET);
    if (la != lb) { fprintf(stderr, "size mismatch: %ld vs %ld\n", la, lb); return 1; }
    std::vector<float> a(la / 4), b(lb / 4);
    std::fread(a.data(), 4, a.size(), fa);
    std::fread(b.data(), 4, b.size(), fb);
    std::fclose(fa); std::fclose(fb);

    double max_abs = 0, max_rel = 0, sum_abs = 0;
    int first_bad = -1;
    int n_nan_a = 0, n_nan_b = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i])) ++n_nan_a;
        if (std::isnan(b[i])) ++n_nan_b;
        double d = std::fabs((double)a[i] - (double)b[i]);
        sum_abs += d;
        if (d > max_abs) { max_abs = d; if (first_bad < 0 && d > 1e-6) first_bad = (int)i; }
        double denom = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
        if (denom > 1e-9) {
            double r = d / denom;
            if (r > max_rel) max_rel = r;
        }
    }
    printf("elems:      %zu (%ld bytes each)\n", a.size(), 4L);
    printf("max_abs:    %g\n", max_abs);
    printf("max_rel:    %g\n", max_rel);
    printf("mean_abs:   %g\n", sum_abs / a.size());
    printf("nans  A/B:  %d / %d\n", n_nan_a, n_nan_b);
    if (first_bad >= 0) {
        printf("first big diff at idx %d:  a=%g  b=%g\n", first_bad, a[first_bad], b[first_bad]);
    }
    if (max_abs < 1e-5 && n_nan_a == 0 && n_nan_b == 0) {
        printf("MATCH\n");
        return 0;
    }
    printf("DIFFER\n");
    return 2;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc >= 3) {
        // diff mode: compare two saved dumps
        return do_diff(argv[1], argv[2]);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.bin>          (run FA once, save dst)\n", argv[0]);
        fprintf(stderr, "       %s <a.bin> <b.bin>    (diff two saved dumps)\n", argv[0]);
        return 1;
    }
    const char * out_path = argv[1];

    ggml_backend_load_all();
    ggml_backend_t backend = find_et_backend();
    if (!backend) {
        fprintf(stderr, "ET backend not available\n");
        return 77;
    }

    const char * uk_env = std::getenv("GGML_ET_UBERKERNEL");
    fprintf(stderr, "FA-UK-DIFF: GGML_ET_UBERKERNEL=%s  out=%s\n",
            uk_env ? uk_env : "(unset)", out_path);

    ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);

    int64_t q_ne[4] = { DK, NQ, NH_Q,  1 };
    int64_t k_ne[4] = { DK, NK, NH_KV, 1 };
    int64_t v_ne[4] = { DV, NK, NH_KV, 1 };

    ggml_tensor * q = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, q_ne);
    ggml_tensor * k = ggml_new_tensor(ctx, GGML_TYPE_F16, 4, k_ne);
    ggml_tensor * v = ggml_new_tensor(ctx, GGML_TYPE_F16, 4, v_ne);
    ggml_set_name(q, "Q");
    ggml_set_name(k, "K");
    ggml_set_name(v, "V");

    float att_scale = 1.0f / std::sqrt((float)DK);

    // Match Llama-3.2-1B O-projection dims and chain N iterations of
    // FA → reshape → Q8_0 → reshape into one uberkernel batch. The bug likely
    // needs many FA+Q8_0 chains in a single launch to surface (one launch =
    // one forward pass = 16 layers in real llama-cli).
    constexpr int OUT_M  = DK * NH_Q;   // = 2048; lets us feed Q8_0 output back as next-iter Q
    int CHAIN = 16;                     // ~ Llama-3.2-1B layer count; override with FA_UK_CHAIN
    if (const char * c = std::getenv("FA_UK_CHAIN")) CHAIN = std::atoi(c);
    fprintf(stderr, "FA-UK-DIFF: CHAIN=%d  NQ=%d  NK=%d\n", CHAIN, NQ, NK);
    int64_t w_ne[4] = { DV * NH_Q, OUT_M, 1, 1 };
    ggml_tensor * w = ggml_new_tensor(ctx, GGML_TYPE_Q8_0, 4, w_ne);
    ggml_set_name(w, "Wo");

    // Two modes:
    //   FA_UK_MODE=chain (default) — FA→MM repeated CHAIN times. dst = last MM.
    //   FA_UK_MODE=mmfa            — single MM (feeds Q) → FA. dst = FA output.
    //                                 Isolates the MM→FA junction (no trailing MM,
    //                                 no leading FA). Minimum debug surface.
    const char * mode = std::getenv("FA_UK_MODE");
    if (!mode) mode = "chain";

    ggml_tensor * out = nullptr;
    ggml_tensor * readout = nullptr;  // tensor to dump (defaults to out)
    // Separate 2-D pre-projection activation tensor for mmfa mode (same byte
    // count as q, so qh fill data is reused).
    int64_t xpre_ne[4] = { DV * NH_Q, NQ, 1, 1 };
    ggml_tensor * x_pre = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, xpre_ne);
    ggml_set_name(x_pre, "x_pre");

    if (std::strcmp(mode, "mmfa") == 0) {
        ggml_tensor * mm_q   = ggml_mul_mat(ctx, w, x_pre);              // [2048, NQ]
        ggml_tensor * q_proj = ggml_reshape_4d(ctx, mm_q, DK, NQ, NH_Q, 1);
        ggml_tensor * fa     = ggml_flash_attn_ext(ctx, q_proj, k, v, nullptr,
                                                   att_scale, 0.0f, 0.0f);
        out = fa;
        fprintf(stderr, "FA-UK-DIFF: mode=mmfa (single MM->FA, dst=fa)\n");
    } else if (std::strcmp(mode, "famfa") == 0) {
        // FA -> MM -> FA. dst = second FA. Tests FA-after-FA contamination.
        ggml_tensor * fa1     = ggml_flash_attn_ext(ctx, q, k, v, nullptr, att_scale, 0.0f, 0.0f);
        ggml_tensor * fa1_flat= ggml_reshape_2d(ctx, fa1, DV * NH_Q, NQ);
        ggml_tensor * mm      = ggml_mul_mat(ctx, w, fa1_flat);          // [2048, NQ]
        ggml_tensor * q2      = ggml_reshape_4d(ctx, mm, DK, NQ, NH_Q, 1);
        ggml_tensor * fa2     = ggml_flash_attn_ext(ctx, q2, k, v, nullptr, att_scale, 0.0f, 0.0f);
        out = fa2;
        fprintf(stderr, "FA-UK-DIFF: mode=famfa (FA->MM->FA, dst=fa2)\n");
    } else if (std::strcmp(mode, "mmfam") == 0) {
        // MM -> FA -> MM. Models the real attention block: Q-proj, FA, Wo.
        ggml_tensor * mm_q   = ggml_mul_mat(ctx, w, x_pre);
        ggml_tensor * q_proj = ggml_reshape_4d(ctx, mm_q, DK, NQ, NH_Q, 1);
        ggml_tensor * fa     = ggml_flash_attn_ext(ctx, q_proj, k, v, nullptr, att_scale, 0.0f, 0.0f);
        ggml_tensor * fa_flat= ggml_reshape_2d(ctx, fa, DV * NH_Q, NQ);
        ggml_tensor * wo     = ggml_mul_mat(ctx, w, fa_flat);
        out = wo;
        fprintf(stderr, "FA-UK-DIFF: mode=mmfam (MM->FA->MM, dst=wo)\n");
    } else if (std::strcmp(mode, "famm") == 0) {
        // FA -> MM -> MM. Two trailing MMs, one FA.
        // FA_UK_DUMP=mm1|fa picks which intermediate to read after compute.
        // The graph always contains all 3 ops so the op stream is constant;
        // only the readout target changes.
        ggml_tensor * fa     = ggml_flash_attn_ext(ctx, q, k, v, nullptr, att_scale, 0.0f, 0.0f);
        ggml_tensor * fa_flat= ggml_reshape_2d(ctx, fa, DV * NH_Q, NQ);
        ggml_tensor * mm1    = ggml_mul_mat(ctx, w, fa_flat);
        ggml_tensor * mm2    = ggml_mul_mat(ctx, w, mm1);
        out = mm2;                                        // always the graph leaf
        const char * dump = std::getenv("FA_UK_DUMP");
        if (dump && std::strcmp(dump, "mm1") == 0) {
            readout = mm1;
            fprintf(stderr, "FA-UK-DIFF: mode=famm graph=mm2 readout=mm1\n");
        } else if (dump && std::strcmp(dump, "fa") == 0) {
            readout = fa;
            fprintf(stderr, "FA-UK-DIFF: mode=famm graph=mm2 readout=fa\n");
        } else {
            fprintf(stderr, "FA-UK-DIFF: mode=famm graph=mm2 readout=mm2\n");
        }
    } else if (std::strcmp(mode, "mmmm") == 0) {
        // MM -> MM (no FA). Tests if MM->MM is racy without any FA in launch.
        ggml_tensor * mm1 = ggml_mul_mat(ctx, w, x_pre);
        ggml_tensor * mm2 = ggml_mul_mat(ctx, w, mm1);
        out = mm2;
        fprintf(stderr, "FA-UK-DIFF: mode=mmmm (MM->MM, dst=mm2)\n");
    } else if (std::strcmp(mode, "fafa") == 0) {
        // FA -> FA (back-to-back, no MM between). Tests if FA-after-FA alone
        // is the bug. q2 is fa1 reshaped (metadata only, no compute).
        ggml_tensor * fa1 = ggml_flash_attn_ext(ctx, q, k, v, nullptr, att_scale, 0.0f, 0.0f);
        ggml_tensor * q2  = ggml_reshape_4d(ctx, fa1, DK, NQ, NH_Q, 1);
        ggml_tensor * fa2 = ggml_flash_attn_ext(ctx, q2, k, v, nullptr, att_scale, 0.0f, 0.0f);
        out = fa2;
        fprintf(stderr, "FA-UK-DIFF: mode=fafa (FA->FA, dst=fa2)\n");
    } else {
        ggml_tensor * q_cur = q;
        for (int i = 0; i < CHAIN; ++i) {
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, q_cur, k, v, /*mask=*/nullptr,
                                                   att_scale, /*max_bias=*/0.0f, /*softcap=*/0.0f);
            ggml_tensor * fa_flat = ggml_reshape_2d(ctx, fa, DV * NH_Q, NQ);
            ggml_tensor * mm = ggml_mul_mat(ctx, w, fa_flat);
            q_cur = ggml_reshape_4d(ctx, mm, DK, NQ, NH_Q, 1);
            out   = mm;
        }
        fprintf(stderr, "FA-UK-DIFF: mode=chain CHAIN=%d (dst=last mm)\n", CHAIN);
    }
    ggml_set_name(out, "chain_out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    // Fill Q (f32).
    std::vector<float> qh((size_t)DK * NQ * NH_Q);
    for (int h = 0; h < NH_Q; ++h)
    for (int t = 0; t < NQ;   ++t)
    for (int d = 0; d < DK;   ++d)
        qh[(h * NQ + t) * DK + d] = fill_q(d, h);
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));
    ggml_backend_tensor_set(x_pre, qh.data(), 0, qh.size() * sizeof(float));

    // Fill K, V (f16).
    std::vector<uint16_t> kh((size_t)DK * NK * NH_KV);
    std::vector<uint16_t> vh((size_t)DV * NK * NH_KV);
    for (int h = 0; h < NH_KV; ++h)
    for (int t = 0; t < NK;    ++t) {
        for (int d = 0; d < DK; ++d) kh[(h * NK + t) * DK + d] = f32_to_f16(fill_k(d, t, h));
        for (int d = 0; d < DV; ++d) vh[(h * NK + t) * DV + d] = f32_to_f16(fill_v(d, t, h));
    }
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size() * sizeof(uint16_t));

    // Build a deterministic f32 weight, quantize to Q8_0, then upload.
    const int64_t n_per_row = DV * NH_Q;
    std::vector<float> wf32((size_t)n_per_row * OUT_M);
    for (size_t i = 0; i < wf32.size(); ++i) {
        int kk = (int)((i * 173 + 991) % 1031);
        wf32[i] = (kk - 515) * 0.001f;
    }
    const size_t wq8_bytes = ggml_nbytes(w);
    std::vector<uint8_t> wq8(wq8_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf32.data(), wq8.data(),
                        /*start=*/0, /*nrows=*/OUT_M, n_per_row, /*imatrix=*/nullptr);
    ggml_backend_tensor_set(w, wq8.data(), 0, wq8_bytes);

    // Pre-fill dst with sentinel (so we'd notice if FA didn't write anything).
    std::vector<float> sentinel(ggml_nelements(out), -987654.0f);
    ggml_backend_tensor_set(out, sentinel.data(), 0, sentinel.size() * sizeof(float));

    // Build a 2-op graph (FA → scale) and compute it through the backend so
    // ggml_et_uberkernel_begin_graph / end_graph are exercised correctly when
    // GGML_ET_UBERKERNEL=1, with both ops batched together.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed: %d\n", (int)st);
        return 1;
    }

    if (!readout) readout = out;
    std::vector<float> out_host(ggml_nelements(readout));
    ggml_backend_tensor_get(readout, out_host.data(), 0, out_host.size() * sizeof(float));

    // Quick on-screen summary so it's obvious whether FA wrote anything.
    float mn = out_host[0], mx = out_host[0]; int n_sent = 0;
    for (float x : out_host) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        if (x == -987654.0f) ++n_sent;
    }
    fprintf(stderr, "chain dst:  nelems=%zu  min=%g  max=%g  untouched_sentinels=%d\n",
            out_host.size(), mn, mx, n_sent);
    fprintf(stderr, "first 8 values: %g %g %g %g %g %g %g %g\n",
            out_host[0], out_host[1], out_host[2], out_host[3],
            out_host[4], out_host[5], out_host[6], out_host[7]);
    fprintf(stderr, "last  8 values: %g %g %g %g %g %g %g %g\n",
            out_host[out_host.size()-8], out_host[out_host.size()-7],
            out_host[out_host.size()-6], out_host[out_host.size()-5],
            out_host[out_host.size()-4], out_host[out_host.size()-3],
            out_host[out_host.size()-2], out_host[out_host.size()-1]);

    FILE * fp = std::fopen(out_path, "wb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    std::fwrite(out_host.data(), 4, out_host.size(), fp);
    std::fclose(fp);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
