// uk-diff: dump per-checkpoint tensor bytes during llama eval, for bit-diffing
// between GGML_ET_UBERKERNEL=0 and GGML_ET_UBERKERNEL=1 runs.
//
// Usage:
//   GGML_ET_UBERKERNEL=0 ./llama-uk-diff -m model.gguf -p "..." \
//       --dump-dir /tmp/no_uk --checkpoint '^(l_out-[0-9]+|result_norm|result_output)$'
//   GGML_ET_UBERKERNEL=1 ./llama-uk-diff ... --dump-dir /tmp/with_uk ...
//   diff -r /tmp/no_uk /tmp/with_uk   # first differing file = first divergent checkpoint
//
// The eval-scheduler honors `ask=true` by splitting the batch at that node, so
// uberkernel still batches the non-checkpoint nodes between consecutive checkpoints.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <vector>

struct uk_cb_data {
    std::vector<uint8_t> scratch;
    std::regex           checkpoint_re;
    std::string          dump_dir;
    int                  counter = 0;
};

static bool uk_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (uk_cb_data *) user_data;

    const bool match = std::regex_search(t->name, d->checkpoint_re);

    if (ask) {
        if (std::getenv("UK_DIFF_LIST_NAMES")) {
            std::fprintf(stderr, "uk-diff name: %s\n", t->name);
        }
        return match;
    }
    if (!match) {
        return true;
    }

    const size_t n_bytes = ggml_nbytes(t);
    d->scratch.resize(n_bytes);

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data = is_host ? (const uint8_t *) t->data : d->scratch.data();
    if (!is_host) {
        ggml_backend_tensor_get(t, d->scratch.data(), 0, n_bytes);
    }

    // sanitize name (replace '/' with '_')
    std::string safe = t->name;
    for (char & c : safe) if (c == '/' || c == ' ') c = '_';

    char path[1024];
    snprintf(path, sizeof(path), "%s/%04d_%s.bin", d->dump_dir.c_str(), d->counter++, safe.c_str());

    FILE * fp = std::fopen(path, "wb");
    if (!fp) {
        LOG_ERR("uk-diff: cannot open %s\n", path);
        return true;
    }
    std::fwrite(data, 1, n_bytes, fp);
    std::fclose(fp);

    LOG("uk-diff: dumped %4d  %-32s  %8zu bytes  (%s)\n",
        d->counter - 1, t->name, n_bytes, ggml_type_name(t->type));
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // pre-parse our own flags: --dump-dir, --checkpoint
    std::string dump_dir = "uk_dump";
    std::string checkpoint = "^(l_out-[0-9]+|result_norm|result_output)$";

    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            checkpoint = argv[++i];
        } else {
            passthrough.push_back(argv[i]);
        }
    }

    mkdir(dump_dir.c_str(), 0755);

    uk_cb_data cb_data;
    try {
        cb_data.checkpoint_re = std::regex(checkpoint);
    } catch (const std::regex_error & e) {
        std::fprintf(stderr, "bad --checkpoint regex: %s\n", e.what());
        return 1;
    }
    cb_data.dump_dir = dump_dir;

    common_params params;
    common_init();
    if (!common_params_parse((int) passthrough.size(), passthrough.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = uk_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, add_bos, true);
    if (toks.empty()) { LOG_ERR("empty prompt\n"); return 1; }

    const int n_decode = params.n_predict > 0 ? params.n_predict : 4;

    LOG_INF("uk-diff: prefill=%zu, then %d greedy decode step(s), dumping /%s/ to %s/\n",
            toks.size(), n_decode, checkpoint.c_str(), dump_dir.c_str());

    // prefill
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) {
        LOG_ERR("prefill decode failed\n"); return 1;
    }

    // greedy decode steps; deterministic so two runs are byte-comparable
    for (int s = 0; s < n_decode; ++s) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { LOG_ERR("no logits at step %d\n", s); return 1; }
        int best = 0;
        float best_v = logits[0];
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        llama_token tok = (llama_token) best;
        LOG_INF("uk-diff: step %d -> token %d (counter=%d)\n", s, tok, cb_data.counter);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode step %d failed\n", s); return 1;
        }
    }

    LOG_INF("uk-diff: wrote %d checkpoint dumps to %s/\n", cb_data.counter, dump_dir.c_str());
    llama_backend_free();
    return 0;
}
