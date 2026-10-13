#ifdef PHI_KNC_KERNELS
#    include "knc-mq.h"
#endif
#include "mtmd-helper.h"
#include "mtmd.h"
#include "stb/stb_image.h"
// Prompts are formatted by the caller. Host builds are reference checks only.
#include "ggml-backend.h"
#include "llama.h"

#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

#ifndef PHI_PROFILE
#    define PHI_PROFILE "HOST_REFERENCE"
#endif
#ifndef PHI_UPSTREAM
#    define PHI_UPSTREAM "4413a192410fad1f054e9216114087a3e01527a0"
#endif
struct options {
    std::string              model, file, dump, mmproj;
    std::vector<std::string> images;
    std::vector<llama_token> stop_tokens;
    int                      context = 2048, batch = 32, threads = 4, prompt_threads = 4, predict = 128, cycles = 1;
    bool                     count = false, trace = false, validate = false, scalar = false, baseline = false;
};

static double now() {
    return llama_time_us() / 1000.0;
}

static long peak_kib() {
    struct rusage r;
    return getrusage(RUSAGE_SELF, &r) == 0 ? r.ru_maxrss : -1;
}

static long rss_kib() {
    std::ifstream f("/proc/self/status");
    std::string   k;
    long          n = 0;
    while (f >> k) {
        if (k == "VmRSS:") {
            f >> n;
            return n;
        }
        std::getline(f, k);
    }
    return -1;
}

static void require(bool yes, const char * why) {
    if (!yes) {
        throw std::runtime_error(why);
    }
}

static int integer(const char * s) {
    char * end = nullptr;
    errno      = 0;
    long n     = strtol(s, &end, 10);
    require(!errno && s != end && !*end && n >= 0 && n <= INT_MAX, "Invalid integer option");
    return int(n);
}

static std::string read_text(const std::string & name) {
    std::ifstream f(name, std::ios::binary);
    require(bool(f), "Cannot open input file");
    f.seekg(0, std::ios::end);
    const auto size = f.tellg();
    require(size >= 0 && size <= 1048576, "Prompt exceeds 1 MiB input bound");
    std::string s(size_t(size), '\0');
    f.seekg(0);
    if (size) {
        f.read(&s[0], size);
    }
    require(bool(f), "Cannot read complete input");
    return s;
}

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help") {
            puts(
                "KNC text/image runner; shared mmap required.\n"
                "-m MODEL -f FORMATTED_UTF8 -c CONTEXT -b BATCH -t THREADS -tb THREADS -n TOKENS\n"
                "auto-resources-v1 (model capacity and allocated RAM)\n"
                "--verbose-prompt (numeric count only) --print-token-count 1\n"
                "--validate-state DIRECTORY --cycles N --trace-kernels\n"
                "--mmproj MATCHING_GGUF --image FILE (repeatable, max 4; <__media__> markers in prompt)\n"
                "--stop-token ID (repeatable, at most 16 additional stop tokens)\n"
                "--no-mmap: explicitly rejected; no copy-mode fallback\n"
                "--describe: runtime identity; not model acceptance");
            exit(0);
        }
        if (a == "--describe") {
            printf(
                "{\"version\":\"knc-models-r1\",\"profile\":\"%s\",\"upstream\":\"%s\",\"kernels\":\"0.2.1\",\"imci_"
                "extension\":\"1.4.2 (Q2_K,Q3_K,Q4_K,Q6_K)\",\"hardware_validation\":\"pending unless locally "
                "tested\"}\n",
                PHI_PROFILE, PHI_UPSTREAM);
            exit(0);
        }
        if (a == "--no-mmap") {
            throw std::runtime_error("Shared mmap required; copy-mode fallback disabled");
        }
        if (a == "--trace-kernels") {
            o.trace = true;
            continue;
        }
        if (a == "--baseline-kernels") {
            o.baseline = true;
            continue;
        }
        if (a == "--scalar-kernels") {
            o.scalar = true;
            continue;
        }
        if (a == "--verbose-prompt") {
            o.count = true;
            continue;
        }
        if (a == "--no-display-prompt" || a == "--log-enable") {
            continue;
        }
        require(i + 1 < argc, "Option value missing");
        const char * v = argv[++i];
        if (a == "--stop-token") {
            require(o.stop_tokens.size() < 16, "Too many stop tokens");
            o.stop_tokens.push_back(integer(v));
        } else if (a == "--mmproj") {
            o.mmproj = v;
        } else if (a == "--image") {
            require(o.images.size() < 4, "At most four images per request");
            o.images.push_back(v);
        } else if (a == "-m") {
            o.model = v;
        } else if (a == "-f") {
            o.file = v;
        } else if (a == "-c") {
            o.context = integer(v);
        } else if (a == "-b") {
            o.batch = integer(v);
        } else if (a == "-t") {
            o.threads = integer(v);
        } else if (a == "-tb") {
            o.prompt_threads = integer(v);
        } else if (a == "-n") {
            o.predict = integer(v);
        } else if (a == "--cycles") {
            o.cycles = integer(v);
        } else if (a == "--validate-state") {
            o.validate = true;
            o.dump     = v;
        } else if (a == "-ngl") {
            require(integer(v) == 0, "Offload is not permitted");
        } else if (a == "--temp") {
            require(std::string(v) == "0", "This deterministic runner currently requires temperature zero");
        } else if (a == "--seed") {
            integer(v); /* greedy sampling has no RNG */
        } else if (a == "--print-token-count") {
            require(integer(v) == 1, "Invalid progress interval");
        } else {
            throw std::runtime_error("Unsupported runtime option: " + a);
        }
    }
    require(o.images.empty() == o.mmproj.empty(), "Image input requires both --mmproj and --image");
    require(!(o.scalar && o.baseline), "Choose only one kernel comparison mode");
    require(!o.model.empty() && !o.file.empty(), "Model and prompt file required");
    require(o.context >= 128 && o.batch >= 1 && o.batch <= o.context && o.predict < o.context && o.cycles >= 1 &&
                o.cycles <= 20,
            "Unsupported context/batch/output/cycle setting");
    require(o.threads >= 1 && o.threads <= 240 && o.prompt_threads >= 1 && o.prompt_threads <= 240,
            "Invalid worker count");
#ifndef PHI_HOST_REFERENCE
    struct utsname u;
    require(uname(&u) == 0 && std::string(u.machine) == "k1om", "Native K1OM required; no host fallback");
    const int cores = std::string(PHI_PROFILE) == "3120A" ? 56 : std::string(PHI_PROFILE) == "5110P" ? 59 : 60;
    require(o.threads % cores == 0 && o.threads <= 4 * cores && o.prompt_threads % cores == 0 &&
                o.prompt_threads <= 4 * cores,
            "Worker count disagrees with compiled SKU profile");
#else
    fprintf(stderr, "HOST_REFERENCE_ONLY: no hardware acceptance\n");
#endif
    return o;
}

struct session {
    llama_model *              model   = nullptr;
    llama_context *            context = nullptr;
    llama_batch                batch{};
    bool                       batch_live = false;
    std::vector<llama_token>   tokens;
    mtmd_context *             vision = nullptr;
    mtmd_input_chunks *        chunks = nullptr;
    std::vector<mtmd_bitmap *> bitmaps;
    size_t                     prompt_tokens = 0;
    llama_pos                  prompt_pos    = 0;
    int                        vocab_size    = 0;
    double                     load_ms       = 0;

    explicit session(const options & o, const std::string & prompt) {
        try {
            const double start = now();
            auto         mp    = llama_model_default_params();
            mp.n_gpu_layers    = 0;
            mp.load_mode       = LLAMA_LOAD_MODE_MMAP;
            model              = llama_model_load_from_file(o.model.c_str(), mp);
            require(model, "GGUF model load failed");
            require(llama_model_has_decoder(model) && !llama_model_has_encoder(model),
                    "This runner requires an autoregressive decoder model");
            require(o.context <= llama_model_n_ctx_train(model), "Context exceeds model training capacity");
            const auto * vocab = llama_model_get_vocab(model);
            vocab_size         = llama_vocab_n_tokens(vocab);
            for (auto stop : o.stop_tokens) {
                require(stop < vocab_size, "Stop token outside vocabulary");
            }
            if (o.images.empty()) {
                require(prompt.find(mtmd_default_marker()) == std::string::npos, "Media marker requires image input");
                int n = llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, false, true);
                require(n < 0, "Empty prompt or tokenizer sizing failed");
                tokens.resize(-n);
                n = llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), int(tokens.size()), false,
                                   true);
                require(n > 0, "Tokenization failed");
                tokens.resize(n);
                prompt_tokens = tokens.size();
                prompt_pos    = n;
            } else {
                auto vp            = mtmd_context_params_default();
                vp.use_gpu         = false;
                vp.n_threads       = o.prompt_threads;
                vp.warmup          = false;
                vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                vision             = mtmd_init_from_file(o.mmproj.c_str(), model, vp);
                require(vision && mtmd_support_vision(vision), "Matching vision projector required");
                bitmaps.reserve(o.images.size());
                for (const auto & path : o.images) {
                    std::ifstream image(path, std::ios::binary);
                    require(bool(image), "Cannot open image");
                    image.seekg(0, std::ios::end);
                    const auto bytes = image.tellg();
                    require(bytes > 0 && bytes <= 32 * 1024 * 1024, "Image file must be between 1 byte and 32 MiB");
                    std::vector<unsigned char> data((size_t(bytes)));
                    image.seekg(0);
                    image.read(reinterpret_cast<char *>(data.data()), bytes);
                    require(bool(image), "Image read failed");
                    int w = 0, h = 0, c = 0;
                    require(stbi_info_from_memory(data.data(), int(data.size()), &w, &h, &c) != 0,
                            "Expected a supported still image (PNG/JPEG/BMP); audio/video disabled");
                    require(w > 0 && h > 0 && w <= 4096 && h <= 4096, "Image dimensions exceed 4096 pixels per side");
                    auto media = mtmd_helper_bitmap_init_from_buf(vision, data.data(), data.size(), false,
                                                                  mtmd_helper_init_opt_default());
                    if (media.video_ctx) {
                        mtmd_helper_video_free(media.video_ctx);
                    }
                    if (media.bitmap) {
                        bitmaps.push_back(media.bitmap);
                    }
                    require(media.bitmap && !media.video_ctx && !mtmd_bitmap_is_audio(media.bitmap),
                            "Still image decode failed");
                }
                chunks = mtmd_input_chunks_init();
                require(chunks, "Cannot allocate media chunks");
                mtmd_input_text                  text = { prompt.data(), prompt.size(), false, true };
                std::vector<const mtmd_bitmap *> inputs(bitmaps.begin(), bitmaps.end());
                require(mtmd_tokenize(vision, chunks, &text, inputs.data(), inputs.size()) == 0,
                        "Image tokenization failed; one media marker per image is required");
                prompt_tokens = mtmd_helper_get_n_tokens(chunks);
                prompt_pos    = mtmd_helper_get_n_pos(chunks);
                require(prompt_tokens > 0 && prompt_pos > 0, "Empty multimodal prompt");
                for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
                    const auto * chunk = mtmd_input_chunks_get(chunks, i);
                    require(!mtmd_decode_use_non_causal(vision, chunk) ||
                                mtmd_input_chunk_get_n_tokens(chunk) <= size_t(o.batch),
                            "Non-causal image chunk exceeds batch; increase -b or reduce image size");
                }
            }
            const bool fits = prompt_tokens + size_t(o.predict) <= size_t(o.context - (o.count ? 4 : 0)) &&
                              int64_t(prompt_pos) + o.predict <= o.context;
            fprintf(stderr, fits ? "PHI_PROMPT_TOKENS %zu\n" : "PHI_PROMPT_REJECTED %zu\n", prompt_tokens);
            require(fits, "Expanded prompt plus reply exceeds allocated context; state shifting is disabled");
            auto cp            = llama_context_default_params();
            cp.n_ctx           = o.context;
            cp.n_batch         = o.batch;
            cp.n_ubatch        = o.batch;
            cp.n_seq_max       = 1;
            cp.n_threads       = o.threads;
            cp.n_threads_batch = o.prompt_threads;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.offload_kqv     = false;
            cp.op_offload      = false;
            cp.type_k          = GGML_TYPE_F16;
            cp.type_v          = GGML_TYPE_F16;
            cp.no_perf         = false;
            context            = llama_init_from_model(model, cp);
            require(context, "Context allocation failed");
            require(mtmd_helper_model_can_chat(context, vision), "Model does not support text generation");
            batch      = llama_batch_init(o.batch, 0, 1);
            batch_live = true;
            load_ms    = now() - start;
        } catch (...) {
            release();
            throw;
        }
    }

    void release() {
        if (batch_live) {
            llama_batch_free(batch);
            batch_live = false;
        }
        if (context) {
            llama_free(context);
            context = nullptr;
        }
        if (chunks) {
            mtmd_input_chunks_free(chunks);
            chunks = nullptr;
        }
        for (auto * bitmap : bitmaps) {
            mtmd_bitmap_free(bitmap);
        }
        bitmaps.clear();
        if (vision) {
            mtmd_free(vision);
            vision = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }

    session(const session &)             = delete;
    session & operator=(const session &) = delete;

    ~session() { release(); }

    void reset() {
        llama_memory_clear(llama_get_memory(context), true);
        llama_perf_context_reset(context);
    }

    void decode(const llama_token * t, int count, int pos, bool all) {
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i]     = t[i];
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = all || i == count - 1;
        }
        require(llama_decode(context, batch) == 0, "Decode failed; session must be reset");
    }

    void prefill(int batch_size) {
        if (chunks) {
            llama_pos end = 0;
            require(mtmd_helper_eval_chunks(vision, context, chunks, 0, 0, batch_size, true, &end) == 0,
                    "Image/text evaluation failed");
            require(end == prompt_pos, "Unexpected multimodal position count");
        } else {
            for (size_t pos = 0; pos < tokens.size();) {
                const int n = int(std::min(size_t(batch_size), tokens.size() - pos));
                decode(tokens.data() + pos, n, int(pos), false);
                pos += n;
            }
        }
    }

    llama_token greedy() {
        const float * p = llama_get_logits_ith(context, -1);
        require(p, "Missing logits");
        int best = 0;
        for (int i = 0; i < vocab_size; ++i) {
            require(std::isfinite(p[i]), "Non-finite logits");
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }
};

static void write_floats(const std::string & file, const std::vector<float> & v) {
    std::ofstream f(file, std::ios::binary);
    require(bool(f), "Cannot create validation output");
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
    require(bool(f), "Validation write failed");
}

static std::vector<float> logits_run(session & s, int chunk) {
    s.reset();
    std::vector<float> out;
    out.reserve(s.tokens.size() * s.vocab_size);
    for (size_t pos = 0; pos < s.tokens.size();) {
        int n = std::min(size_t(chunk), s.tokens.size() - pos);
        s.decode(s.tokens.data() + pos, n, int(pos), true);
        for (int i = 0; i < n; ++i) {
            auto p = llama_get_logits_ith(s.context, i);
            require(p, "Missing validation logits");
            out.insert(out.end(), p, p + s.vocab_size);
        }
        pos += n;
    }
    return out;
}

static void compare(const std::vector<float> & a, const std::vector<float> & b, const char * label) {
    require(a.size() == b.size(), "Validation shape mismatch");
    double sq = 0, ref = 0, max_abs = 0;
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "Non-finite validation logits");
        double d = double(a[i]) - b[i];
        sq += d * d;
        ref += double(a[i]) * a[i];
        max_abs = std::max(max_abs, std::abs(d));
        if (std::abs(d) > 0.03 + 0.005 * std::abs(double(a[i]))) {
            ++bad;
        }
    }
    const double nr = std::sqrt(sq / std::max(ref, 1e-30));
    fprintf(stderr, "PHI_NUMERICS %s count=%zu max_abs=%.9g nrmse=%.9g outliers=%zu\n", label, a.size(), max_abs, nr,
            bad);
    require(nr <= 0.002 && bad == 0, "State/prefill numerical tolerance exceeded");
}

static void validate(session & s, const options & o, int cycle) {
    if (s.chunks) {
        s.reset();
        s.prefill(o.batch);
        const float * p = llama_get_logits_ith(s.context, -1);
        require(p, "Missing image logits");
        std::vector<float> first(p, p + s.vocab_size);
        s.reset();
        s.prefill(o.batch);
        p = llama_get_logits_ith(s.context, -1);
        require(p, "Missing reset image logits");
        std::vector<float> second(p, p + s.vocab_size);
        compare(first, second, "image_reset_vs_fresh");
        const std::string base = o.dump + "/cycle-" + std::to_string(cycle);
        write_floats(base + "-image.f32", first);
        write_floats(base + "-image-reset.f32", second);
        fprintf(stderr, "PHI_VALIDATION image_tokens=%zu positions=%d vocab=%d\n", s.prompt_tokens, s.prompt_pos,
                s.vocab_size);
        return;
    }
    require(s.tokens.size() <= 64, "Validation prompt must be at most 64 tokens");
    auto        prefill     = logits_run(s, o.batch);
    auto        incremental = logits_run(s, 1);
    std::string base        = o.dump + "/cycle-" + std::to_string(cycle);
    write_floats(base + "-prefill.f32", prefill);
    write_floats(base + "-incremental.f32", incremental);
    compare(prefill, incremental, "prefill_vs_incremental");
    // Poison both attention and convolution states, then clear and rerun.
    llama_token poison = 0;
    for (int i = 0; i < 4; ++i) {
        s.decode(&poison, 1, int(s.tokens.size()) + i, false);
    }
    auto reset = logits_run(s, o.batch);
    compare(prefill, reset, "reset_vs_fresh");
    auto chunked = logits_run(s, std::min(3, o.batch));
    compare(prefill, chunked, "three_token_chunks");
    write_floats(base + "-chunked.f32", chunked);
    write_floats(base + "-reset.f32", reset);
    std::ofstream ids(base + "-tokens.txt");
    for (auto t : s.tokens) {
        ids << t << '\n';
    }
    require(bool(ids), "Token evidence write failed");
    fprintf(stderr, "PHI_VALIDATION cycle=%d tokens=%zu vocab=%d peak_rss_kib=%ld rss_kib=%ld\n", cycle,
            s.tokens.size(), s.vocab_size, peak_kib(), rss_kib());
}

static void generate(session & s, const options & o) {
    s.reset();
    const double start = now();
    s.prefill(o.batch);
    fprintf(stderr, "Tokens consumed so far = %zu\n", s.prompt_tokens);
    const double prefill_ms = now() - start;
    double       ttft_ms = 0, decode_ms = 0;
    int          produced = 0, evals = 0;
    const auto * vocab = llama_model_get_vocab(s.model);
    char         piece[32768];
    for (int i = 0; i < o.predict; ++i) {
        llama_token t = s.greedy();
        if (llama_vocab_is_eog(vocab, t) ||
            std::find(o.stop_tokens.begin(), o.stop_tokens.end(), t) != o.stop_tokens.end()) {
            break;
        }
        // Preserve reasoning delimiters for the caller's model-specific parser.
        int n = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
        require(n >= 0, "Token piece exceeds reusable output buffer");
        if (!produced) {
            ttft_ms = now() - start;
        }
        require(fwrite(piece, 1, n, stdout) == size_t(n), "Output stream failed");
        fflush(stdout);
        ++produced;
        if (i + 1 < o.predict) {
            double t0 = now();
            s.decode(&t, 1, s.prompt_pos + i, false);
            decode_ms += now() - t0;
            ++evals;
        }
        fprintf(stderr, "Tokens consumed so far = %zu\n", s.prompt_tokens + produced);
    }
    fprintf(stderr, "llama_print_timings: load time = %.3f ms\n", s.load_ms);
    fprintf(
        stderr,
        "llama_print_timings: prompt eval time = %.3f ms / %zu tokens ( %.3f ms per token, %.3f tokens per second)\n",
        prefill_ms, s.prompt_tokens, prefill_ms / std::max(size_t(1), s.prompt_tokens),
        1000 * s.prompt_tokens / std::max(prefill_ms, 1e-9));
    fprintf(stderr, "llama_print_timings: eval time = %.3f ms / %d runs ( %.3f ms per token, %.3f tokens per second)\n",
            decode_ms, evals, decode_ms / std::max(1, evals), 1000 * evals / std::max(decode_ms, 1e-9));
    fprintf(stderr,
            "PHI_METRICS "
            "{\"load_ms\":%.3f,\"prefill_ms\":%.3f,\"ttft_after_load_ms\":%.3f,\"ttft_with_load_ms\":%.3f,\"decode_"
            "eval_ms\":%.3f,\"decode_evals\":%d,\"generated_tokens\":%d,\"peak_rss_kib\":%ld,\"trace_enabled\":%s}\n",
            s.load_ms, prefill_ms, ttft_ms, s.load_ms + ttft_ms, decode_ms, evals, produced, peak_kib(),
            o.trace ? "true" : "false");
}

int main(int argc, char ** argv) {
    try {
        options o = parse(argc, argv);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_nested(0);
        omp_set_max_active_levels(1);
#endif
#ifdef PHI_KNC_KERNELS
        if (o.scalar || o.baseline) {
            require(setenv("GGML_KNC_MQ", "0", 1) == 0, "Cannot disable staging for comparison");
        }
        ggml_knc_mq_init();
#else
        require(!o.scalar && !o.baseline && !o.trace, "KNC diagnostics require a KNC build");
#endif
        llama_backend_init();
#ifdef PHI_KNC_KERNELS
        ggml_knc_trace_enable(o.trace ? 1 : 0);
        ggml_knc_optimized_enable(o.scalar ? 0 : 1);
        ggml_knc_extended_enable(o.baseline ? 0 : 1);
        ggml_knc_mq_report();
#endif
        const std::string prompt = read_text(o.file);
        for (int cycle = 0; cycle < o.cycles; ++cycle) {
            {
                session s(o, prompt);
                if (o.validate) {
                    validate(s, o, cycle);
                } else if (o.predict && !o.count) {
                    generate(s, o);
                }
            }
            fprintf(stderr, "PHI_UNLOADED cycle=%d rss_kib=%ld peak_rss_kib=%ld\n", cycle, rss_kib(), peak_kib());
        }
#ifdef PHI_KNC_KERNELS
        if (o.trace) {
            ggml_knc_trace_report();
            ggml_knc_mq_report();
        }
#endif
        llama_backend_free();
        if (o.validate) {
            puts("VALIDATION_COMPLETE: fixed-input numerical evidence written; compare with the pinned reference.");
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "PHI_ERROR: %s\n", e.what());
        return 1;
    }
}
