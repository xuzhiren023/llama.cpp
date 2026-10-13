# Isolated KNC text and image candidate

This optional runner uses the ordinary llama model loader and the upstream mtmd image pipeline. It preserves actual GGUF tensor types, including the Q6_K and floating-point tensors in Q3_K_M/Q4_K_M/Q5_K_M models. It does not force every weight to one quantization type.

The runner replaces the previous private text runner's `llama`/`lfm2` architecture allowlist with decoder capability checks. It accepts architectures implemented by the pinned llama source; this is not a promise that arbitrary checkpoints or every model supported by upstream work on KNC. Native execution, model quality, memory capacity and performance require separate qualification.

## Target models

| Model | Implementation | Required files |
|---|---|---|
| Existing Llama text models | `llama` | Text GGUF |
| Existing LFM2 text models | `lfm2` | Text GGUF |
| LFM2.5-8B-A1B | `lfm2moe`, hybrid short convolution, attention and routed experts | Text GGUF |
| Step3-VL-10B | `qwen3` text model and `step3vl` vision projector | Text GGUF plus matching projector GGUF |

The linked [LFM2.5 GGUF repository](https://huggingface.co/gaston-parravicini/LFM2.5-8B-A1B-Uncensored-Gaston-GGUF) supplies quantized text weights. The linked [Step3-VL repository](https://huggingface.co/huihui-ai/Huihui-Step3-VL-10B-abliterated) supplies a Transformers checkpoint. Its configuration at revision `82c3e704ed330acfcc37a1b01b5fde548ca29679` matches `conversion/step3.py`: `StepVLForConditionalGeneration`, Qwen3 text layers, 728-pixel vision size, 1536 vision channels, 47 vision layers and stride 2.

Convert a trusted local Step3 snapshot on a modern host, using this source's conversion dependencies:

```sh
python3 convert_hf_to_gguf.py /models/step3-snapshot --outfile step3-text-f16.gguf --outtype f16
python3 convert_hf_to_gguf.py /models/step3-snapshot --mmproj --outfile step3-mmproj-f16.gguf --outtype f16
llama-quantize step3-text-f16.gguf step3-text-Q4_K_M.gguf Q4_K_M
```

Keep text and projector from the same checkpoint. Embedding dimension checks cannot prove that unrelated checkpoints are semantically compatible. Do not quantize the projector as though it were a text-only K-quant model. Full checkpoint conversion and trained-model image quality remain separate from the synthetic tests below.

## Build without installing

The pinned compiler is MPSS GCC 5.1.1 targeting `k1om-mpss-linux`, with MPSS 3.8.6 sysroot. It uses `-mknc`, not an AVX-512 target. ICC is not required. The multimodal library uses C++14 on KNC and disables the ordinary x86 SIMD paths in image/audio decoding. Ordinary hosts retain C++17.

Inside the isolated toolchain environment:

```sh
bash examples/knc/build-candidates.sh "$PWD" /build/knc /artifacts/knc-models-r1
```

The output directory must be new. The script builds 3120A, 5110P and 7120P artifacts with hashes and ELF/ABI reports. It runs no native code, performs no installation and does not connect to any card. Prefetch values are inherited candidates, not new measured optima.

For an explicitly labelled Linux host reference, configure with KNC disabled, `LLAMA_BUILD_COMMON=OFF`, `LLAMA_BUILD_TOOLS=OFF`, `LLAMA_BUILD_APP=OFF`, `LLAMA_BUILD_MTMD=ON`, `LLAMA_BUILD_KNC_RUNNER=ON`, `LLAMA_KNC_HOST_REFERENCE=ON`, `LLAMA_KNC_TESTS=ON`, `MTMD_VIDEO=OFF`, `LLAMA_SUBPROCESS=OFF`. Disable CPU extensions unavailable on the host or emulator. Build `llama-knc-runner`, `knc-fixture-quantize`, `test-knc-mtmd` and `test-knc-mtmd-upstream`.

## Input contract

```sh
llama-knc-runner -m text.gguf -f formatted-prompt.txt -c 2048 -b 32 -t 180 -tb 180 -n 16
llama-knc-runner -m step3-text-Q4_K_M.gguf --mmproj step3-mmproj-f16.gguf \
    --image photo.png -f formatted-image-prompt.txt -c 2048 -b 32 -t 180 -tb 180 -n 16
```

These thread counts are examples for a 7120P only. Keep the existing topology-derived affinity/thermal guard outside the runner. Reserve the whole OS core; do not assume logical CPU numbering. Use multiples of 56/59/60 worker cores for 3120A/5110P/7120P respectively, and benchmark SMT occupancy after reconnecting. Linking libgomp does not make `KMP_AFFINITY` effective.

The caller renders the model's own chat template into UTF-8. Each image location contains one `<__media__>` marker. Supply images in marker order. Image boundaries, patch tokens, non-causal masks and M-RoPE are handled by mtmd. Reasoning tokens remain ordinary model output; the runner neither removes `<think>` sections nor invents a universal thinking template. Sampling is currently deterministic greedy (`--temp 0`).

Limits: four images, 32 MiB compressed bytes per image, 4096 pixels per side and a 1 MiB prompt file. Audio/video input is rejected. PNG/JPEG/BMP and other still formats supported by stb are accepted; WebP needs a different decoder and is not enabled in this build. Context shifting is disabled. Expanded input tokens and positions plus the reply must fit `-c`; non-causal image chunks must fit `-b`. Step3's fixed crop grid is not controlled by a generic image-token limit.

The text model uses mmap, but vision weights and scratch buffers still use RAM. An 8B MoE's roughly 1B active parameters do not mean only 1B parameters need storage. Step3's text model and large projector may exceed a smaller card's capacity even when the text model alone fits. Start with a small context and one small image. Do not advertise full-size Step3 support on 3120A/5110P until actual peak memory and image quality have passed.

## Small offline checks

Install NumPy and PyYAML in the isolated host environment, then:

```sh
python3 examples/knc/make-fixtures.py --quantizer /build/host/bin/knc-fixture-quantize --output /fixtures/knc-models
python3 examples/knc/test-runner.py --runner /build/host/bin/llama-knc-runner --fixtures /fixtures/knc-models --output /results/knc-models
/build/host/bin/test-knc-mtmd
/build/host/bin/test-knc-mtmd-upstream
```

Fixtures contain random weights with a small synthetic tokenizer. They exercise dense, convolution, expert routing and Step3 vision graphs, but provide no language/vision quality or representative tokens/sec evidence. Tests compare prefill vs incremental logits, state reset and chunking, repeated load/unload, image-dependent logits, and bounded input rejection. Native test invocations additionally require `--reference /results/host-reference --native-guarded`, the identical fixture manifest, a valid SKU thread count, and supervision by the existing external affinity/thermal guard. The flag acknowledges that guard; it does not implement temperature monitoring. Never report host/emulator throughput as KNC throughput.

For kernel isolation, `--baseline-kernels` disables activation staging and the extended Q2/Q3/Q4/Q6 dot kernels; it retains the original accelerated Q5 path. `--scalar-kernels` disables optimized dot kernels and staging. These controls compare kernel behavior within this candidate; neither is a substitute for timing the preserved installed runtime with identical weights, prompt, context, threads and cache state.

## Remaining optimization work

Activation staging still accelerates eligible dense Q3/Q4/Q5/Q6 matrix multiplications. Routed MoE `MUL_MAT_ID` uses the existing generic dispatch and KNC dot routines; expert-specific staging needs profiling before another optimization. Vision introduces substantial floating-point/convolution work that K-quant text kernels do not accelerate. After native correctness, measure text prefill, expert routing, image encoding and token decoding separately on each SKU, then select per-card settings from matched measurements.

Keep the compatibility changes, runner and backend optimization patches reviewable separately. This is a private candidate, not a claim of upstream acceptance or native qualification.
