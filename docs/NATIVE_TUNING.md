# Tune native H3 generation

Use the default `max-quality` profile with a built native engine and a staged W4 pack. The two cache options apply to different parts of the DiT stage.

| Option | Scope | Cost on the tested model |
| --- | --- | --- |
| `--modulation-cache <directory>` | Sets the on-disk cache directory for the six-step BF16 AdaLN tables | About 55.5 MiB per compatible entry |
| `--no-modulation-cache` | Recomputes the AdaLN tables for each request | The first sampling step streams an additional 26 GB of projection weights on the H3 W4 pack |
| `--dit-cache-mib <MiB>` | Sets the in-memory budget for reusing packed DiT weights across the six sampling steps | Zero streams the weights |

The default `max-quality` profile prefetches text and DiT weights. Use `--no-prefetch-weights` to measure serial reads in both stages or `--no-dit-prefetch` to isolate DiT. A nonzero DiT weight-cache budget disables DiT prefetch by default. [Execution profiles](PROFILES.md)

The default cache directory is `~/Library/Caches/onyx-infer/modulation`. After writing a new entry, the launcher attempts to retain the two newest files by modification time. Manage directories specified with `--modulation-cache` yourself. Cache cleanup takes place after the output image is saved.

The modulation cache keys entries by the model file identity, PDD schedule, native build, GPU, and macOS version. Missing entries and checksum failures trigger recomputation. The launcher reports `miss` after computing and saving a table, `hit` after loading one, `write_failed` if saving fails, and `disabled` when caching is off. [Cold and warm cache measurements](../benchmarks/results/campaign-20261123/round5-default-modulation-cache.json)

## Pick a budget

Start with the default streaming budget. On the 36 GB M5 Max, a 10 GiB DiT weight cache increased 1024×1024 median wall time from 18.70 to 21.88 s and peak RSS from 1.80 to 6.35 GiB compared with the default prefetch path. An earlier 4096×4096 run with the same budget added about 853 MiB of swap. When tuning another workload, compare end-to-end latency, peak memory, and swap. [Prefetch comparison](../benchmarks/results/campaign-20261123/final-comparison.json) · [4K stress run](OPTIMIZATION_CAMPAIGN.md#cache-the-repeated-dit-weights)

A matched 4 GiB comparison on M5 Max tested default prefetch, fixed LRU, and static residency with bounded prefetch. At 1024², static residency reduced reads from 59.51 GB to 46.99 GB while end-to-end median increased from 17.31 to 17.97 s and peak DiT physical footprint increased from 6.44 to 8.81 GiB. The default remains the M5 Max baseline. [Measured 1024² and 4K comparisons](../benchmarks/results/m5-max-36gb-dit-shared-weight-budget-v1.json)

## Check the two caches locally

Run this from the repository root after [staging the model pack](VALIDATION.md#model-preparation). The first run creates a new table, the second reuses it, and the third also enables the 10 GiB weight budget. All three use the same prompt and seed.

```bash
trial=$(mktemp -d)
prompt=$(uv run python -c 'import json; print(json.load(open("benchmarks/requests/teapot-1024.json"))["request"]["prompt"])')

generate() {
  name=$1
  shift
  uv run onyx native-generate \
    --models artifacts/models/h3-w4 \
    --profile max-quality --width 1024 --height 1024 --seed 42 \
    --modulation-cache "$trial/modulation" --prompt "$prompt" \
    --output "$trial/$name.png" "$@" > "$trial/$name.json"
}

generate first --dit-cache-mib 0
generate warm --dit-cache-mib 0
generate resident --dit-cache-mib 10240
shasum -a 256 "$trial"/{first,warm,resident}.png
```

Compare the three PNG hashes. In each JSON record, inspect `total_seconds`, `dit_weight_cache_bytes`, and the first line of `stage_logs.dit.stdout`. That line is a JSON object with `modulation_cache` and the `weight_cache.hits` and `weight_cache.read_bytes` counters. The expected cache sequence is `miss`, `hit`, `hit`. The `resident` run should show weight hits and fewer bytes read. For timing claims, warm up first and repeat each arm in alternating orders. [Validation workflow](VALIDATION.md)

## Measure foreground DiT weight reads

For the 24 GB MacBook Air M4 baseline, run the teapot cases with a staged H3 W4 pack, a Release native build, VAE batch 25, and a fixed power source and power mode. The target file checks the actual Mac model, chip, and memory before generation. Warm both arms and keep the zero-cache prefetch policy in each arm.

```bash
for size in 1024 4096; do
  env -u ONYX_PROFILE uv run python benchmarks/ab_compare.py \
    --label "m4-weight-waits-${size}" --case "benchmarks/requests/teapot-${size}.json" \
    --target config/targets/m4-air-24gb.json --profile max-quality \
    --measure-dit --rounds 3 --warmup 1 \
    --arm 'default .build/onyx-native --dit-cache-mib 0 --dit-prefetch' \
    --arm 'diagnostic .build/onyx-native --dit-cache-mib 0 --dit-prefetch' \
    --env default=ONYX_VAE_BATCH=25 --env diagnostic=ONYX_VAE_BATCH=25 \
    --env default=ONYX_DIT_WEIGHT_PROFILE=0 --env diagnostic=ONYX_DIT_WEIGHT_PROFILE=1
done
```

The reports in `artifacts/benchmarks/ab` record both latency measurements, device and build identity, the staged model manifest and package report when present, system swap change, and separate DiT lifetime peak footprint and RSS from macOS `time -l`. Check `outputs_identical_across_arms` for exact latent and PNG hashes and use samples with a modulation cache hit. The `default` arm measures end-to-end latency with weight profiling disabled.

In the diagnostic arm, `stages.dit.foreground_weight_reads` groups synchronous reads and blocked prefetch lookups by DiT block. Keys outside the block weights appear under `other`. The `direct_ns` counter covers allocation and synchronous reads, and `blocked_prefetch_ns` records time waiting for in-flight weights. `prefetch_drain_ns` measures the wait at the final queue drain. A missing group had no recorded foreground reads or blocked lookups. Compare these diagnostics with `weight_cache.read_bytes` to see how much of the streamed traffic was exposed on the foreground path.

## DiT modulation for reference edits

Prefix `onyx native-generate` with `ONYX_DIT_INDEXED_MODULATION=1` to try the experimental path. It reads each token's modulation values from a compact table inside the normalization and residual graphs. A 16-reference, 1024×1024 output on the M5 Max measured a DiT child-process peak of 18.35 GiB with the option enabled and 20.22 GiB with the default path. Both outputs had the same PNG SHA-256. At 33,942 tokens the operator ABBA measured 79.25 ms on the default path and 45.39 ms with the option enabled. Another GPU job overlapped the candidate's complete generation, so the report excludes those end-to-end timings from speed comparisons. [Usage](QUICKSTART.md#edit-with-reference-images) · [Measurements](../benchmarks/results/m5-max-36gb-h3-ref16-adaln-v1.json)
