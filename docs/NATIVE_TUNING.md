# Tune native H3 generation

Use the default `max-quality` profile with a built native engine and a staged W4 pack. The two cache options apply to different parts of the DiT stage.

| Option | Scope | Cost on the tested model |
| --- | --- | --- |
| `--modulation-cache <directory>` | Reuses the six-step BF16 AdaLN tables across prompts, seeds, and image sizes | About 55.5 MiB per compatible table on disk |
| `--dit-cache-mib <MiB>` | Keeps packed DiT weights in memory for reuse across the six steps of one request | A configurable in-memory budget. Zero streams the weights. |

Weight prefetch is part of the default `max-quality` profile: the text and DiT stages read the next block's weights while the GPU computes the current one. Pass `--no-prefetch-weights` to compare against serial reads. The DiT stage prefetches only with a zero weight-cache budget, so `--dit-cache-mib` above zero replaces prefetch with the cache. [Round 1 measurements](OPTIMIZATION_CAMPAIGN.md#local-single-image-campaign)

Without a modulation cache, the first sampling step computes each AdaLN table just before its block uses it, which streams the 26 GB of projection weights alongside the block weights. That step is bounded by disk reads and takes about 5.5 s on the reference machine where the other five take about 2.7 s each. A modulation cache removes those reads as well as the table computation.

The modulation cache keys entries by the model file identity, PDD schedule, native build, GPU, and macOS version. A missing entry or checksum failure triggers recomputation. Entries are disposable and can be removed to reclaim space. Each request reports its modulation-cache status, including `miss` after computing and saving a table and `hit` after loading one. A request also reports `write_failed` if it could compute the table but could not save it. Check the cache directory and permissions if this status recurs.

## Pick a budget

On the M5 Max with 36 GiB of unified memory, the 10 GiB weight-cache budget improved the 1024×1024 teapot request with a warm modulation table. The 4096×4096 stress run with the same budget increased system swap by about 853 MiB. For 16 MP on this machine, use the modulation cache with the default `max-quality` weight budget. When tuning another workload, compare end-to-end latency and swap usage alongside DiT time. [Measured results](OPTIMIZATION_CAMPAIGN.md#cache-the-repeated-dit-weights)

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
