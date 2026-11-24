# Generated-image evidence for `opt/local-single-image-20261123`

This branch holds the pull request's generated-image evidence and nothing else. It is not meant to be merged and can be deleted once the pull request closes. The [contribution guide](https://github.com/phantom5125/onyx-infer/blob/main/CONTRIBUTING.md#generated-image-evidence) asks for ad hoc evidence to be attached to the pull request rather than committed to the product branch.

- Base revision: `35bc6f9` (`main`), native binary SHA-256 `bdeb725f3a49b7f697d635e781d9b4631c69eacda16e0a3fc4364df1a3c15fd9`
- Candidate revision: `e0b4244` (this pull request), native binary SHA-256 `ec58291ce7f112f40170a84de8647a74dcfe1dc727ee07d4d54bbd34c7e7f5f3`
- Hardware: Apple M5 Max, 32 GPU cores, 36 GB unified memory, macOS 27.2.1; Release builds
- Model: MiniMax-H3-Image W4 package, source revision `df334556f585`, recipe `h3.pdd6.v1`, six sampling steps, seed 42

`manifest.json` lists every file with its SHA-256, and `reports/` holds the interleaved A/B reports (`pr-*.json`) and the `onyx.ci.v1` validation reports and summaries for both revisions.

## 1024×1024, `max-quality`

**Teapot, `benchmarks/requests/teapot-1024.json`**

| | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Baseline `35bc6f9` | [`1024/teapot-baseline-35bc6f9.png`](1024/teapot-baseline-35bc6f9.png) | 1.33 MiB | `4160adb7ac0aa777d6cfe8bafaa9284dd0202198a3ea289c377edf8e93b577d1` |
| Candidate `e0b4244` | [`1024/teapot-candidate-e0b4244.png`](1024/teapot-candidate-e0b4244.png) | 1.33 MiB | `6fa24b90ed429f699f8ff39b6f46361bb0a6ddba0a68bdc72ccde0f37efbf3ac` |

Three interleaved runs per arm: 33.54 s against 19.58 s median. 653 of 50,331,648 channel values differ, each by at most 1 level of 255

**Moka pot blueprint, `benchmarks/requests/text-baseline-moka-pot.json`**

| | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Baseline `35bc6f9` | [`1024/moka-pot-baseline-35bc6f9.png`](1024/moka-pot-baseline-35bc6f9.png) | 1.53 MiB | `2219cae12b928f46c4744256d3fa500bc5d2aaa0e230c85beb67eb1236c26053` |
| Candidate `e0b4244` | [`1024/moka-pot-candidate-e0b4244.png`](1024/moka-pot-candidate-e0b4244.png) | 1.53 MiB | `6f6fc7faf79704c300610b82d04a90246bb4cfe5185f39ec5752db67b708a1ae` |

One run per arm. 71 of 3,145,728 channel values differ, each by at most 1 level of 255

**Jet engine, `benchmarks/requests/text-baseline-jet-engine.json`**

| | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Baseline `35bc6f9` | [`1024/jet-engine-baseline-35bc6f9.png`](1024/jet-engine-baseline-35bc6f9.png) | 1.63 MiB | `234e4c17fd5cba139dc9f1f842329789e829651551e0f3e22c9f4fc8e29413e0` |
| Candidate `e0b4244` | [`1024/jet-engine-candidate-e0b4244.png`](1024/jet-engine-candidate-e0b4244.png) | 1.63 MiB | `c1fa829c0692220f3472a5203692a413ad099d6f7704778802b3cb9299e1d231` |

One run per arm. 71 of 3,145,728 channel values differ, each by at most 1 level of 255

## 4096×4096, `max-quality`

**Teapot, `benchmarks/requests/teapot-4096.json`**

| | File | Size | SHA-256 |
| --- | --- | ---: | --- |
| Baseline `35bc6f9` | [`4096/teapot-baseline-35bc6f9.png`](4096/teapot-baseline-35bc6f9.png) | 20.14 MiB | `0b9761b2416b57639d9fe387a528136c072b867fa54f394399a651be36557e3f` |
| Candidate `e0b4244` | [`4096/teapot-candidate-e0b4244.png`](4096/teapot-candidate-e0b4244.png) | 20.14 MiB | `c68d8fbe635ade8748f2be69477bc5d26f5493a5322d9052428e07d3e59d0c43` |

One run per arm: 323.04 s against 227.93 s. 653 of 50,331,648 channel values differ, each by at most 1 level of 255. `4096/preview/` holds 2048 px JPEG previews and `4096/detail/` holds 100% crops of the knob, spout, cup, and table grain

## `m5-fast`

`m5-fast` is the opt-in numeric variant of this pull request. It has no baseline on `main`, so its images sit beside the `max-quality` candidate of the same revision.

| Case | File | Size | SHA-256 | Against the `max-quality` candidate |
| --- | --- | ---: | --- | --- |
| 1024/teapot-m5-fast-e0b4244.png×1024/teapot-m5-fast-e0b4244.png teapot | [`1024/teapot-m5-fast-e0b4244.png`](1024/teapot-m5-fast-e0b4244.png) | 1.33 MiB | `57f005b360f0428a30e0bee27532e05763e69c04bb6ed8c11c7c66e342520ba6` | correlation 0.987, PSNR 28.3 dB |
| 4096/teapot-m5-fast-e0b4244.png×4096/teapot-m5-fast-e0b4244.png teapot | [`4096/teapot-m5-fast-e0b4244.png`](4096/teapot-m5-fast-e0b4244.png) | 20.12 MiB | `b5b97fb3755632d9eebf45613803b22fce7c64eeae0626dceb35564a0db49886` | correlation 0.996, PSNR 32.8 dB |

The committed record of hashes and pixel differences is `benchmarks/results/campaign-20261123/generated-image-evidence.json` on the pull request branch.
