# OnyxAdaLN gated-residual evidence

Issue [#9](https://github.com/phantom5125/onyx-infer/issues/9). Product commit `8a39605530723534448120654d07923fac9e534f` on base `a82f0cf907648e9dabf1457cf35efbbfccd28f4d`.

The original baseline and Metal PNGs at each resolution have identical content. They are the originals from the paired runs, published using the matching Git blob IDs already in the repository. The 4K JPEG is a preview of the shared PNG.

- 1024×1024 PNG SHA-256 `6fa24b90ed429f699f8ff39b6f46361bb0a6ddba0a68bdc72ccde0f37efbf3ac`
- 4096×4096 PNG SHA-256 `c68d8fbe635ade8748f2be69477bc5d26f5493a5322d9052428e07d3e59d0c43`

Both cases use the H3 W4 pack, seed 42, six PDD steps, `max-quality`, and VAE batch 25 on M5 Max 36 GiB.

The [full campaign report](reports/campaign-full.json) contains raw paired and diagnostic measurements. The individual `onyx.ci.v1` reports validate a separate clean build of the product commit. Their generation checks are correctness replays.

`replay.py` is the local paired-run harness. Run it from the product checkout with the model pack staged in `artifacts/models/h3-w4` and the two arm names pointing to a Release build under `.build/`.
