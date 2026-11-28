# Native AdaLN validation

- Metal gate at 1a201c8d6df4f912d8f17b1d20a3617d82eda62e. Fixed-input 1024 replay, a warm 4K candidate replay, and the isolated gate probe.
- Indexed RMSNorm at 3983d90b31c01f2553fa67762013c9f20a7ae4bc. Official onyx.ci.v1 native replay reports at both resolutions.

The teapot cases use the same H3 W4 pack, seed 42, and six PDD evaluations. Both paths match the baseline latent and PNG hashes in their paired comparison.
