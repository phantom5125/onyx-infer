# LoRA adapter design proposal

**Draft.** This proposal requires separate protocol review. [Protocol evolution](PROTOCOLS.md) · [Chinese design notes](zh-CN/LORA_ADAPTERS.md)

## 1. H3 target operators

| Component | Layout and candidate targets |
| --- | --- |
| DiT blocks | Width 5376, fused `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` |
| DiT shared layers | Timestep projections, AdaLN, patch projection, and output heads where eligible |
| Qwen3-VL encoder | Width 5120, split `q_proj/k_proj/v_proj/o_proj` and `gate/up/down_proj` |

The existing string-based linear call provides a natural module path for binding adapters. Fused DiT QKV and split text-encoder projections require a row-slice mapping. Preserve existing high-precision exceptions. The proposed runtime computes the base Q4 GEMM plus the sum of scaled low-rank products in the activation domain.

## 2. Performance hypotheses

For one adapter, the extra arithmetic scales with rank relative to the projection size. DiT QKV at 1024×21504×5376 and text Q projection at 69×8192×5120 are initial benchmark cases. Measure both skinny GEMMs and launch overhead before choosing fusion. A fused epilogue or batched low-rank operations can amortize launches across targets. Whole-pipeline results depend on the selected base GEMM and must be measured separately.

## 3. Import formats and numerical handling

Normalize PEFT/diffusers, kohya, and generic ComfyUI keys to a single internal module path. Support the H3 bare-path mapping and PDD head-0 slicing used by ComfyUI. PEFT metadata and per-module rank and alpha patterns take precedence over inference where available. For kohya keys without alpha, use the convention alpha equals rank.

Use the source `lora_A/lora_B` or `lora_down/lora_up` pairs, optionally with a declared row offset and size. Standard LoRA scales by alpha divided by rank. rsLoRA scales by alpha divided by the square root of rank. Retain FP16 factors and FP32 intermediate accumulation. DoRA uses the norm of the merged weights and is proposed only through an offline high-precision merge followed by repackaging.

## 4. Proposed pipeline

```text
SafeTensors adapter
  → Python importer: detect format, normalize H3 keys, validate, resolve scaling
  → versioned AdapterManifest and independent hashed ONXPACK artifact
  → plan binds module paths and row slices to adapter identities
  → Metal runtime uses the selected Q4 and low-rank operators
```

A manifest would include base model digest, adapter ID, rank, alpha, scaling rule, tensor descriptors, and source format. A plan would carry adapter bindings and variant identifiers. Adapter weights can stay resident because their size is small relative to the base package. The draft proposes preallocation to a maximum rank for hot swapping, one adapter set per job, and declared strength and composition order.

Initial runtime coverage would include plain linear LoRA, linear LoCon, and rsLoRA in SafeTensors. Report unsupported LyCORIS variants, GGUF adapters, base-mutating initialization, bias, or extra saved modules clearly. The proposed offline DoRA path creates a derived package with its own identity.

## 5. Acceptance and open questions

- Compare PEFT and kohya fixtures on real H3 shapes with a CPU FP32 reference.
- Measure Q4 alone and with adapters at matching shapes, then compare a fixed 1024×1024 generation with a reference implementation.
- Include adapter digests and output hash reproduction in replay tests.
- Review per-module suffix matching, base revision compatibility, batched shrink performance, and the initial text-encoder target set before freezing a contract.
