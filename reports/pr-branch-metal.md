# Onyx validation: `pr-branch-metal`

- Protocol: `onyx.ci.v1`
- Suite: `metal`
- Status: **PASS**
- Git revision: `e0b424489c61b6dad5433bd5d522eaf0b9a3f18f` (dirty=False)
- Target: Apple M5 Max, Metal 4
- Duration: 12.819 s

## Kernel

- Operator: `onyx.q4_gemm.v1`
- Variant: `tile16`
- Shape: `{'m': 1024, 'n': 21504, 'k': 5376, 'group_size': 64}`
- Wall median: **111.7565 ms**
- GPU median: **111.51025 ms**
- Numerical check: **True**

## Checks

| Check | Status | Seconds | Exit |
| --- | --- | ---: | ---: |
| metal-ctest | pass | 10.797 | 0 |
| kernel-tile16 | pass | 1.993 | 0 |
