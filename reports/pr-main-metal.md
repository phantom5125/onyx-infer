# Onyx validation: `pr-main-metal`

- Protocol: `onyx.ci.v1`
- Suite: `metal`
- Status: **PASS**
- Git revision: `35bc6f9b94ec8599d504900fcd21395a22f54f26` (dirty=False)
- Target: Apple M5 Max, Metal 4
- Duration: 11.701 s

## Kernel

- Operator: `onyx.q4_gemm.v1`
- Variant: `tile16`
- Shape: `{'m': 1024, 'n': 21504, 'k': 5376, 'group_size': 64}`
- Wall median: **111.629583 ms**
- GPU median: **111.403125 ms**
- Numerical check: **True**

## Checks

| Check | Status | Seconds | Exit |
| --- | --- | ---: | ---: |
| metal-ctest | pass | 9.682 | 0 |
| kernel-tile16 | pass | 1.990 | 0 |
