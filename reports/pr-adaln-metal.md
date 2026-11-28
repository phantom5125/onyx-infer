# Onyx validation: `pr-adaln-metal`

- Protocol: `onyx.ci.v1`
- Suite: `metal`
- Status: **PASS**
- Git revision: `8a39605530723534448120654d07923fac9e534f` (dirty=False)
- Target: Apple M5 Max, Metal 4
- Duration: 12.954 s

## Kernel

- Operator: `onyx.q4_gemm.v1`
- Variant: `tile16`
- Shape: `{'m': 1024, 'n': 21504, 'k': 5376, 'group_size': 64}`
- Wall median: **123.880917 ms**
- GPU median: **123.720208 ms**
- Numerical check: **True**

## Checks

| Check | Status | Seconds | Exit |
| --- | --- | ---: | ---: |
| metal-ctest | pass | 10.811 | 0 |
| kernel-tile16 | pass | 2.120 | 0 |
