# Onyx protocols and interfaces

The v1 Protobuf contracts are stable. [Compatibility baseline](../schema/v1.compatibility.json) · [Chinese design notes](zh-CN/PROTOCOLS.md)

## 1. Compatibility

Identity and hashes, tensor axes and numerical semantics, ownership, job terminal states, asynchronous completion, RNG, and evidence remain stable within a major version. The public protocol allows checked additive changes such as optional fields whose defaults preserve behavior. Plans, plan caches, kernel variants, memory algorithms, and the in-tree backend SPI can evolve independently. A binary plugin ABI would have a separate review and version.

## 2. Protocol families and transport

| Protocol | Records |
| --- | --- |
| `onyx.common.v1` | Shared identifiers, tensor descriptors, artifact references, and errors |
| `onyx.job.v1` | Requests, execution hints, job events, cancellation, and outputs |
| `onyx.model.v1` | Manifests, semantic graphs, components, recipes, and quantization profiles |
| `onyx.plan.v1` | Resolved shapes, physical layouts, residency, and kernel choices |
| `onyx.trace.v1` | Ordered events, resolved identities, measurements, and replay inputs |
| `onyx.loop.v1` | Optimization cases, candidates, experiments, evaluations, and registry decisions |

Protobuf is the canonical schema. JSON provides its standard mapping for tools and local services. In-process calls, Unix sockets, or later transports carry the same message semantics. Large tensor and image bytes travel through artifact or buffer handles. Standard units include bytes, monotonic nanoseconds, pixels, UUIDs, and SHA-256.

## 3. Sessions and jobs

```text
Engine.create_session(SessionConfig) -> Session
Session.prepare(ModelRef, PrepareOptions) -> PreparedModel
Session.submit(GenerateRequest) -> Job
Job.events() -> ordered stream<RunEvent>
Job.cancel(CancelReason) -> accepted | already_terminal
Session.release(handle) -> Status
Session.close(drain | cancel_jobs) -> Status
```

These signatures describe behavior across languages. `prepare` resolves model artifacts, capabilities, and a plan. Residency is plan-defined. `submit` is asynchronous and records the resolved model, backend, plan, and numeric profile. Handles belong to their session. Physical release can wait for outstanding references and device work. Each job emits exactly one terminal event.

### 3.1 ModelRef

`model_id` names the model. `source_revision` pins its immutable upstream revision. `manifest_digest` binds the package for replay. `recipe_id` selects the model's sampling and preprocessing contract. The RunRecord stores resolved values after any human-friendly aliases have been expanded.

### 3.2 GenerateRequest

A request includes the ID, ModelRef, final UTF-8 prompt, output size in pixels, optional seed, ordered reference artifacts, image count, output format, execution hints, and optional client metadata. Metadata is outside the default model semantic digest.

- Submit the same request ID and digest to retrieve its existing job. Reusing the ID with a different digest returns `IDEMPOTENCY_CONFLICT`.
- Resolve a missing seed once and record it. JSON represents `uint64` as a decimal string. A multi-image recipe defines the per-sample seed and noise order and records those values.
- The model recipe validates dimensions, reference ordering, sample counts, and sampling options. The initial H3 text-to-image recipe accepts no reference images, one output, six PDD steps, and no CFG. Invalid overrides return `UNSUPPORTED_PARAMETER`.
- Execution hints cover backend, compilation, cache, and deadline preferences. Any numerical approximation has a declared profile recorded in the trace.
- Request readers reject unknown parameters. If a caller expands a prompt, the caller submits the expanded prompt and can record both forms and the expansion provenance in its replay case.

## 4. Artifacts and identity

An `ArtifactRef` identifies content by ID, media type, bytes, and SHA-256. A descriptor supplies kind, schema, and an optional opaque local locator. File hashes cover raw bytes. Locations are excluded from artifact identity. Artifacts are immutable and may be collected when no manifest, session, or trace references them. Derived weights retain source and tool provenance. A model package supplies data and declarations.

The logical model `semantic_digest` covers IR, recipe, and weight content identity. It excludes device plans, paths, kernels, and physical layout. `onyx-canon-v1` uses the RFC 8785 canonical JSON mapping with schema-defined numeric representations.

## 5. Events and errors

A RunEvent includes run and request IDs, sequence, monotonic time, optional UTC audit time and stage, and one payload. Payloads include accepted, stage started, progress, metric, artifact ready, completed, failed, and cancelled.

The sequence starts at zero and increases strictly for deduplication and resume. Progress is scoped to the declared stage or step. Completion references the final RunRecord and output artifacts. Cancellation is cooperative at declared boundaries.

Errors have a stable code, retryability, optional stage, diagnostic text, and typed details. Branch on the code. Initial codes include `INVALID_REQUEST`, `UNSUPPORTED_PARAMETER`, `MODEL_NOT_FOUND`, `ARTIFACT_INTEGRITY`, `CAPABILITY_MISSING`, `PLAN_MISMATCH`, `RESOURCE_EXHAUSTED`, `NUMERIC_FAILURE`, `CANCELLED`, `DEADLINE_EXCEEDED`, `BACKEND_FAILURE`, `INTERNAL`, and `IDEMPOTENCY_CONFLICT`.

## 6. Model semantics and numerical identity

A manifest records provenance, components, stages, recipes, request capabilities, numeric and quantization identities, and the semantic digest. The semantic graph uses versioned operators with typed values, ordered inputs and outputs, and declared symbolic constraints. A recipe describes stage scheduling, iterative regions, auxiliary artifacts, and RNG.

- Reject unknown operations, dtypes, and required attributes. Resolve shapes at prepare time or compile for a declared dynamic range.
- Define accumulation, rounding, epsilon, and boundary behavior per operator. Declare RNG algorithm, version, seed, layout, and transformations. A new GPU RNG has its own identity.
- The PDD schedule and output heads belong to the H3 recipe. H3 import handles QKV storage `[head][q/k/v][128]` explicitly. With `flip_sin_to_cos=True`, timestep features use `[cos,sin]`. Separate numerical tests cover those layouts.
- A quantized package declares its source semantic digest and receives a new semantic digest. The quantization profile records algorithm and tool identity, calibration and evaluation data, per-tensor bit width and group size, scale and zero-point layouts, rounding and clipping, precision exceptions, packed sizes and hashes, and quality evidence.

## 7. Tensors, memory, and synchronization

The logical tensor records dtype, axes, and shape. The physical descriptor adds storage dtype, quantization, memory domain, byte size, strides, alignment, and layout. Kernels receive explicit axis and packing contracts.

Device capability queries supply physical memory, recommended working set, maximum buffer, unified-memory support, and tensor capabilities. The planner uses those measured inputs. Buffers are owned opaque handles. Mapped host pointers require synchronization before access, and arena liveness protects any storage still referenced by GPU work. A zero-copy plan may elide a physical transfer while preserving its dependency. Resource exhaustion is reported at prepare time or a stage boundary.

Backend preparation resolves programs and resource needs. Encoding submits work with dependency events and returns a completion event. Waiting or consuming the backend event stream yields completion, failure, or diagnostic counters. Backend events and public Job events have separate lifetimes.

## 8. Operators and backend interfaces

`OpDefinition` combines the versioned operator, logical types, attributes, and numerical contract. `KernelVariant` additionally declares its physical signature, device and shape support, implementation and build identity, tiling, memory, alignment, asynchronous lifetime, rounding, determinism, and reference evidence.

A future binary ABI would use fixed-width types, `struct_size`, explicit `abi_version`, opaque handles, capability negotiation, and queried function tables. Exceptions, allocators, STL, RTTI, and C++ object layouts stay within one library. Stable slots keep their order and semantics.

## 9. Physical plans

A plan records the semantic digest, resolved shape and recipe, device and toolchain identities, kernels, numerical profile, physical layout, fusion and tiling, residency, temporary memory, arena reuse, and transfer dependencies. An origin map links optimized operations to semantic nodes. The complete cache key covers every execution input. An identity mismatch yields `PLAN_MISMATCH` and calls for rebuilding the plan.

## 10. Replay records

A RunRecord stores the request digest and resolved model, the actual plan and variants, weights and numeric settings, RNG and initial noise identity, device and builds, ordered events, artifacts, stage metrics, cache state, and peak memory. Available platform telemetry can add energy and measurement provenance. High-frequency kernel traces can use separate shards linked by run, stage, and operator identity.

An audit-only record can retain a prompt hash and artifact references. Full replay requires the original inputs, exported explicitly through a case when they contain private data.

## 11. Schema evolution

Add fields, capabilities, events, artifact kinds, metrics, and implementations with backward-compatible defaults. Existing field numbers, units, defaults, enumeration semantics, numerical behavior, and RNG keep their meaning. Reserve removed field numbers and enum slots. Metadata and event readers may preserve or ignore optional extension fields where safe. Negotiate capabilities explicitly. A semantic break requires a new major version with explicit migration or rejection. Protocol evolution goes through design review separately from performance experiments.
