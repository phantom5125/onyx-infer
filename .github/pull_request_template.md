## Summary

- Change and motivation
- Related issue or discussion
- Effects on public contracts, model semantics, numerical output, or dependencies

## Validation

List the commands run or attach the `onyx.ci.v1` reports.

- [ ] Contract checks
- [ ] Relevant Metal or kernel checks
- [ ] Relevant native 1024×1024 or 4096×4096 checks
- [ ] Performance, peak memory, and output hashes recorded where applicable

## Generated-image evidence

Required for every change that could affect numerical accuracy or visual output. Write `N/A — documentation only` or explain why this change cannot affect output.

- [ ] The original candidate PNG is uploaded below and renders directly in this PR.
- [ ] The baseline PNG is uploaded for comparison, or the absence of a baseline is explained.
- [ ] Baseline and candidate use the same request, model, seed, resolution, sampling steps, and execution profile.
- [ ] Base/candidate revisions, hardware, both output SHA-256 hashes, and any expected visual differences are documented.

**Request / seed / resolution / steps / profile:**

**Hardware / base revision / candidate revision:**

**Baseline image:**

<!-- Drag the original baseline PNG here. -->

**Candidate image:**

<!-- Drag the original candidate PNG here. Do not replace this with a local path or hash. -->

**Output hashes / visual review notes:**

## Scope

- [ ] The diff contains the intended change and its supporting evidence.
- [ ] Model weights, local artifacts, logs, and credentials are excluded.
- [ ] Larger design changes have been discussed with the maintainers.

## Hardware follow-up

List the kernel shapes, prompts, resolutions, or memory conditions a maintainer should verify.
