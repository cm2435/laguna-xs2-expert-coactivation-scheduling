# `train_dense_reconstruction.py` — change log

Two side-by-side versions of the reconstruction trainer are kept in this directory:

| File | What it is |
|---|---|
| `train_dense_reconstruction_original.py` | **Baseline** snapshot (commit `8a687e2`) — single-dataset trainer, before the data-mixture work. Frozen for reference; not imported. |
| `train_dense_reconstruction.py` | **Current** trainer — adds weighted multi-dataset interleaving. Same entry point/name. |

The delta below is the `8a687e2 → b8e50c6` change ("multi-dataset interleave mixture").

---

## What changed

### 1. New `--datasets` mixture argument
```
--datasets "GPUMODE/KernelBook:0.35,nvidia/OpenCodeInstruct:0.2,…"
```
Comma-separated `name:weight` pairs. When present it **overrides** the single
`--dataset` path; weight defaults to `1.0` if omitted. The original kept only the
single-`--dataset` loader.

### 2. New `mixed_rows()` generator
Probabilistically **interleaves several streaming datasets by weight**:
- Seeded RNG (`random.Random(0)`) → deterministic, reproducible mixture order.
- Weighted-random source pick per row.
- Drains each source independently; drops it from the pool on `StopIteration` and
  keeps going until all are exhausted.

### 3. `main()` wiring + run-config logging
- If `--datasets` is set, build `(name, weight)` specs and stream from `mixed_rows()`;
  otherwise fall back to the original single-dataset `load_dataset` path.
- The saved run config now records the actual mixture string (`dataset_desc`) under
  `dataset`, so a run's metadata reflects the mix rather than one name.

### Companion change (separate file)
`src/densify/reconstruction_data.py` widened the SFT row formatter to recognize
`query` (instruction) and `kernel`/`code` (raw content) keys, so kernel/code datasets
(KernelBook, cuda_kernels, CodeFeedback) format without per-dataset glue.

---

## Why

The reconstruction loss only depends on **what text drives the frozen teacher's forward
pass**, so the data mix is a direct model-quality lever. C4 expert-activation
diagnostics show coverage is **domain-dependent** — densifying on a single domain
overfits the dense surrogate to that domain's active experts. Interleaving
code / Triton / CUDA broadens the activated-expert footprint the surrogate must
reconstruct, anchored on the kernel-optimization downstream target.

See `docs/MODEL_CHANGES.md` for the full model-change rationale.
