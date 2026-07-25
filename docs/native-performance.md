# Native Performance Report

**Date:** 2026-07-25  
**Platform:** Windows x86_64  
**Python:** CPython 3.14.6  
**Rust:** 1.97.1 (crate MSRV 1.85)  
**PyTorch:** 2.13.0+cu132  
**Native build:** maturin release profile

These results use generated JPEG images and deterministic random feature
fixtures. The benchmark JSON records the workload, platform, commit,
`git_dirty`, and a SHA-256 digest of tracked plus untracked source changes.
They validate the implementation-level performance gates on this machine, but
they do not replace the required Market-1501/MSMT17 rerun because no real
dataset root was available in the workspace.

## Results

| Path | Fixture | Python | Rust | Speedup | Gate |
|---|---|---:|---:|---:|---:|
| Training data | 256 JPEG, 392x196 output, batch 16, 4 workers, 2 Rust threads/worker | 279.9 images/s | 609.7 images/s | 2.18x | >=1.5x pass |
| Primary evaluation | Q=512, G=4096, D=256, chunk 128 | 1.987 s | 0.035 s | 56.9x | >=3x pass |
| Exact rerank | Q=256, G=2048, D=128, chunk 128 | 9.416 s | 0.124 s | 75.7x | >=2x pass |

Primary and reranked mAP/CMC matched their Python references. The rerank cold
process peak RSS delta was 153.6 MB for the dense Python reference and 38.5 MB
for Rust sparse rerank, a ratio of 0.250 against the <=0.4 gate.

The production training CLI intentionally retains the conservative
`--rust-data-threads 1` default. On this machine, an earlier one-thread-per-worker
synthetic data result was 1.27x; setting `--rust-data-threads 2` cleared the
throughput gate without changing DataLoader worker count. Exact timings vary
with filesystem cache and concurrent system load.

## Commands

```bash
uv run python -m t2c_reid.cli.benchmark_native \
  --mode data \
  --data-samples 256 \
  --batch-size 16 \
  --num-workers 4 \
  --rust-data-threads 2 \
  --image-height 392 \
  --image-width 196 \
  --runs 5 \
  --warmup-runs 1

uv run python -m t2c_reid.cli.benchmark_native \
  --mode evaluation \
  --query-count 512 \
  --gallery-count 4096 \
  --feature-dim 256 \
  --chunk-size 128 \
  --runs 5 \
  --warmup-runs 1

uv run python -m t2c_reid.cli.benchmark_native \
  --mode rerank \
  --rerank-query-count 256 \
  --rerank-gallery-count 2048 \
  --feature-dim 128 \
  --chunk-size 128 \
  --runs 3 \
  --warmup-runs 1
```

## Remaining Validation

Run the data benchmark separately on both supported operating systems with a
real dataset root and the same storage/cache conditions:

```bash
uv run python -m t2c_reid.cli.benchmark_native \
  --mode data \
  --dataset market1501 \
  --data-root path/to/Market-1501-v15.09.15 \
  --rust-data-threads 2 \
  --output output/market1501-native-benchmark.json
```

Repeat with `--dataset msmt17`. Record disk type, CPU model, worker/thread
settings, and whether the filesystem cache was warm before comparing results.
