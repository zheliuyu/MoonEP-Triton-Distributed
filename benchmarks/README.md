# MoonEP-Triton-D Benchmarks

Ported from MoonEP `benchmarks/` for the Triton-distributed backend.

## Quick (2 GPU)

```bash
torchrun --nproc_per_node=2 benchmarks/bench_comm.py
torchrun --nproc_per_node=2 benchmarks/bench_grad_reduce.py
torchrun --nproc_per_node=2 benchmarks/bench_prefetch.py
```

## Full (8 GPU)

Requires `>= 8` visible GPUs; scripts exit cleanly when fewer GPUs are available.

```bash
torchrun --nproc_per_node=8 benchmarks/bench_comm.py --full
torchrun --nproc_per_node=8 benchmarks/bench_grad_reduce.py --full
torchrun --nproc_per_node=8 benchmarks/bench_prefetch.py --full
```

## Pipeline mode (Phase 13)

`MOONEP_TD_PIPELINE=1` uses higher warp counts (8 vs 4) on dispatch/combine/prefetch for improved memory-level parallelism on V100/A100.

```bash
MOONEP_TD_PIPELINE=1 torchrun --nproc_per_node=2 benchmarks/bench_comm.py
```
