# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is verl?

verl (Volcano Engine Reinforcement Learning) is a production-ready RL training library for LLMs by ByteDance Seed team. It implements the HybridFlow programming model where a single controller (the trainer) orchestrates distributed workers via Ray. It supports PPO, GRPO, REINFORCE++, RLOO, DAPO, and other RL algorithms, with backends including FSDP, FSDP2, and Megatron-LM for training and vLLM/SGLang/TensorRT-LLM for inference.

## Build & Development Commands

```bash
# Install for development (pick your inference backend)
pip install -e ".[test,vllm]"
# or
pip install -e ".[test,sglang]"

# Linting/formatting (pre-commit required for all contributions)
pip install pre-commit && pre-commit install
pre-commit run                    # staged changes only
pre-commit run --all-files        # all files
pre-commit run --all-files ruff   # ruff only

# Run CPU unit tests (file names must end with _on_cpu.py)
pytest -s -x --asyncio-mode=auto tests/ -k "_on_cpu"

# Run a single test file
pytest -s tests/test_protocol_on_cpu.py

# Run GPU unit tests (everything except _on_cpu.py files and special_* dirs)
pytest -s tests/ --ignore=tests/special_distributed --ignore=tests/special_e2e --ignore=tests/special_npu --ignore=tests/special_standalone --ignore=tests/special_sanity

# Build docs
cd docs && pip install -r requirements-docs.txt && make clean && make html
```

## Code Style

- **Ruff** for linting and formatting. Line length: 120. Rules: E, F, UP, B, I, G.
- **MyPy** for type checking (most modules have `ignore_errors = true`; strict checking on `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`).
- All source files require Apache 2.0 license headers (enforced by pre-commit).
- Pre-commit also runs: auto-generate trainer config YAMLs, docstring coverage check, `compileall`.

## Architecture Overview

### Single Controller + Distributed Workers (HybridFlow)

The core architectural pattern is **single controller**: one process (the `RayPPOTrainer`) orchestrates multiple distributed worker groups via Ray. Workers are decorated with `@register(dispatch_mode, execute_mode)` to define how data flows to/from them.

```
RayPPOTrainer (single controller)
  ├── ActorRolloutRefWorker  ─── handles policy model, rollout generation, reference policy
  ├── CriticWorker           ─── handles value function training
  └── RewardManager          ─── computes rewards (rule-based or model-based)
```

Key dispatch modes (`verl/single_controller/base/decorator.py`):
- `DP_COMPUTE_PROTO`: splits DataProto across data-parallel ranks, gathers results
- `RANK_ZERO`: runs only on rank 0
- `ONE_TO_ALL`: broadcasts from controller to all ranks

### DataProto (`verl/protocol.py`)

The universal data transfer object. Wraps a `TensorDict` of batched tensors plus a `non_tensor_batch` dict for non-tensor data (strings, metadata). All inter-module communication uses `DataProto`. Key methods: `chunk()`, `concat()`, `select()`, `union()`, `to()`.

### Configuration System

Dual config approach:
1. **Hydra YAML** (`verl/trainer/config/*.yaml`): hierarchical configs with defaults composition. Entry point: `@hydra.main(config_path="config", config_name="ppo_trainer")`.
2. **Typed dataclasses** (`verl/trainer/config/`): `BaseConfig` subclasses that act like both dataclasses and dicts (inherit `collections.abc.Mapping`). Fields are frozen by default (immutable after init).

Config YAMLs are auto-generated from dataclass definitions via `scripts/generate_trainer_config.sh`. Files prefixed with `_generated_` must not be manually edited.

### Worker Implementations

Two main backend families, selected by config:

| Backend | Workers file | Engine dir | Use case |
|---------|-------------|------------|----------|
| FSDP/FSDP2 | `verl/workers/fsdp_workers.py` | `verl/workers/engine/fsdp/` | Standard distributed training |
| Megatron-LM | `verl/workers/megatron_workers.py` | `verl/workers/engine/megatron/` | Large-scale tensor/pipeline parallelism |

### Rollout Engines

Inference engines live under `verl/workers/rollout/`:
- `vllm_rollout/` - vLLM integration (primary)
- `sglang_rollout/` - SGLang (multi-turn, tool-calling)
- `trtllm_rollout/` - TensorRT-LLM
- `hf_rollout.py` - HuggingFace Transformers (fallback)

### Training Loop (PPO example)

`verl/trainer/ppo/ray_trainer.py` contains the main loop:
1. Sample data from dataset
2. Generate rollouts (actor generates responses)
3. Compute rewards (rule-based or model-based)
4. Compute advantages and returns (`core_algos.py`)
5. Update actor policy (PPO/GRPO loss)
6. Update critic (if using PPO)
7. Log metrics, checkpoint

### Algorithm Core (`verl/trainer/ppo/core_algos.py`)

Contains implementations of all RL loss functions: PPO clip loss, GRPO, RLOO, ReMax, REINFORCE++, DAPO, and advantage estimation (GAE, GRPO-style).

## Test Layout

- `tests/<module>/` — mirrors `verl/<module>/` namespace
- `tests/special_distributed/` — multi-GPU unit tests
- `tests/special_e2e/` — end-to-end training tests
- `tests/special_sanity/` — quick sanity checks (docstrings, licenses)
- Files ending in `_on_cpu.py` run on CPU; all others require GPU

## Key Patterns

- **Colocated workers**: `create_colocated_worker_cls()` fuses multiple worker types onto the same GPU group (e.g., actor + rollout + reference share GPUs).
- **Resource pools**: `ResourcePoolManager` maps named pools to GPU sets. Workers are placed onto pools. Different pools can overlap (colocated) or be disjoint (split placement).
- **Sequence length balancing**: `verl/utils/seqlen_balancing.py` partitions batches so GPU memory usage is balanced across DP ranks despite variable-length sequences.
- **External libraries**: Users can register custom models/reward functions via `external_lib` config field and `load_extern_object()`.

## Running Training

PPO/GRPO training is launched via Hydra:
```bash
python -m verl.trainer.main_ppo \
    data.train_files=... \
    actor_rollout_ref.model.path=... \
    actor_rollout_ref.rollout.name=vllm \
    +algorithm.adv_estimator=grpo
```

See `examples/` for full training scripts with model-specific configs.
