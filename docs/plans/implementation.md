# CommunityQwen Implementation Plan

> **For agentic workers:** Execute inline by default. Track tasks with checkboxes.

**Goal:** Build and execute an educational local Qwen specialization experiment with honest held-out evidence.

**Architecture:** Plain Python modules and explicit PyTorch loops. Immutable raw data → normalized term groups → frozen splits → chat tokenization → baseline and PEFT training → deterministic generation → lexical scoring, human review and statistical analysis. No service dependencies during training.

**Tech Stack:** PyTorch, Transformers, Datasets, PEFT, NumPy, matplotlib, pytest.

**Spec:** ../project_spec.md

## Global Constraints

- Reuse the existing local Qwen weights.
- Do not use paid APIs.
- Do not hide important PyTorch mechanics behind `Trainer`.
- Do not fabricate experiment results.
- Do not leak test terms into training.
- Do not leak gold definitions into prompts.
- Do not force QLoRA if hardware cannot support it.

## Deliverables and checks

- [x] Audit existing checkpoint files, environment, device and repository before edits. Found complete BF16 Qwen2.5 checkpoints; selected 494,032,768 parameter checkpoint for 8 GB M3.
- [x] Data contract: `prepare_data.py` writes immutable raw snapshot, audit, and train/validation/test JSONL with stable IDs. Test normalization, duplicate grouping, disjoint terms and rejected rows.
- [x] Foundations and architecture: tensor/MLP/attention scripts plus actual module inspection. Check shapes, causal masking, autograd and parameter counts.
- [x] Tokenization: `encode_example(row, tokenizer, max_length)` returns input IDs, attention mask, assistant-only labels. Check exact chat prefix, assistant termination, prompt overflow rejection and padding.
- [x] Baselines: evaluate untouched checkpoint zero-shot and fixed training-only demonstrations on validation before training; freeze final test protocol before final comparisons.
- [x] Training: explicit token-normalized accumulated gradients, optimizers, scheduler, clipping, validation, early stopping and resumable adapter-only checkpoints. Test finite loss, frozen base, adapter updates and resumed optimizer/scheduler/RNG equivalence on a tiny real Qwen architecture.
- [x] Smoke: 16 training rows, 2 updates, validation and save/reload on actual local checkpoint; use observed memory/runtime to size main run.
- [x] Main experiment: practical data budget, rank 8, fixed decoding. Controlled term-only context ablation. QLoRA CUDA path with an explicit unsupported-device result here.
- [x] Evaluation: paired unseen-term metrics and bootstrap interval; independent authored OOD and 25 general prompts. Separate lexical proxies from human correctness. Reject incomplete/duplicate joins.
- [x] Documentation and final evidence: plots, results, limitations, concept checklist, study guide, exercises and truthful resume bullets. Run full pytest and inspect saved results.
