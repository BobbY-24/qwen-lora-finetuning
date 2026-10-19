# Twelve files to study

Use the original request as your concept map, then follow these files in order. Paths below are relative to the repository root.

| File | Input → output | Mechanism to understand | Your modification |
|---|---|---|---|
| `src/community_qwen/fundamentals/mlp.py` | Small float tensors → predictions, loss and gradients | nn.Module, parameter registration, nonlinearities, autograd | Replace GELU with ReLU and compare gradients |
| `src/community_qwen/data/split.py` | Audited rows → disjoint term groups | Experimental leakage control before tensor construction | Add a carefully reviewed alias map and prove groups remain disjoint |
| `src/community_qwen/data/tokenize.py` | Term, sentence, answer → token IDs and masked labels | Causal target alignment; prompt visibility versus supervision | Inspect decoded labels and add a new supported chat format |
| `src/community_qwen/data/collator.py` | Variable-length examples → padded tensors | Mini-batches, attention masks, long integer IDs | Add padding to a multiple of eight; preserve the label mask |
| `src/community_qwen/model/lora.py` | Actual model modules + rank config → frozen base with trainable adapters | Parameter registration, requires_grad and low-rank matrix updates | Compare Q/V versus Q/K/V/O at a fixed budget |
| `src/community_qwen/model/qlora.py` | Device, bit width, dtype → quantized-loading config or clear rejection | Frozen quantized storage versus trainable higher-precision adapters | On CUDA, validate 8-bit loading before NF4 training |
| `src/community_qwen/training/train.py` | Encoded train/validation batches → adapters, checkpoints and telemetry | Forward/backward, token-weighted accumulation, clipping, parameter update | Manually derive the final short accumulation window's weights |
| `src/community_qwen/training/optimizer.py` | Trainable parameters + hyperparameters → optimizer | SGD, Adam and AdamW state/update rules | Switch AdamW to Adam on validation and interpret weight decay |
| `src/community_qwen/training/validate.py` | Model + held-out loader → token-weighted CE and perplexity | eval mode and no_grad; causal shift denominator | Add per-example loss records without retaining graphs |
| `src/community_qwen/training/checkpoint.py` | Model/optimizer/scheduler/RNG state ↔ durable checkpoint | state_dict, atomic persistence, exact continuation | Interrupt after an epoch and compare resumed updates |
| `src/community_qwen/evaluation/generate.py` | Query-only fields → deterministic answer and token counts | Autoregressive generation, KV cache, inference mode | Compare output caps without changing the selected test protocol |
| `src/community_qwen/evaluation/statistical.py` | Paired term scores → mean difference and uncertainty | Sampling assumptions rather than autograd | Write the paired bootstrap manually, then test constant differences |

`docs/training_walkthrough.md` connects these pieces into one complete optimizer update. `tests/test_training.py` is worth reading beside the training files: it checks behavior, including token-weighted accumulation and checkpoint continuation, rather than merely checking function names.

## Personal exercises

These are for you to implement; they are not completed assignments.

1. Build your own two-layer `nn.Module` without copying TinyMLP, and predict every parameter shape before printing it.
2. Replace GELU with ReLU and compare the fraction of zero activations and the gradient norms.
3. Implement stable softmax on a small tensor; compare with `torch.softmax` for large logits.
4. Differentiate a scalar quadratic by hand and check `loss.backward()` against your derivative.
5. Decode an SFT example's masked and unmasked tokens. Explain why the assistant header is visible but ignored as a target.
6. Switch AdamW to Adam on the same training/validation subset and explain coupled versus decoupled decay.
7. Remove weight decay while holding all other factors fixed. Predict how the train/validation gap could change.
8. Disable gradient clipping in a tiny controlled run. Compare unclipped norms and failed/finite updates without scaling prematurely.
9. Change LoRA rank while keeping alpha/r constant. Calculate trainable parameters by hand before running.
10. Implement token-weighted gradient accumulation yourself and verify equality to a combined batch with dropout disabled.
11. Resume from a saved checkpoint and compare the next update with an uninterrupted run; include random-number state in your explanation.
12. Compare zero-shot, two-shot and LoRA answers without seeing model identities. Annotate correct sense, hallucination and generic answers.
13. Add a genuinely different OOD community dataset with provenance, term-overlap checks and independent reviewed definitions.
14. Calculate a paired bootstrap interval manually and explain why resampling rows is inappropriate when terms repeat.
15. Inspect one attention head's causal weights in the educational demo and predict how changing a key changes its weight.
16. Explore a learning-rate change on validation only, then freeze the chosen setting before any fresh test evaluation.
