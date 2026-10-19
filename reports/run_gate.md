# Main-run readiness gate

Passed: 16 real training examples, two optimizer updates, finite loss, backward, nonzero LoRA gradient, unchanged sampled frozen weights, updated adapters, gradient clipping, validation, checkpoint save, and separate adapter reload/logit equality. Unit tests also verify exact resumed next optimizer update on a small real Qwen architecture and token-weighted accumulation equivalence.

Smoke observations: 8.55 seconds including validation/checkpoint, 1,081,344 trainable parameters, sampled MPS allocated memory about 1.15 GB and final MPS driver allocation about 2.47 GB. These are allocated/driver samples, not the whole machine's peak unified-memory usage.

Authorized main scope: rank 8, alpha 16, dropout .05, Q/K/V/O targets, AdamW 1e-4, weight decay .01, cosine schedule with 5% warmup, BF16 frozen base, FP32 adapters, micro-batch 1 × accumulation 8, all 1,412 training examples, 179 validation examples, three epochs / 531 planned updates. Seed 42. 3,600-second soft training cap; check after every update, plus bounded validation/checkpoint overhead. No paid APIs, concurrency one model process, no base-weight downloads. Expected training order of 10–20 minutes from smoke throughput; this is an estimate, not a result.

Stop on nonfinite loss/gradient, checkpoint error, validation patience of two, or time cap. Best adapter selected by validation loss only. Final greedy comparisons use all 181 frozen test rows plus the predeclared OOD/general diagnostics and term-only inference ablation. No test-driven selection. Full-weight fine-tuning and QLoRA hardware comparisons are outside demonstrated local capability.
