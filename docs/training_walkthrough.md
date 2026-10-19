# Trace one optimizer update

1. `prepare_data.py` loads the pinned Hugging Face dataset revision and preserves raw records. Audit produces stable IDs, quarantine records and normalized term keys. Splitting happens before prompt construction.
2. `data/prompts.py` builds a fixed system instruction and one of three equivalent training queries. Example supplies usage; Description supplies the assistant answer only. Source Context metadata stays out.
3. `data/tokenize.py` uses the local tokenizer chat template, verifies token-aligned prefix and sets prefix labels to -100. It rejects prompt overflow and explicitly truncates long responses. The installed Transformers release returns a dictionary by default; `return_dict=False` makes token-list behavior explicit.
4. `AssistantCollator` pads a mini-batch, marking padded attention positions 0 and labels -100. DataLoader shuffles via an epoch-specific generator.
5. `load_model` reads only local files. `attach_lora` freezes the base and creates adapters after inspecting module names. Gradient checkpointing recomputes activations in backward to reduce memory.
6. In `training/train.py`, `model.train()` enables adapter dropout. Several micro-batches form an accumulation window. `optimizer.zero_grad(set_to_none=True)` clears earlier gradients once per window.
7. `model(**batch)` runs the forward pass. `loss` averages the shifted assistant token negative log-likelihood. Multiplying by `micro_target_tokens/window_target_tokens` gives correct accumulation weighting.
8. `loss.backward()` builds adapter gradients through the frozen base computation. Frozen parameters still participate in forward/backward propagation; they simply do not accumulate trainable gradients.
9. Unscale if CUDA FP16, clip the joint gradient norm, then optimizer step and scheduler step. Log LR used, unclipped norm, loss, examples, target tokens and elapsed time.
10. At an epoch boundary, validation uses no gradients. Improved validation loss saves the best adapter. Every boundary publishes a resumable checkpoint; early stopping observes validation only. The final partial accumulation window is weighted by its actual tokens.
11. Evaluation reloads the best adapter on the unchanged base. Greedy generation receives no target definition. Targets are used only afterward in metrics and a separate teacher-forced perplexity pass.

Resume command: `python scripts/train_lora.py --config configs/lora.yaml --output experiments/lora/run_001 --resume experiments/lora/run_001/checkpoint-177`. Use the actual checkpoint number and original unchanged config. Local training checkpoints contain Python/NumPy RNG state and use trusted pickle loading; load only your own checkpoints. Strict source/config/model/data fingerprints prevent accidentally resuming a different experiment.

The time cap is checked after an optimizer window, so one window plus validation/checkpoint work can overrun it. A completed checkpoint marks the safe cursor. Wall time already consumed is retained on resume. A time-limited run is reported as such rather than silently advertised as a completed epoch budget.

## Execution source versions in this delivery

The main run and first rank controls used execution source v1. During packaging, v2 fixed one optional-path issue: `AUTO_DISCOVER` now writes its resolved checkpoint path back into the config so downstream manifests can identify it. Every measured run uses the same explicit checkpoint path, so this branch is not exercised and numerical training behavior is unchanged. Execution source versions are archived in `experiments/reproducibility/`; `source_versions.json` gives their hashes. Each run's environment identifies its source hash.

Strict resume deliberately rejects a checkpoint created by different source bytes. To resume an old experiment after editing code, use its matching archived source in a separate checkout with the same data/config; do not bypass the fingerprint check or overwrite the current implementation. The main run is already complete. Final evaluation and later supplemental runs record v2. Final packaging v3 extends the same optional AUTO_DISCOVER handling to tokenizer-only preparation and evaluation fingerprinting; the explicit-path numerical execution remains unchanged. Restore v1 or v2 as identified by the run when resuming or reusing generation caches.
