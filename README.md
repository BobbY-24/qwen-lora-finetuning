# CommunityQwen

An educational research project that specializes an existing local Qwen checkpoint to explain slang in context. It exposes the complete PyTorch path from tensors and chat tokenization to assistant-only loss, LoRA gradients, optimization, validation and held-out evaluation. It does not aim to make the model speak in slang.

The project uses the existing **Qwen2.5-0.5B-Instruct** checkpoint (494,032,768 base parameters) on an Apple M3 with 8 GB unified memory. It trains **1,081,344 adapter parameters** (0.2184% of the model with adapters). Model loading is local-only. The dataset is [MLBtrio/genz-slang-dataset](https://huggingface.co/datasets/MLBtrio/genz-slang-dataset); its exact downloaded revision and snapshot hashes are in `data/processed/manifest.json`.

Start with [the final report](reports/final_report.md), [the study guide](docs/study_guide.md), and [the training walkthrough](docs/training_walkthrough.md). The source request is preserved in [project_spec.md](docs/project_spec.md). Current measured outputs are in `experiments/` and `reports/`; unrun conditions must never be described as results.

## Reproduce locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/inspect_model.py
python scripts/prepare_data.py
python scripts/learn_tensors.py
python -m pytest -q
```

The existing `.venv` is already installed on this machine. `requirements.txt` records its exact packages. For a new platform, `pip install -e '.[test]'` resolves platform-compatible dependencies; record a new lock and do not claim bitwise reproduction. Edit `model.local_path` in configs to another **existing** local Transformers checkpoint if needed, or use `AUTO_DISCOVER` (selects the smallest complete-looking local Qwen safetensors candidate). Do not download a replacement base model. The selected path is recorded in `reports/model_inspection.json`.

The default device is MPS/BF16. Run from a normal host terminal for Apple GPU access; a sandboxed process may report MPS unavailable. For CPU, set `device: cpu` and `dtype: float32` in a copied config. Base loading always passes `local_files_only=True` and disables remote code. After `prepare_data.py` downloads the public dataset once, training/evaluation require no network; `prepare_data.py --offline` reuses the raw snapshot.

## Run order

```bash
python scripts/evaluate.py --output experiments/baseline/validation_new --limit 16
python scripts/evaluate.py --output experiments/few_shot/validation_new --limit 16 --shots 2
python scripts/train_lora.py --config configs/smoke.yaml --output experiments/lora/smoke_new
python scripts/inspect_gradients.py
python scripts/train_lora.py --output experiments/lora/new_run
```

Use new output paths for new experiments. Existing evaluation outputs resume only when their fingerprints match. Training resume requires the original config and checkpoint path:

```bash
python scripts/train_lora.py --output experiments/lora/new_run \
  --resume experiments/lora/new_run/checkpoint-177
```

Use the checkpoint that actually exists. The number above illustrates a full first epoch at this batch configuration. Preserve source/config/data identity. Adapters and full training state are stored locally and ignored by Git; frozen base weights are never copied into run directories.

The predeclared completed-run protocol for `run_001` is `python scripts/final_evaluations.py`. For another adapter, call `evaluate.py` explicitly with `--adapter PATH`, `--input data/processed/test.jsonl`, and `--final-test` only after validation selection. Generate reports with `build_reports.py` after predictions finish. Reports for a new experiment need its paths supplied or the report mapping updated; the default mapping documents this delivered experiment.

## Research design

The audit retains 1,772 rows representing 1,539 normalized terms. Term-level splits contain 1,412/179/181 rows and 1,231/154/154 terms for train/validation/test. Repeated surface variants remain within a split. `Example` is the usage sentence. `Description` is the target. `Context` is excluded because it often explains the target. Descriptions are imperfect weak references, not adjudicated contextual truth. No definitions enter the queried model's prompt.

The training-only length audit supports 128 tokens without truncating any training example (maximum 118). Prompt variants are used only in training; evaluation is fixed. Loss ignores system, user and padding tokens and includes assistant tokens. The loop uses token-weighted accumulation, AdamW, cosine warmup/decay, clipping, validation selection, early stopping and resumable adapter-only checkpoints.

Controls are untouched Qwen zero-shot and fixed two-shot prompts. Evaluation reports lexical ROUGE-L, reference-word coverage, perplexity and behavior diagnostics. Paired bootstrap operates on normalized terms. These proxies do **not** establish semantic accuracy. Review sheets support independent semantic annotation. The OOD examples and 25 general prompts are authored diagnostics, not external benchmarks. See [the frozen evaluation contract](docs/evaluation_protocol.md).

## Experiments and hardware limits

`run_ablations.py --factor rank` (or `data`, `context`, `targets`, `optimizer`) generates controlled config variants; `--execute` runs them serially. A generated config is not a measured experiment. Final context ablations remove the usage sentence with unchanged weights and decoding. Do not feed the dataset's definition-bearing metadata back as 'extra context'.

QLoRA configuration supports optional CUDA bitsandbytes NF4/8-bit loading and a real optional integration test. This MPS machine cannot exercise that path, so QLoRA is guarded and unrun. No QLoRA speed, memory or quality claim is made. Full-weight fine-tuning is also unrun; LoRA's performance relative to it cannot be established here. Accelerate is installed via PEFT, while the primary training loop remains native PyTorch.

The learning sequence is foundations → architecture → data → baselines → causal pipeline → training → LoRA → quantization → reliability → evaluation → optimization → ablations → OOD → statistics → errors → portfolio. Documentation and exercises are organized by these concepts, not arbitrary weeks.

## Supplemental runs and derived artifacts

The delivered one-epoch supplemental experiments are defined in `configs/ablations/one_epoch_*.yaml`. Their serial runners and deterministic analysis are in `analysis/`:

```bash
python analysis/run_supplemental.py
python scripts/final_evaluations.py
python analysis/evaluate_supplemental.py
python scripts/build_reports.py
python analysis/plot_supplemental.py
python analysis/verify_artifacts.py
python analysis/write_final_report.py
```

Training runners require new run directories if results already exist; individual training commands allow checkpoint resume. Analysis scripts can regenerate reports from saved predictions without another model run. Run `plot_supplemental.py` after `build_reports.py` so rank/data figures include only the matched one-epoch conditions. The seed-123 repeat shares the original frozen split. The two-seed result describes one-epoch repeatability, not uncertainty in the selected three-epoch result.

Execution source archives are in `experiments/reproducibility/`. Optional `AUTO_DISCOVER` identity fixes distinguish packaged v3 from the v1/v2 execution snapshots; all measured runs use an explicit path, so their numerical path is unchanged. For strict checkpoint continuation after source changes, restore the matching archived source in a separate checkout. See the training walkthrough for the version boundary.
