from pathlib import Path
import json,csv

def load(p):return json.loads(Path(p).read_text())
s=load('experiments/lora/run_001/summary.json');c=load('reports/comparison.json');a=load('data/processed/manifest.json')
env=load('reports/environment.json');ood=load('reports/ood_evaluation.json');general=load('reports/general_regression.json')
supp=load('reports/supplemental_experiments.json');grad=load('reports/gradient_inspection.json')
rows=list(csv.DictReader(Path('experiments/lora/run_001/metrics.csv').open()))
val=[r for r in rows if r['validation_loss']]
text=f'''# CommunityQwen: measured project report

Held-out ROUGE-L improved from 0.0352 to 0.2967 and assistant-token perplexity fell from 161.97 to 7.74. However, the adapter still makes clear meaning errors, and much shorter answers contribute to the overlap gain. It is not a reliably accurate contextual slang interpreter.

The existing local Qwen2.5-0.5B-Instruct checkpoint was reused for an explicit PyTorch LoRA experiment on contextual slang/acronym explanation. The project completed data audit, zero/few-shot controls, real-model smoke/reload checks, three-epoch LoRA training, frozen held-out evaluation, context/rank/data ablations, a two-seed one-epoch repeatability check, OOD diagnostics and a general-capability sanity suite. QLoRA is guarded and unrun on MPS. Human semantic review remains outstanding.

## Environment and checkpoint

- Checkpoint: `{load('reports/model_inspection.json')['checkpoint']['path']}`
- Base architecture: Qwen2ForCausalLM (Qwen2.5), 494,032,768 unique parameters.
- Device: Apple M3 MPS, 8 GB unified memory. BF16 base, FP32 PEFT adapters (verified in the saved safetensors header).
- Python: {env['python'].splitlines()[0]}
- Versions: {', '.join(k+' '+str(env['packages'][k]) for k in ['torch','transformers','peft','datasets','accelerate'])}.
- Frozen weight/config/tokenizer SHA-256 hashes: `checkpoint_sha256.json`; the local weight hash matches the [published checkpoint hash](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/model.safetensors). No base download; dataset/dependency downloads only. Training and inference use local files without paid APIs.

## Dataset and task validity

{a['raw_rows']} raw rows; {a['retained_rows']} retained rows; {a['retained_terms']} normalized terms. Train/validation/test have 1,412/179/181 rows and 1,231/154/154 terms. Seven records were quarantined: three duplicate examples, two empty normalized terms and two too-short definitions. Missing cells: {a['missing_values']}.

The Example field is usage; Description is the weak target. Context metadata is excluded because it often reveals the answer. Normalized term groups and exact usage-sentence checks prevent measured cross-split overlap. The original descriptions are not independently adjudicated and can be ambiguous or inaccurate. Uppercase terms occupy 1,260/1,412 training rows, so results primarily concern an acronym-heavy corpus rather than broad Gen-Z language competence.

Training token-length percentiles: {a['training_token_lengths']}. Sequence budget 128, no training truncation. Tokenizer base vocabulary 151,643, including added entries 151,665; embedding/head rows 151,936. EOS `<|im_end|>`, PAD `<|endoftext|>`. Tokenization examples and audit details are in `dataset_audit.md` and `data_quality_notes.md`.

## Architecture learned

Embedding width 896; 24 decoder layers; 14 query heads / 2 KV heads, dimension 64 (GQA). Query/key/value projections feed scaled causal attention, with an output projection and residual connection. RMSNorm precedes sublayers; gated SiLU MLP projections expand to 4,864 and return to 896. RoPE supplies positional information; context configuration is 32,768. The LM head shares the token embedding weights. See `../docs/qwen_architecture.md` and the actual module inspection JSON.

## Training and reliability

Q/K/V/O LoRA, rank 8, alpha 16, dropout .05. {s['trainable_parameters']:,} adapter parameters, {s['trainable_percent']:.4f}% of base plus adapters. All {s['frozen_parameters']:,} base parameters remain frozen. AdamW LR 1e-4, weight decay .01, cosine schedule, 5% warmup, micro-batch 1 × accumulation 8, effective nominal batch 8. Gradient norm clipped to 1.0 after correctly weighting each micro-batch by supervised assistant tokens.

Completed status: **{s['status']}**, {s['global_step']} updates, {s['epoch']} epochs, {s['examples_processed']} example exposures and {s['tokens_processed']} assistant target tokens. Runtime {s['runtime_seconds']:.1f} seconds ({s['runtime_seconds']/60:.1f} minutes). Initial full-validation CE {s['initial_validation']['loss']:.4f}; best validation CE {s['best_validation_loss']:.4f}. First optimizer-window loss {float(rows[0]['training_loss']):.4f}; last window loss {s['final_training_loss']:.4f}. Those windows contain different examples, so they are not a paired metric.

Observed maximum unclipped gradient norm {s['maximum_unclipped_gradient_norm']:.3f}. Sampled peak MPS tensor allocation {s['sampled_peak_mps_allocated_bytes']/1e9:.3f} GB; final MPS driver allocation {s['mps_driver_allocated_bytes']/1e9:.3f} GB. These are process/device allocator measurements, not peak total system unified-memory usage; CPU RSS is separately logged. The real-model adapter reload matched logits. A tiny real Qwen test verifies checkpoint continuation, partial accumulation and early stopping. Best adapter selection uses validation only.

| Epoch | Validation CE |
|---|---:|
'''
for r in val:text+=f'| {float(r["epoch"]):.0f} | {float(r["validation_loss"]):.4f} |\n'
text+='\n![Training loss, validation loss and learning rate](../experiments/lora/run_001/learning_curve.png)\n'
text+='''
## Final unseen-term comparison

The adapter is not a reliably accurate slang interpreter: clear semantic errors remain despite improved reference metrics. See the qualitative case review and length-stratified analysis before interpreting overlap gains.

All conditions use the same 181 frozen rows / 154 normalized terms and greedy decoding. Scores below are row-mean ROUGE-L lexical overlap, not semantic accuracy. Memory is the final MPS driver allocator sample during evaluation, not peak training memory.

| Model | Trainable parameters during adaptation | Eval driver memory GB | Test ROUGE-L | Test PPL | OOD ROUGE-L |
|---|---:|---:|---:|---:|---:|
'''
for name,r in c['conditions'].items():
 params=s['trainable_parameters'] if name=='LoRA' else 0
 o=ood.get('LoRA' if name=='LoRA' else 'Base') if name!='Base few-shot' else None
 oscore=f'{o["metrics"]["rouge_l"]:.4f}' if o else 'not run'
 text+=f'| {name} | {params:,} | {r.get("mps_driver_allocated_bytes",0)/1e9:.3f} | {r["metrics"]["rouge_l"]:.4f} | {r["assistant_loss"]["perplexity"]:.2f} | {oscore} |\n'
text+='\nQLoRA: not run; no measured quantization or efficiency comparison. Full-weight fine-tuning: not run, so “most of its gains” cannot be claimed for LoRA.\n'
text+='\n## Paired statistics and uncertainty\n\n```json\n'+json.dumps(c['paired_term_statistics'],indent=2)+'\n```\n'
text+='\nTerms, not repeated rows, are the resampling units. Intervals condition on one selected main adapter; they do not estimate training-seed uncertainty. Permutation p-values are exploratory and unadjusted for multiple comparisons. Reference bias and output-length changes can affect overlap. No confirmatory significance or semantic-accuracy claim is made.\n'
text+='\n## Ablations and repeatability\n\nSee [actual ablations](ablations.md) for the prompt/context diagnostic (whose wording also changes) and the predeclared one-epoch rank/data controls. Alpha is held fixed in the rank experiment; the data-size comparison keeps epochs fixed and therefore varies total updates.\n\n![Data scaling](data_scaling.png)\n\n![Rank scaling](rank_scaling.png)\n\n```json\n'+json.dumps(supp['seed_statistics'],indent=2)+'\n```\n'
text+='\n## OOD and general-capability checks\n\nThe technical OOD set contains '+str(ood['Base']['rows'])+' assistant-authored, source-linked examples with no normalized-term overlap against any split. It is a small diagnostic, not an external human-validated benchmark.\n\nGeneral checks contain 25 prompts: 20 objective short-answer cases and five summaries awaiting review. Strict case/text/format checks may mark verbose or case-varied answers wrong; this is a format-sensitive sanity check.\n\n```json\n'+json.dumps(general,indent=2)+'\n```\n'
text+='''
See [general-capability interpretation](general_findings.md) for the 7/20 → 15/20 strict result, its exact-copy regression, and qualitative summarization losses.

## Error analysis and remaining evidence

[Descriptive strata](stratified_results.md) compare uppercase terms and reference lengths, alongside output brevity, to expose possible format effects.

[Selected qualitative cases](qualitative_cases.md) give explicitly labeled assistant judgments, including real improvements, both-fail examples and lexical-metric counterexamples. These are not independent human labels.

[Error analysis](error_analysis.md) lists the largest lexical increases/decreases with source IDs, contexts, weak references and both model outputs. These are review candidates; they must not be silently converted into “base wrong / tuned correct” labels. Paired human correctness, wrong sense, context ignored, hallucination, generic answers, excessive slang style and possible memorization fields are prepared in `paired_human_review.jsonl`. Independent semantic review has not been performed.

The next decisive experiment is blinded human annotation of these frozen predictions, including reference disputes and low-overlap paraphrases. Do not change prompts or tune on the inspected test errors. Any test-informed redesign needs fresh confirmation data. Additional optimization/regularization sweeps and independently sourced extra context remain available as future exercises, not completed results.

## Verification and learning deliverables

See `pytest.txt` for the current actual test count, and `artifact_verification.json` for exact evaluation coverage, recomputed scores and unchanged split hashes. All generated examples are saved. Scientific plots derive only from observed runs. The measured numerical runs used archived execution source v1/v2; packaged v3 adds optional AUTO_DISCOVER identity handling only, an unexercised branch for every measured explicit-path run. Source archives and config/environment manifests preserve run provenance; checkpoint fingerprints reject mismatched code/config/data.

[93-concept checklist](../docs/concept_checklist.md), [12-file study guide and 16 exercises](../docs/study_guide.md), [PyTorch foundations](../docs/pytorch_foundations.md), [Transformer concepts](../docs/transformer_concepts.md), [LoRA](../docs/lora.md), [QLoRA limits](../docs/qlora.md), and [research-question boundaries](../docs/research_questions.md).
'''
Path('reports/final_report.md').write_text(text)
bullet=f'''# Truthful resume bullets

- Built an explicit PyTorch/Transformers fine-tuning pipeline for a local 494M-parameter Qwen model, training 1.08M LoRA parameters (0.2184%) with assistant-only loss, token-weighted gradient accumulation, validation and resumable checkpoints.
- Audited 1,779 slang/acronym records and created leakage-controlled splits across 1,539 normalized terms; evaluated zero-shot, few-shot and LoRA on 154 unseen terms with paired bootstrap analysis and qualitative review artifacts.
- Ran controlled one-epoch LoRA rank/data ablations, a two-seed repeatability check, technical-community OOD diagnostics and 25 general-capability prompts; documented quantization hardware limits and the distinction between lexical proxies and semantic accuracy.

Do not claim a semantic-accuracy improvement, QLoRA training, CUDA/GPU hardware other than Apple MPS, or superiority to full fine-tuning. Add a quality-improvement claim only after independent contextual correctness review. Measured proxy values are in the final report if a role calls for a precise, explicitly lexical result.
'''
Path('reports/resume_bullets.md').write_text(bullet)
print('Wrote final report and evidence-grounded resume bullets.')
base_lines=['# Untouched Qwen baselines','','Untouched local weights are the control. Validation baselines were executed before any fine-tuning. Final test baselines use the complete frozen test population after the protocol was fixed. Few-shot demonstrations come only from training terms.','',
            '| Population | Condition | Rows | ROUGE-L | Assistant PPL |','|---|---|---:|---:|---:|']
for pop,condition,path in [('validation smoke','zero-shot','experiments/baseline/validation_zero'),('validation smoke','two-shot','experiments/few_shot/validation_two'),('final test','zero-shot','experiments/baseline/test'),('final test','two-shot','experiments/few_shot/test')]:
    r=load(Path(path)/'summary.json');base_lines.append(f'| {pop} | {condition} | {r["rows"]} | {r["metrics"]["rouge_l"]:.4f} | {r["assistant_loss"]["perplexity"]:.2f} |')
base_lines+=['','Do not compare validation and test means as though they were the same population. Lexical scores and reference likelihood are proxies, not independent semantic correctness judgments.']
Path('reports/baseline.md').write_text('\n'.join(base_lines)+'\n')
