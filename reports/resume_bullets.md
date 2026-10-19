# Truthful resume bullets

- Built an explicit PyTorch/Transformers fine-tuning pipeline for a local 494M-parameter Qwen model, training 1.08M LoRA parameters (0.2184%) with assistant-only loss, token-weighted gradient accumulation, validation and resumable checkpoints.
- Audited 1,779 slang/acronym records and created leakage-controlled splits across 1,539 normalized terms; evaluated zero-shot, few-shot and LoRA on 154 unseen terms with paired bootstrap analysis and qualitative review artifacts.
- Ran controlled one-epoch LoRA rank/data ablations, a two-seed repeatability check, technical-community OOD diagnostics and 25 general-capability prompts; documented quantization hardware limits and the distinction between lexical proxies and semantic accuracy.

Do not claim a semantic-accuracy improvement, QLoRA training, CUDA/GPU hardware other than Apple MPS, or superiority to full fine-tuning. Add a quality-improvement claim only after independent contextual correctness review. Measured proxy values are in the final report if a role calls for a precise, explicitly lexical result.
