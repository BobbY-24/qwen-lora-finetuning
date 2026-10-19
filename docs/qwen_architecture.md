# Actual local Qwen architecture

Selected model: Qwen2.5-0.5B-Instruct, implemented by `Qwen2ForCausalLM`. Complete local checkpoint:

`models/Qwen2.5-0.5B-Instruct`

The 1.5B and 3B siblings were also found with complete safetensors weights. The 0.5B model was selected for an Apple M3 with 8 GB unified memory. No base model was downloaded. Stored and unique loaded parameter count: **494,032,768**. Stored dtype BF16; execution uses BF16 base weights on MPS and PEFT's higher-precision trainable adapters. The sandbox cannot expose MPS, while host execution can; `model.device` therefore explicitly requests MPS and fails clearly when inaccessible. CPU FP32 is an explicit portable fallback.

| Concept | Inspected implementation |
|---|---|
| Embeddings | `model.embed_tokens`, 151,936 × 896 |
| Decoder stack | `model.layers`, 24 Qwen2 decoder layers |
| Attention | `self_attn`, PyTorch SDPA implementation |
| Q/K/V | `q_proj`, `k_proj`, `v_proj` |
| Query / KV heads | 14 / 2, head dimension 64, GQA |
| Attention output | `o_proj`, back to 896 dimensions |
| Normalization | layer input/post-attention RMSNorm; final `model.norm` |
| Feed-forward | `gate_proj`, `up_proj`, `down_proj`, width 4,864, SiLU gate |
| Position | RoPE, theta 1,000,000 |
| Context | configured 32,768 positions; sliding window disabled |
| Output | `lm_head`, tied embedding weights |
| Attention dropout | 0.0 in checkpoint config |

The exact serialized config, module class names, tokenizer and environment are in `reports/model_inspection.json` and `reports/environment.json`. Inspect `scripts/inspect_model.py` rather than relying on a model-name guess.

Architecture references: [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), local `transformers/models/qwen2/modeling_qwen2.py` and [Transformers Qwen2 documentation](https://huggingface.co/docs/transformers/model_doc/qwen2). The local installed implementation is the execution authority.
