# Transfer learning, SFT and LoRA

Transfer learning starts from pretrained weights and adapts them to a new task. Supervised fine-tuning (SFT) learns from input → desired output pairs. Instruction tuning is SFT expressed as user tasks and assistant answers. CommunityQwen does all three: the desired behavior is contextual explanation, not imitation of slang style.

Parameter-efficient fine-tuning (PEFT) trains a small subset of additional parameters. Low-rank adaptation (LoRA) modifies a linear transformation by `W' = W + (alpha/r) BA`, with A of shape `[r,in]` and B of shape `[out,r]` under PyTorch's convention. W stays frozen. Each adapted projection adds `r*(in+out)` parameters rather than updating every element of W. Rank controls representational capacity; alpha scales the update and dropout regularizes its input.

`model/lora.py` inspects actual Linear module names and chooses Q/K/V/O projections. It rejects absent requested targets. Rank 8 and alpha 16 mean a scale of 2. PEFT usually initializes B to zero, so the initial adapter output is zero. On the first backward pass, B can receive a gradient while A's gradient is zero; a zero A gradient at initialization is not a failure. Later steps can update both. Inspect `reports/gradient_inspection.json` for observed values.

The base embedding, attention, MLP and LM head remain frozen. The tests verify no gradients and no updates in base parameters, plus nonzero adapter gradients and actual updates. `inspect_gradients.py` compares an actual local model before/after a step and reloads the saved adapter.

LoRA is not guaranteed to match full fine-tuning; this project does not run a full-weight control on the 8 GB machine. RQ2 can establish parameter efficiency and specialization against the untouched base, but 'most of full fine-tuning's improvement' requires a full fine-tuning comparison that remains unmeasured.

References: [LoRA paper](https://arxiv.org/abs/2106.09685), [PEFT LoRA API](https://huggingface.co/docs/peft/package_reference/lora).

## Calculate this adapter's parameter count

Q and O are 896→896 projections. K and V are 896→128 because there are two KV heads of dimension 64. With rank 8, the added weights per layer are:

`8 × [(896+896) + (896+128) + (896+128) + (896+896)] = 45,056`.

Across 24 layers, this is `24 × 45,056 = 1,081,344` trainable parameters. Divide by `494,032,768 + 1,081,344` for 0.2184% of the augmented model. Rank 4 adds 540,672 parameters. The saved main adapter has 192 tensors stored in F32; see `reports/adapter_inspection.json`. This demonstrates why grouped-query K/V shapes matter for accounting.
