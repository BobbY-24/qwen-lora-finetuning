# Transformer concepts through this checkpoint

A Transformer processes token representations with attention and position-wise feed-forward networks. Qwen is a decoder-only Transformer: its causal mask allows each position to use only earlier positions and itself. Unlike a recurrent network, training can calculate many token positions in parallel while preserving this information constraint.

For hidden states X, projections form `Q=XWq`, `K=XWk`, `V=XWv`. A query represents what a position seeks; a key represents what each position offers for matching; a value carries the information to combine. Attention computes `softmax(QKᵀ/sqrt(d_k) + mask)V`. Dot products measure compatibility, scaling controls magnitude, softmax turns scores into weights, and the weighted sum aggregates values. In PyTorch Linear notation, stored weight matrices are transposed relative to this row-vector formula.

Multi-head attention repeats this operation in different projected subspaces. This checkpoint has 14 query heads but only 2 key/value heads: grouped-query attention (GQA) shares each KV head among seven query heads, reducing the KV cache. With hidden size 896, each head has dimension 64. `fundamentals/attention_demo.py` demonstrates dense causal attention, not the full Qwen implementation.

An attention mask can hide padded positions. A causal mask hides the future. They serve different purposes and neither replaces the loss mask. At generation time, the KV cache reuses earlier keys/values instead of recalculating every previous position.

Classic sinusoidal positional encoding adds position-dependent sine/cosine vectors to embeddings. Qwen uses Rotary Positional Embeddings (RoPE), rotating pairs of query/key coordinates by position-dependent angles. Their dot products thereby encode relative positional differences. RoPE avoids a learned additive position table but does not make context length unlimited: this checkpoint is configured for 32,768 positions. Training sequences here are only 128 tokens.

Normalization stabilizes feature scale. LayerNorm subtracts feature mean and divides by standard deviation, then applies learned affine parameters. RMSNorm scales by root mean square without subtracting the mean; Qwen uses RMSNorm with epsilon 1e-6. Residual connections add a block's transformed signal to its input, preserving a direct route for information and gradients. Qwen applies normalization before attention and before its MLP.

A classic Transformer MLP is Linear → GELU → Linear. This checkpoint uses a gated feed-forward network: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`. SiLU is `x * sigmoid(x)`. The two intermediate projections have width 4,864; their elementwise product is projected back to width 896. ReLU/GELU in the educational MLP are concepts to compare, not replacements imposed on Qwen.

A tokenizer splits text into vocabulary units, often subwords or bytes rather than words. Tokenization maps these units to integer IDs, including special tokens for chat boundaries. Unusual slang may split into several pieces. The tokenizer has 151,643 base vocabulary entries and 151,665 entries including added tokens, while the model's padded embedding/output tables have 151,936 rows. See the actual inspection JSON and audit for authoritative counts and examples. Vocabulary length and allocated LM-head dimension need not be equal.

The LM head maps hidden states to logits `[B,T,V]`. This model ties its head weight to its token embedding weight. Causal language modeling trains next-token prediction. Masked language modeling instead replaces tokens such as 'The weapon was [MASK] after the patch' and uses bidirectional context to recover them, traditionally in BERT-like encoders. We explain MLM but do not train Qwen with it.

Perplexity is `exp(mean cross entropy)` over the same supervised token population. Lower perplexity means the reference tokens receive greater likelihood; it is not a direct score of open-ended meaning accuracy. Comparisons require the same tokenizer, reference text, masking convention and population.
