You are implementing a machine-learning portfolio and research-learning project using an **existing local Qwen model checkpoint**.

# Project

## CommunityQwen — Specializing a Local Qwen Model for Contextual Gen-Z Slang Understanding

The goal is to use my existing local Qwen model weights and fine-tune the model using the Hugging Face dataset:

`MLBtrio/genz-slang-dataset`

The main task is:

> Given an internet/Gen-Z slang term and its usage context, accurately explain what the expression means in that context.

The goal is NOT merely to make Qwen speak using Gen-Z slang.

The goal is to study:

- model training,
- PyTorch,
- Transformer internals,
- causal language modeling,
- supervised fine-tuning,
- LoRA,
- QLoRA,
- quantization,
- model evaluation,
- generalization,
- ablation studies,
- optimization,
- training dynamics,
- and experimental methodology.

This project should function as both:

1. a serious ML portfolio project,
2. a hands-on course in modern LLM training.

---

# Core research questions

The project should help answer:

### RQ1
Does supervised fine-tuning improve a small Qwen model's ability to interpret contextual slang?

### RQ2
Does LoRA provide most of the improvement of task specialization while updating only a small fraction of parameters?

### RQ3
Does QLoRA achieve similar results with lower memory usage?

### RQ4
How much training data is needed before specialization becomes measurable?

### RQ5
Does contextual information improve understanding compared with seeing the slang term alone?

### RQ6
Can training on one slang domain generalize to previously unseen terms?

### RQ7
Can specialization generalize out of distribution to other community-specific language?

### RQ8
Does specialization degrade general-purpose capabilities?

### RQ9
How do optimizer, learning rate, regularization, LoRA rank, precision, and training-data scale affect convergence and generalization?

---

# Critical first instruction

Before changing anything:

## Inspect the repository, environment, hardware, and existing model.

Determine:

- repository structure,
- Python version,
- virtual environment,
- installed packages,
- PyTorch version,
- Transformers version,
- PEFT version,
- Datasets version,
- Accelerate version,
- bitsandbytes availability if relevant,
- existing local Qwen checkpoint(s),
- exact model architecture,
- parameter count,
- tokenizer,
- dtype,
- device,
- CUDA availability,
- MPS availability,
- CPU fallback,
- disk availability if relevant.

The Qwen weights already exist locally.

## Do NOT redownload the base model unless absolutely necessary.

Find the local model.

Report the exact checkpoint path.

Do not assume whether it is:

- Qwen2,
- Qwen2.5,
- Qwen3,
- 0.5B,
- 1.5B,
- 1.7B,
- 3B,
- or another version.

Inspect the checkpoint.

Adapt the implementation to the actual model.

---

# Project stages

Organize the project conceptually as:

```text
Stage 1
PyTorch Foundations
        ↓
Stage 2
Inspect Real Qwen Architecture
        ↓
Stage 3
Dataset Audit and Task Construction
        ↓
Stage 4
Zero-Shot / Few-Shot Baselines
        ↓
Stage 5
Tokenization and Causal LM Data Pipeline
        ↓
Stage 6
Full PyTorch Training Mechanics
        ↓
Stage 7
LoRA Fine-Tuning
        ↓
Stage 8
QLoRA + Quantization
        ↓
Stage 9
Validation + Checkpointing + Training Reliability
        ↓
Stage 10
Model Evaluation
        ↓
Stage 11
Optimization + Regularization Experiments
        ↓
Stage 12
Ablation Studies
        ↓
Stage 13
OOD + Generalization Evaluation
        ↓
Stage 14
Statistical Analysis
        ↓
Stage 15
Error Analysis
        ↓
Stage 16
Documentation + Portfolio Results
```

Do not organize the project by arbitrary weeks.

Each stage should build on previous concepts.

---

# STAGE 1 — PyTorch Foundations

Before relying on Hugging Face abstractions, create a small educational module showing the mechanics of neural-network training.

This does NOT need to become a major secondary project.

The purpose is to expose the fundamentals used later during Qwen fine-tuning.

---

# 1.1 torch.Tensor

Create a small script:

```text
scripts/learn_tensors.py
```

Demonstrate:

```python
torch.Tensor
```

and common shapes:

```text
[B]
[B, T]
[B, T, D]
[B, H, T, D_head]
[B, T, V]
```

where:

```text
B = batch size
T = sequence length
D = hidden dimension
H = attention heads
V = vocabulary size
```

Demonstrate:

- indexing,
- reshaping,
- matrix multiplication,
- broadcasting,
- device movement,
- dtype conversion.

---

# 1.2 nn.Module

Explain that PyTorch neural networks subclass:

```python
torch.nn.Module
```

Create a very small educational model:

```python
class TinyMLP(nn.Module):
    ...
```

The purpose is to understand:

- parameters,
- modules,
- `forward()`,
- `model.parameters()`,
- gradients.

---

# 1.3 Multilayer perceptron

Implement:

```text
input
↓
Linear
↓
ReLU
↓
Linear
↓
GELU
↓
output
```

Explain:

- neural network,
- multilayer perceptron / MLP,
- activation function,
- ReLU,
- GELU.

Do not use this MLP as the final language model.

It is an educational demonstration.

---

# 1.4 Initialization

Inspect how PyTorch initializes:

```text
Linear weights
biases
embedding weights
```

Create a short explanation of:

- random initialization,
- Xavier initialization,
- Kaiming initialization.

Do not unnecessarily modify Qwen's pretrained initialization.

The purpose is to understand why training a pretrained model differs from training from scratch.

---

# 1.5 Embedding layer

Create an educational example with:

```python
nn.Embedding
```

Show:

```text
token IDs
[B,T]

↓

embedding lookup

↓

[B,T,D]
```

Explain how this corresponds to Qwen's token embedding layer.

---

# STAGE 2 — Inspect Real Qwen

Create:

```text
scripts/inspect_model.py
```

Report:

```text
checkpoint path
model class
parameter count
number of transformer layers
hidden size
intermediate size
attention heads
KV heads
vocabulary size
max context length
dtype
device
embedding layer
normalization mechanism
attention implementation
MLP implementation
LM head
```

Inspect module names such as:

```text
embed_tokens
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
lm_head
```

depending on the actual model.

---

# 2.1 Transformer architecture mapping

Create:

```text
docs/qwen_architecture.md
```

Connect general Transformer concepts to the actual Qwen implementation.

Cover:

```text
Transformer
decoder-only Transformer
embedding layer
self-attention
multi-head attention
query
key
value
attention mask
causal mask
feed-forward network
normalization
residual connection
logits
LM head
```

---

# 2.2 Q / K / V

Explain conceptually and mathematically:

```text
Q = XWq
K = XWk
V = XWv
```

and:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / sqrt(d_k))V
```

Explain:

- query,
- key,
- value,
- attention scores,
- scaling,
- softmax,
- weighted aggregation.

Do not reimplement Qwen.

A small educational standalone attention example is encouraged.

---

# 2.3 Multi-head attention

Explain how hidden states are divided into attention heads.

Document actual Qwen values:

```text
num_attention_heads
num_key_value_heads
head_dim
```

If the Qwen checkpoint uses grouped-query attention, explain GQA.

---

# 2.4 Positional information

Explain classic:

```text
sinusoidal positional encoding
```

briefly as historical context.

Then inspect what the actual Qwen checkpoint uses.

If it uses:

```text
RoPE
```

explain:

**Rotary Positional Embeddings / RoPE**

and why modern decoder-only LLMs often use this instead of additive sinusoidal embeddings.

---

# 2.5 Normalization

Explain:

```text
normalization
LayerNorm
RMSNorm
```

If Qwen uses RMSNorm, explain:

- how it differs from LayerNorm,
- why the generic Transformer concept is still normalization,
- which implementation the actual checkpoint uses.

---

# 2.6 MLP / feed-forward block

Inspect the Qwen feed-forward block.

If it uses gated projections such as:

```text
gate_proj
up_proj
down_proj
```

explain them.

Compare the architecture conceptually with a classic:

```text
Linear
→ GELU
→ Linear
```

Transformer feed-forward network.

---

# STAGE 3 — Hugging Face Dataset

Load:

```text
MLBtrio/genz-slang-dataset
```

using:

```text
Hugging Face Datasets
```

Create:

```text
scripts/inspect_dataset.py
```

Audit:

- rows,
- columns,
- unique slang terms,
- missing values,
- duplicate terms,
- duplicate examples,
- malformed rows,
- description lengths,
- example lengths,
- context lengths,
- unusual records.

Save:

```text
reports/dataset_audit.md
```

---

# 3.1 Hugging Face concepts

Document:

```text
Hugging Face
Hugging Face Datasets
Hugging Face Transformers
```

Explain their roles separately.

Datasets:

```text
data loading
splitting
mapping
processing
```

Transformers:

```text
model configuration
tokenizers
pretrained model loading
generation
```

---

# STAGE 4 — Train / Validation / Test

Create:

```text
train
validation
test
```

Use an approximate:

```text
80 / 10 / 10
```

split unless dataset characteristics justify another ratio.

## Critical leakage rule

Perform splitting at the **slang-term level**.

The same term must not exist across train and test.

Test:

```text
train_terms ∩ test_terms = ∅
```

and similarly for validation.

---

# 4.1 Experimental roles

Document explicitly:

### Training set
Used to update parameters.

### Validation set
Used for:

- model selection,
- early stopping,
- hyperparameter decisions.

### Test set
Used only for final evaluation.

Explain:

```text
data leakage
```

and why repeatedly evaluating on test data invalidates conclusions.

---

# STAGE 5 — Define the SFT Task

Main task:

> Explain the contextual meaning of a slang expression.

Example:

```text
SYSTEM:
You interpret specialized internet and community language accurately.

USER:
Term: cooked

Context:
"He tried to defend that take and got completely cooked in the replies."

Explain the expression in this context.

ASSISTANT:
"Cooked" means decisively defeated, embarrassed, or overwhelmed.
```

---

# 5.1 Instruction tuning

Explain:

```text
supervised fine-tuning / SFT
instruction tuning
transfer learning
```

Clarify:

### Transfer learning
Start from a pretrained model and adapt it.

### SFT
Train on input → desired-output examples.

### Instruction tuning
SFT where inputs are formulated as instructions/tasks.

This project is all three.

---

# STAGE 6 — Prompt Templates

Use multiple semantically equivalent prompt templates.

Examples:

```text
Explain "{term}" in this context:
{context}
```

```text
What does "{term}" mean here?

Usage:
{context}
```

```text
Interpret this community-specific expression.

Expression: {term}
Sentence: {context}
```

Do not make the output task inconsistent.

---

# STAGE 7 — Tokenizer / Tokenization

Inspect the Qwen tokenizer.

Cover:

```text
tokenizer
tokenization
vocabulary
token IDs
special tokens
attention mask
sequence length
context window
```

Use:

```python
tokenizer.apply_chat_template(...)
```

if supported.

Do not assume the chat format.

---

# 7.1 Sequence-length audit

Tokenize the dataset and calculate:

```text
P50
P75
P90
P95
P99
max
```

sequence lengths.

Choose:

```text
max_sequence_length
```

based on evidence.

Explain truncation.

---

# 7.2 Vocabulary

Report:

```text
vocabulary size
special tokens
EOS token
PAD token
```

Explain subword tokenization.

Inspect how slang expressions are tokenized.

This is especially interesting for slang terms.

Record examples where one slang term becomes:

```text
1 token
2 tokens
3+ tokens
```

---

# STAGE 8 — Dataset / DataLoader

Use appropriate PyTorch/Hugging Face dataset integration.

Make the actual data pipeline visible:

```text
Dataset
↓
tokenization
↓
collation
↓
DataLoader
↓
mini-batch
```

Explicitly cover:

```text
Dataset
DataLoader
mini-batch training
batch size
shuffle
padding
```

---

# STAGE 9 — Causal Language Modeling

Qwen is a causal language model.

Explain:

```text
causal language modeling
next-token prediction
```

Conceptually:

```text
x₁ x₂ x₃
       ↓
predict x₄
```

Model output:

```text
logits
[B,T,V]
```

Explain:

```text
logits
softmax
probability distribution
next-token prediction
```

---

# 9.1 Cross-entropy loss

Explain the loss function.

Cover:

```text
loss function
cross-entropy loss
negative log-likelihood
softmax
```

Conceptually:

```text
logits
↓
softmax
↓
probability assigned to gold token
↓
negative log probability
↓
cross-entropy loss
```

Explain the relationship between cross entropy and negative log-likelihood in classification / next-token prediction.

---

# 9.2 Assistant-only loss masking

Train primarily on assistant-response tokens.

Use:

```text
-100
```

for ignored labels.

Conceptually:

```text
system tokens       → -100
user tokens         → -100
padding tokens      → -100
assistant tokens    → actual token IDs
```

Create unit tests.

---

# 9.3 Masked language modeling comparison

Qwen should NOT be trained using masked language modeling.

However, document the distinction between:

### Causal language modeling

```text
The model predicts future tokens from previous tokens.
```

and:

### Masked language modeling

```text
"The weapon was [MASK] after the patch."
```

where a bidirectional encoder predicts masked tokens.

Explain that BERT-like models traditionally use MLM, while Qwen is decoder-only and uses causal language modeling.

This should be educational comparison only.

---

# STAGE 10 — Baselines

Before fine-tuning, evaluate the untouched Qwen.

Mandatory.

Create:

```text
scripts/evaluate.py
```

---

# 10.1 Zero-shot baseline

Prompt Qwen directly without examples.

Label this explicitly:

```text
zero-shot
```

---

# 10.2 Few-shot baseline

Add 1–3 demonstration examples before the test query.

Label:

```text
few-shot
```

Keep demonstrations fixed and contamination-free.

---

# 10.3 Control condition

Treat untouched Qwen as the primary:

```text
control condition
```

against which fine-tuned variants are compared.

Do not modify the test set.

---

# STAGE 11 — PyTorch Training Mechanics

The primary LoRA training implementation must expose the actual training loop.

Do not hide everything behind:

```python
Trainer.train()
```

unless a secondary comparison implementation is provided.

The educational implementation should expose:

```python
model.train()

for batch in train_loader:

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    loss = outputs.loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(...)

    optimizer.step()

    scheduler.step()
```

---

# 11.1 Forward pass

Document:

```text
input_ids
↓
embedding
↓
Transformer blocks
↓
hidden states
↓
LM head
↓
logits
↓
loss
```

---

# 11.2 Computational graph

Explain:

```text
forward pass
↓
PyTorch builds computational graph
↓
loss
↓
backward()
```

---

# 11.3 Autograd

Explain:

```text
automatic differentiation / autograd
```

and how:

```python
loss.backward()
```

uses the computational graph to calculate gradients.

---

# 11.4 Backpropagation

Explicitly cover:

```text
backpropagation
gradient
chain rule
```

Explain how the gradient of loss propagates backward through the network.

---

# 11.5 Gradient inspection

Create:

```text
scripts/inspect_gradients.py
```

Report for selected trainable parameters:

```text
parameter name
parameter norm
gradient norm
parameter value changed after optimizer step?
```

Demonstrate that:

```text
frozen Qwen base parameter → no update
LoRA parameter → receives gradient and updates
```

---

# STAGE 12 — Gradient Descent and Optimizers

Document:

```text
gradient descent
stochastic gradient descent
mini-batch gradient descent
optimizer
SGD
Adam
AdamW
```

Explain differences.

---

# 12.1 SGD

Support PyTorch:

```python
torch.optim.SGD
```

as an educational optimizer option for small experiments.

---

# 12.2 Adam

Support:

```python
torch.optim.Adam
```

---

# 12.3 AdamW

Default:

```python
torch.optim.AdamW
```

Explain decoupled weight decay.

---

# 12.4 Optimizer experiment

When computationally feasible, compare:

```text
Adam
AdamW
```

for LoRA.

SGD may be demonstrated on the educational MLP rather than requiring an expensive Qwen run if it performs poorly or is computationally wasteful.

---

# STAGE 13 — Learning Rate

Expose:

```text
learning rate
```

as a configuration variable.

Potential LoRA values:

```text
5e-5
1e-4
2e-4
```

Do not run every configuration unless practical.

---

# 13.1 Learning-rate scheduler

Support:

```text
learning-rate scheduler
```

such as:

```text
linear warmup + decay
```

or:

```text
cosine decay
```

Log learning rate over time.

---

# STAGE 14 — Weight Decay and Regularization

Expose:

```text
weight decay
```

Document it as a form of:

```text
regularization
```

Explain why regularization can improve generalization.

---

# 14.1 Dropout

Cover:

```text
dropout
```

both conceptually and via:

```text
LoRA dropout
```

If appropriate, inspect dropout already used inside Qwen.

---

# 14.2 Overfitting

Track:

```text
train loss
validation loss
```

Explain classic pattern:

```text
train loss ↓
validation loss ↓
then validation loss ↑
```

as overfitting.

---

# 14.3 Underfitting

Explain signs such as:

```text
high training loss
high validation loss
weak performance everywhere
```

---

# 14.4 Bias-variance tradeoff

Document conceptually:

```text
high bias → underfitting
high variance → overfitting
```

Connect it to:

- model capacity,
- dataset size,
- regularization,
- LoRA rank.

---

# STAGE 15 — LoRA

Use the PEFT library.

Cover:

```text
parameter-efficient fine-tuning / PEFT
LoRA
low-rank adaptation
```

Explain:

```text
W' = W + ΔW
ΔW = BA
```

Base weights stay frozen.

Only low-rank adapters train.

---

# 15.1 LoRA target modules

Inspect the architecture.

Do not blindly hard-code.

Candidate targets may include:

```text
q_proj
k_proj
v_proj
o_proj
```

and possibly:

```text
gate_proj
up_proj
down_proj
```

Choose based on the real checkpoint.

---

# 15.2 Parameter accounting

Report:

```text
total parameters
trainable parameters
frozen parameters
trainable %
```

---

# 15.3 LoRA hyperparameters

Expose:

```text
rank
alpha
dropout
target modules
```

Example baseline:

```yaml
rank: 8
alpha: 16
dropout: 0.05
```

---

# STAGE 16 — Mixed Precision

Cover:

```text
mixed precision
FP16
BF16
```

Inspect hardware support.

Use:

### CUDA
Prefer BF16 if hardware supports it reliably.

Otherwise use FP16 where appropriate.

### Apple MPS
Verify supported operations and dtype behavior.

### CPU
Use safe fallback.

Explain:

```text
FP32
FP16
BF16
```

and their tradeoffs.

---

# STAGE 17 — Gradient Accumulation

Support:

```text
gradient accumulation
```

Report:

```text
micro batch size
gradient accumulation steps
effective batch size
```

For one device:

```text
effective batch size
=
micro batch size
×
gradient accumulation steps
```

Implement accumulation correctly.

Do not call:

```python
optimizer.step()
```

after every micro-batch when accumulation is active.

---

# STAGE 18 — Gradient Clipping

Support:

```python
torch.nn.utils.clip_grad_norm_
```

Expose:

```text
max_grad_norm
```

Default approximately:

```text
1.0
```

Log unclipped gradient norm where practical.

---

# STAGE 19 — Validation Loop

Implement a distinct validation loop.

Use:

```python
model.eval()
```

and:

```python
torch.no_grad()
```

or inference mode.

Report:

```text
validation loss
generation metrics
```

Explain why gradient computation is unnecessary during validation.

---

# STAGE 20 — Checkpointing

Implement:

```text
checkpointing
```

Save:

```text
LoRA adapter
optimizer state
scheduler state
epoch
global step
best validation loss
configuration
random state / seed metadata
```

Do not duplicate the frozen Qwen model each run.

Support resume.

---

# STAGE 21 — Early Stopping

Support:

```text
early stopping
```

based on validation loss.

Example:

```text
patience = 2–3 evaluations
```

Log why training stopped.

---

# STAGE 22 — QLoRA

After the normal LoRA implementation works, add a separate QLoRA configuration.

Cover:

```text
QLoRA
quantization
4-bit training
8-bit training
```

---

# 22.1 Quantization

Explain:

```text
FP32
BF16
FP16
INT8
4-bit
```

Do not imply all representations behave identically.

---

# 22.2 8-bit

If supported by environment/hardware, allow an 8-bit loading experiment.

Do not force it if local hardware does not support the relevant libraries.

---

# 22.3 4-bit QLoRA

Use a standard QLoRA-compatible configuration if hardware supports it.

The frozen base model should be quantized while LoRA adapter parameters remain trainable in an appropriate higher precision.

Report:

```text
memory usage
trainable parameters
runtime
quality
```

---

# 22.4 Hardware limitation handling

QLoRA libraries may behave differently on:

```text
CUDA
MPS
CPU
```

Do not pretend unsupported features work.

If QLoRA is not practical on the current hardware:

1. implement configuration/support where reasonable,
2. test what can safely be tested,
3. clearly document the limitation,
4. do not fake results.

---

# STAGE 23 — Hugging Face Accelerate

Use:

```text
Accelerate
```

only where it improves:

- device placement,
- mixed precision,
- accumulation,
- portability.

Do not let Accelerate hide the actual training mechanics.

Document what it does.

---

# STAGE 24 — Experimental Telemetry

Record per run:

```text
training loss
validation loss
learning rate
gradient norm
tokens processed
examples processed
epoch
global step
runtime
peak memory
device
dtype
```

Save:

```text
metrics.csv
summary.json
training.log
```

---

# STAGE 25 — Perplexity

Calculate validation/test:

```text
perplexity
```

where meaningful.

Explain:

```text
perplexity = exp(cross_entropy_loss)
```

for token-level language modeling.

Important:

Do not treat perplexity as the sole metric for open-ended definition quality.

---

# STAGE 26 — Evaluation Metrics

Use multiple complementary metrics.

### Semantic similarity

Optional local embedding model.

### ROUGE-L

Lexical overlap.

### Key-concept coverage

Transparent approximate metric.

### Perplexity

Token-level modeling measure.

### Behavioral diagnostics

```text
empty response rate
response length
repetition
```

### Human review

Generate deterministic qualitative examples.

---

# STAGE 27 — Baseline vs Fine-Tuned Comparison

Evaluate:

```text
Base Qwen zero-shot
Base Qwen few-shot
Qwen + LoRA
Qwen + QLoRA
```

where supported.

Use identical test examples.

Use deterministic generation.

---

# STAGE 28 — Learning Curves

Create:

```text
learning curves
```

for:

```text
training loss vs step
validation loss vs step
```

and optionally:

```text
evaluation score vs training examples
```

---

# STAGE 29 — Convergence

Document:

```text
convergence
```

using:

- training loss stabilization,
- validation performance,
- gradient behavior,
- learning curves.

Do not assume lower train loss always means better model.

---

# STAGE 30 — Hyperparameter Tuning

Run only a modest experiment set.

Possible parameters:

```text
learning rate
LoRA rank
LoRA alpha
LoRA dropout
weight decay
batch size
gradient accumulation
number of epochs
```

Do not launch giant sweeps.

---

# STAGE 31 — Ablation Studies

Implement several controlled ablations.

## Ablation A — Training data size

Compare:

```text
10%
25%
50%
100%
```

or a smaller practical subset.

---

# Ablation B — LoRA rank

Compare:

```text
r = 4
r = 8
r = 16
```

where practical.

---

# Ablation C — Context

Evaluate:

```text
term only
term + sentence
term + additional context
```

---

# Ablation D — Fine-tuning strategy

Compare:

```text
zero-shot
few-shot
LoRA
QLoRA
```

---

# Ablation E — LoRA targets

Potential comparison:

```text
Q/V only
vs
Q/K/V/O
```

if computationally practical.

---

# STAGE 32 — Control Conditions

Clearly identify:

```text
untouched base Qwen
```

as primary control.

For each ablation, vary ONE factor when possible.

Keep constant:

```text
seed
test set
decoding
evaluation
training budget
```

unless the experiment explicitly changes one.

---

# STAGE 33 — Random Seeds

Centralize:

```text
random seed
```

Seed:

```text
Python
NumPy
PyTorch
CUDA
data split
DataLoader
```

where applicable.

Default:

```text
42
```

---

# STAGE 34 — Reproducibility

Record:

```text
git commit
Python version
PyTorch version
Transformers version
Datasets version
PEFT version
Accelerate version
model path
model identity
dataset revision
seed
device
dtype
config
```

---

# STAGE 35 — Multiple Seed Evaluation

For important final comparisons, run multiple seeds if computationally feasible.

Example:

```text
42
123
2026
```

If runtime prevents full repeated training, document the limitation.

Do not invent multi-seed results.

---

# STAGE 36 — Statistical Analysis

For major comparisons, calculate:

```text
mean
standard deviation
confidence interval
```

when multiple independent runs or bootstrap-able test examples permit.

Use a reasonable:

```text
95% confidence interval
```

---

# 36.1 Statistical significance

Where valid, test whether performance differences may be meaningful.

Potential methods:

```text
paired bootstrap
permutation test
paired test on per-example scores
```

Choose a method compatible with the metric.

Do not blindly run a t-test on inappropriate data.

Explain assumptions.

---

# STAGE 37 — Generalization

Primary evaluation must include slang terms absent from training.

Question:

> Did fine-tuning teach general slang interpretation behavior, or just memorize training terms?

---

# STAGE 38 — Out-of-Distribution Evaluation

Add an optional external/OOD evaluation set using community-specific terminology that is meaningfully different from Gen-Z slang.

Potential domains:

```text
gaming terminology
cybersecurity terminology
technical community slang
```

Do NOT train on the OOD evaluation data.

Evaluate:

```text
base Qwen
LoRA Qwen
QLoRA Qwen
```

Question:

> Does specializing on Gen-Z slang transfer to other specialized-language communities?

---

# STAGE 39 — Error Analysis

Create:

```text
scripts/error_analysis.py
```

Categorize:

### Improvement

```text
base wrong
fine-tuned correct
```

### Regression

```text
base correct
fine-tuned wrong
```

### Both fail

### Possible memorization

### Wrong sense selected

### Context ignored

### Hallucinated definition

### Overly generic definition

### Excessively slang-style response

Generate:

```text
reports/error_analysis.md
```

---

# STAGE 40 — General Capability Regression

Build 20–30 lightweight general prompts covering:

```text
basic reasoning
factual QA
summarization
instruction following
simple coding
```

Evaluate before/after.

This is a sanity check, not a benchmark claim.

Question:

> Did specialization cause catastrophic degradation?

---

# Repository Structure

Use approximately:

```text
community-qwen/
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   ├── lora.yaml
│   ├── qlora.yaml
│   └── ablations/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ood/
│
├── src/
│   └── community_qwen/
│       ├── __init__.py
│       │
│       ├── fundamentals/
│       │   ├── tensors.py
│       │   ├── mlp.py
│       │   └── attention_demo.py
│       │
│       ├── data/
│       │   ├── load.py
│       │   ├── audit.py
│       │   ├── split.py
│       │   ├── prompts.py
│       │   ├── tokenize.py
│       │   └── collator.py
│       │
│       ├── model/
│       │   ├── load.py
│       │   ├── inspect.py
│       │   ├── lora.py
│       │   └── qlora.py
│       │
│       ├── training/
│       │   ├── train.py
│       │   ├── validate.py
│       │   ├── checkpoint.py
│       │   ├── optimizer.py
│       │   ├── scheduler.py
│       │   ├── metrics.py
│       │   └── seed.py
│       │
│       ├── evaluation/
│       │   ├── generate.py
│       │   ├── metrics.py
│       │   ├── perplexity.py
│       │   ├── statistical.py
│       │   ├── regression.py
│       │   └── compare.py
│       │
│       └── utils/
│           ├── device.py
│           ├── logging.py
│           └── environment.py
│
├── scripts/
│   ├── learn_tensors.py
│   ├── inspect_gradients.py
│   ├── inspect_model.py
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── evaluate.py
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── run_ablations.py
│   └── error_analysis.py
│
├── experiments/
│   ├── baseline/
│   ├── few_shot/
│   ├── lora/
│   ├── qlora/
│   └── ablations/
│
├── tests/
│   ├── test_fundamentals.py
│   ├── test_split.py
│   ├── test_prompts.py
│   ├── test_tokenization.py
│   ├── test_loss_masking.py
│   ├── test_lora.py
│   ├── test_checkpoint.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── docs/
│   ├── pytorch_foundations.md
│   ├── qwen_architecture.md
│   ├── transformer_concepts.md
│   ├── training_walkthrough.md
│   ├── lora.md
│   ├── qlora.md
│   └── experimental_methodology.md
│
└── reports/
    ├── dataset_audit.md
    ├── baseline.md
    ├── comparison.md
    ├── ablations.md
    ├── ood_evaluation.md
    └── error_analysis.md
```

Adapt this to the existing repo instead of restructuring unnecessarily.

---

# Configuration

Example:

```yaml
seed: 42

model:
  local_path: AUTO_DISCOVER
  max_sequence_length: auto

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: auto_inspect

training:
  epochs: 3
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_ratio: 0.05
  scheduler: cosine

evaluation:
  do_sample: false
  max_new_tokens: 128
```

Adjust only based on actual hardware/data/model evidence.

---

# Experiment outputs

Every run should produce:

```text
experiments/lora/run_001/
├── config.yaml
├── environment.json
├── metrics.csv
├── summary.json
├── predictions.jsonl
├── training.log
├── learning_curve.png
└── adapter/
```

For QLoRA:

```text
experiments/qlora/run_001/
```

---

# Tests

Use:

```text
pytest
```

At minimum test:

### Dataset

- loads successfully,
- term-level split works,
- no leakage.

### Tokenization

- tokenizer output works,
- padding works,
- truncation works,
- attention mask correct.

### Loss masking

- system tokens ignored,
- user tokens ignored,
- padding ignored,
- assistant tokens trained.

### LoRA

- base parameters frozen,
- LoRA parameters trainable,
- LoRA gradients exist,
- LoRA parameters change.

### QLoRA

When supported:

- quantized base loads,
- adapters train,
- base remains frozen.

### Training

- finite loss,
- backward works,
- optimizer step changes trainable parameters,
- scheduler updates LR.

### Validation

- runs without gradients.

### Checkpoint

- save,
- resume,
- adapter reload.

### Evaluation

- deterministic evaluation reproducible.

Do not use fake tests.

---

# Important learning documentation

Create:

```text
docs/pytorch_foundations.md
```

Explicitly explain:

```text
PyTorch
torch.Tensor
nn.Module
Dataset
DataLoader
neural network
multilayer perceptron
embedding layer
activation function
ReLU
GELU
initialization
forward pass
computational graph
autograd
backpropagation
gradient
gradient descent
SGD
Adam
AdamW
learning rate
scheduler
weight decay
loss function
cross entropy
negative log likelihood
softmax
training loop
validation loop
checkpointing
early stopping
gradient clipping
overfitting
underfitting
bias-variance tradeoff
regularization
hyperparameter tuning
```

---

Create:

```text
docs/transformer_concepts.md
```

Explain:

```text
Transformer
decoder-only Transformer
self-attention
multi-head attention
query/key/value
attention mask
causal masking
positional encoding
RoPE
tokenizer
tokenization
vocabulary
sequence length
context window
logits
perplexity
causal language modeling
masked language modeling
LayerNorm
RMSNorm
MLP
residual connection
```

---

Create:

```text
docs/lora.md
```

Explain:

```text
transfer learning
supervised fine-tuning
instruction tuning
PEFT
LoRA
low-rank adaptation
```

---

Create:

```text
docs/qlora.md
```

Explain:

```text
QLoRA
quantization
8-bit training
4-bit training
mixed precision
FP16
BF16
gradient accumulation
```

---

Create:

```text
docs/experimental_methodology.md
```

Explain:

```text
train/validation/test split
baseline
control condition
ablation study
random seed
reproducibility
learning curve
convergence
generalization
OOD evaluation
zero-shot
few-shot
fine-tuned
statistical significance
confidence interval
```

Use the actual project experiments as examples.

---

# Important conceptual distinction

Some terms are important to understand but should not be forced into the Qwen implementation unnaturally.

For example:

### ReLU

Teach it using the educational MLP.

Qwen may use another activation.

### LayerNorm

Teach standard LayerNorm, but inspect whether Qwen actually uses RMSNorm.

### Positional encoding

Explain classical sinusoidal positional encoding, then study Qwen's actual positional mechanism such as RoPE.

### Masked language modeling

Explain and contrast with causal language modeling.

Do NOT train Qwen using MLM.

### SGD

Teach and optionally test on a small model.

Do not waste major Qwen runs if SGD is clearly unsuitable.

The objective is conceptual mastery, not keyword stuffing.

---

# Required visualizations

Produce simple matplotlib plots for:

### Training

```text
training loss vs step
validation loss vs step
```

### Learning rate

```text
learning rate vs step
```

### Data scaling

```text
test performance vs training-set size
```

### LoRA rank

```text
performance vs trainable parameters
```

### Base vs fine-tuned

A concise metric comparison.

Do not fabricate results.

---

# Final comparison table

Aim to generate something like:

```text
Model                Trainable Params   Memory   Test Score   OOD Score
-----------------------------------------------------------------------
Base Qwen             0                  ...      ...          ...
Base Qwen Few-Shot    0                  ...      ...          ...
Qwen + LoRA           ...                ...      ...          ...
Qwen + QLoRA          ...                ...      ...          ...
```

Only include actual results.

---

# Final report

At completion, provide:

## 1. Environment

Report:

```text
local Qwen checkpoint
model family
parameter count
device
dtype
Python
PyTorch
Transformers
PEFT
Datasets
Accelerate
```

## 2. Dataset

Report:

```text
rows
unique terms
train terms
validation terms
test terms
missing values
token-length distribution
```

## 3. Qwen architecture

Describe:

```text
embeddings
Transformer layers
attention
Q/K/V
heads
normalization
MLP
positional mechanism
LM head
```

## 4. Baselines

Report:

```text
zero-shot
few-shot
```

## 5. LoRA training

Report:

```text
target modules
rank
alpha
dropout
trainable params
trainable %
learning rate
optimizer
scheduler
batch size
gradient accumulation
effective batch size
epochs
runtime
memory
```

## 6. QLoRA

Report:

```text
quantization
bit width
dtype
memory difference
performance difference
```

if supported and actually run.

## 7. Training dynamics

Report:

```text
initial loss
final loss
best validation loss
gradient norms
learning curve
convergence
overfitting signs
```

## 8. Evaluation

Compare:

```text
zero-shot
few-shot
LoRA
QLoRA
```

## 9. Ablations

Report only actual runs.

## 10. OOD evaluation

Evaluate unseen community-specific terminology if available.

## 11. Statistics

Where valid report:

```text
mean
standard deviation
95% confidence interval
significance analysis
```

## 12. Error analysis

Show representative:

```text
improvements
regressions
both-fail examples
context failures
hallucinations
```

## 13. General-capability regression

Report whether specialization caused obvious deterioration.

## 14. Concepts learned

Create a checklist covering every concept listed below.

Mark each as:

```text
IMPLEMENTED
USED
EXPERIMENTED
EXPLAINED ONLY
NOT COVERED
```

---

# Required concept checklist

## Core training

```text
PyTorch
torch.Tensor
nn.Module
Dataset
DataLoader
forward pass
backpropagation
automatic differentiation / autograd
computational graph
gradient
gradient descent
stochastic gradient descent
mini-batch training
optimizer
Adam
AdamW
learning rate
learning-rate scheduler
weight decay
loss function
cross-entropy loss
negative log-likelihood
training loop
validation loop
checkpointing
early stopping
gradient clipping
```

## Deep learning

```text
neural network
multilayer perceptron
embedding layer
activation function
ReLU
GELU
softmax
normalization
LayerNorm
dropout
initialization
overfitting
underfitting
bias-variance tradeoff
regularization
hyperparameter tuning
```

## Transformer / LLM

```text
Transformer
self-attention
multi-head attention
query
key
value
positional encoding
tokenizer
tokenization
vocabulary
attention mask
causal language modeling
masked language modeling
sequence length
context window
logits
perplexity
```

## Fine-tuning

```text
supervised fine-tuning / SFT
instruction tuning
transfer learning
parameter-efficient fine-tuning / PEFT
LoRA
QLoRA
low-rank adaptation
quantization
8-bit training
4-bit training
mixed precision
FP16
BF16
gradient accumulation
Hugging Face Transformers
Hugging Face Datasets
PEFT library
Accelerate
```

## Experimental methodology

```text
train/validation/test split
baseline
ablation study
control condition
random seed
reproducibility
learning curve
convergence
generalization
out-of-distribution evaluation
zero-shot
few-shot
fine-tuned
statistical significance
confidence interval
```

---

# Resume-oriented outputs

When the project is complete, provide 2–3 truthful resume bullets emphasizing measured results.

Potential vocabulary:

```text
PyTorch
Hugging Face
Transformers
LoRA
QLoRA
fine-tuning
PEFT
GPU training
mixed precision
ablation studies
model evaluation
OOD evaluation
```

Do NOT create resume claims without measured evidence.

Example structure only:

```text
Fine-tuned a local Qwen language model using PyTorch, Hugging Face,
and LoRA for contextual slang understanding, updating X% of model
parameters while improving held-out semantic evaluation by Y%.
```

Another:

```text
Designed ablation studies across LoRA rank, training-data scale,
context availability, and quantization; evaluated zero-shot,
few-shot, LoRA, and QLoRA models on unseen and OOD terminology.
```

Populate X and Y only with real measurements.

---

# Final files I should personally study

Identify the 8–12 most educational implementation files.

Prioritize:

```text
training loop
loss masking
tokenization
DataLoader
LoRA
QLoRA
optimizer
validation
evaluation
statistical analysis
```

For each file explain:

1. what goes in,
2. what comes out,
3. what PyTorch mechanism it teaches,
4. what I should modify myself.

---

# Personal exercises

Give me at least 10 exercises after implementation.

Examples:

1. Implement a tiny MLP using `nn.Module`.
2. Replace GELU with ReLU and observe behavior.
3. Write manual softmax for a small tensor.
4. Inspect `loss.backward()` gradients.
5. Switch AdamW to Adam.
6. Remove weight decay.
7. Disable gradient clipping.
8. Change LoRA rank.
9. Implement gradient accumulation manually.
10. Compare zero-shot vs few-shot vs LoRA.
11. Add a new OOD community dataset.
12. Calculate a bootstrap confidence interval.

Do not implement these exercises for me unless necessary to establish the base project.

---

# Implementation workflow

Proceed in this order:

### Phase A — Audit

Inspect:

```text
repository
environment
hardware
local Qwen
dataset
```

### Phase B — Fundamentals

Create lightweight PyTorch educational examples.

### Phase C — Architecture

Inspect Qwen and document Transformer concepts.

### Phase D — Dataset

Audit and build leakage-safe splits.

### Phase E — Baselines

Run zero-shot and few-shot evaluations.

### Phase F — LoRA

Implement and smoke-test training.

### Phase G — Main LoRA run

Train only after smoke tests pass.

### Phase H — QLoRA

Implement after LoRA is stable.

### Phase I — Evaluation

Compare all supported conditions.

### Phase J — Ablations

Run controlled experiments.

### Phase K — OOD

Evaluate generalization.

### Phase L — Statistics and error analysis

Analyze results.

### Phase M — Documentation

Produce final educational and portfolio artifacts.

---

# Smoke-test requirement

Before any expensive run, use a tiny subset.

Example:

```text
16–32 samples
2–10 optimization steps
```

Verify:

```text
finite loss
forward pass works
backpropagation works
LoRA gradients exist
parameters update
validation works
checkpoint saves
checkpoint reloads
```

For QLoRA separately verify quantized loading.

Do not spend substantial compute before these checks pass.

---

# Important constraints

1. Reuse the existing local Qwen weights.
2. Do not unnecessarily redownload models.
3. Do not use paid APIs.
4. Do not require web services during training.
5. Do not use agent frameworks.
6. Do not add LangChain.
7. Do not add vector databases.
8. Do not add RAG.
9. Do not build a web application.
10. Do not build a database.
11. Do not introduce distributed training unless truly necessary.
12. Do not hide important PyTorch mechanics behind `Trainer`.
13. Do not fabricate experiment results.
14. Do not leak test terms into training.
15. Do not leak gold definitions into prompts.
16. Do not accidentally train the frozen base model.
17. Do not optimize against the test set.
18. Do not claim statistical significance without valid analysis.
19. Do not force QLoRA if hardware cannot support it.
20. Prefer a clear research implementation over excessive code.

---

# Success criteria

The project is successful when:

```text
✓ Existing Qwen weights are reused
✓ Dataset is loaded from Hugging Face
✓ Dataset is audited
✓ Term-level leakage is controlled
✓ PyTorch Tensor and nn.Module concepts are demonstrated
✓ Dataset and DataLoader are understood
✓ Zero-shot baseline exists
✓ Few-shot baseline exists
✓ Causal LM tokenization is implemented
✓ Assistant-only loss masking works
✓ Explicit PyTorch training loop works
✓ Forward pass is understood
✓ Autograd is demonstrated
✓ Backpropagation is demonstrated
✓ Gradient behavior is inspectable
✓ AdamW optimization works
✓ Scheduler works
✓ Weight decay is configurable
✓ Gradient accumulation works
✓ Gradient clipping works
✓ Mixed precision is handled correctly
✓ Checkpointing works
✓ Validation loop works
✓ Early stopping works
✓ LoRA trains correctly
✓ Base Qwen remains frozen
✓ QLoRA is implemented/tested if hardware supports it
✓ Transformer internals are documented
✓ Q/K/V and attention are understood
✓ Tokenizer/vocabulary/context window are inspected
✓ Perplexity is measured
✓ Learning curves are produced
✓ At least one meaningful ablation is run
✓ OOD evaluation is attempted
✓ Generalization is measured
✓ Multiple seeds are used where practical
✓ Confidence intervals are produced where statistically valid
✓ Base vs fine-tuned comparison exists
✓ Error analysis exists
✓ General capability regression is checked
✓ Tests pass
✓ Results are reproducible
✓ README clearly explains the research methodology
```

The final result should not merely prove that I can run Hugging Face fine-tuning code.

It should demonstrate that I understand:

```text
data
→ tensors
→ tokenization
→ Transformer
→ forward pass
→ logits
→ loss
→ computational graph
→ autograd
→ gradients
→ optimization
→ LoRA / QLoRA
→ validation
→ evaluation
→ generalization
→ statistical evidence
```

The priority is **learning real model training while producing a credible, measurable, resume-worthy Qwen specialization project**.