# Concept coverage checklist

Statuses describe evidence in this repository, not personal mastery. IMPLEMENTED means runnable code/test coverage; USED means exercised in local runs; EXPERIMENTED means compared under controlled conditions; EXPLAINED ONLY means no executed project experiment; NOT COVERED means absent.

| Concept | Status | Evidence / study location |
|---|---|---|
| PyTorch | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| torch.Tensor | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| nn.Module | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| Dataset | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| DataLoader | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| forward pass | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| backpropagation | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| automatic differentiation / autograd | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| computational graph | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| gradient | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| gradient descent | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| stochastic gradient descent | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| mini-batch training | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| optimizer | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| Adam | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| AdamW | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| learning rate | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| learning-rate scheduler | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| weight decay | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| loss function | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| cross-entropy loss | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| negative log-likelihood | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| training loop | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| validation loop | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| checkpointing | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| early stopping | IMPLEMENTED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| gradient clipping | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| neural network | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| multilayer perceptron | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| embedding layer | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| activation function | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| ReLU | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| GELU | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| softmax | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| normalization | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| LayerNorm | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| dropout | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| initialization | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| overfitting | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| underfitting | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| bias-variance tradeoff | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| regularization | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| hyperparameter tuning | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| Transformer | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| self-attention | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| multi-head attention | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| query | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| key | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| value | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| positional encoding | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| tokenizer | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| tokenization | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| vocabulary | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| attention mask | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| causal language modeling | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| masked language modeling | EXPLAINED ONLY | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| sequence length | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| context window | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| logits | USED | docs/qwen_architecture.md; docs/transformer_concepts.md; reports/model_inspection.json |
| perplexity | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| supervised fine-tuning / SFT | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| instruction tuning | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| transfer learning | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| parameter-efficient fine-tuning / PEFT | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| LoRA | EXPERIMENTED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| QLoRA | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| low-rank adaptation | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| quantization | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| 8-bit training | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| 4-bit training | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| mixed precision | USED | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| FP16 | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| BF16 | USED | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| gradient accumulation | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| Hugging Face Transformers | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| Hugging Face Datasets | USED | docs/pytorch_foundations.md; docs/training_walkthrough.md; src/community_qwen/; tests/ |
| PEFT library | USED | src/community_qwen/model/lora.py; docs/lora.md; reports/gradient_inspection.json |
| Accelerate | EXPLAINED ONLY | docs/qlora.md; src/community_qwen/model/qlora.py; reports/environment.json |
| train/validation/test split | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| baseline | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| ablation study | EXPERIMENTED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| control condition | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| random seed | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| reproducibility | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| learning curve | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| convergence | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| generalization | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| out-of-distribution evaluation | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| zero-shot | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| few-shot | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| fine-tuned | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| statistical significance | EXPLAINED ONLY | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |
| confidence interval | USED | docs/experimental_methodology.md; docs/evaluation_protocol.md; experiments/; reports/ |

QLoRA/8-bit/4-bit have guarded configuration code but are EXPLAINED ONLY here because no quantized training ran. Early stopping is implemented and tested for selection mechanics; a patience stop need not occur in a successful fixed-budget run. No NOT COVERED concept is hidden by an unsupported experiment claim.
