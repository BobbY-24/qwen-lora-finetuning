# PyTorch foundations

Read `scripts/learn_tensors.py` first, then trace one example through the real training loop. PyTorch combines multidimensional arrays, differentiable operations, and neural-network modules. A `torch.Tensor` has shape, dtype, device and (optionally) a gradient history. Integer token IDs cannot carry ordinary floating-point gradients; embedding weights can.

## Shapes and operations

A batch label vector is `[B]`; token IDs and padding masks are `[B,T]`; embeddings and hidden states are `[B,T,D]`; attention projections can be reshaped to `[B,H,T,D_head]`; vocabulary logits are `[B,T,V]`. Indexing selects part of a tensor. Reshaping changes the view of its elements without changing their order; transposing changes axis order. Matrix multiplication combines features. Broadcasting adds a vector such as a bias `[D]` to every position in `[B,T,D]`. `.to(device)` moves storage; `.to(dtype)` changes representation. Neither operation changes the mathematical role of an axis.

## Modules, MLPs and initialization

A neural network composes parameterized functions. `nn.Module` registers child modules and `nn.Parameter` objects, implements `forward()`, and exposes parameters through `model.parameters()`. A multilayer perceptron (MLP) applies dense linear transformations and nonlinear activation functions. Our educational `TinyMLP` is Linear → ReLU → Linear → GELU; it is not a language model. ReLU is `max(0,x)`. GELU approximately weights x by the probability that a standard normal variable is below x, giving a smooth gate. Without nonlinearities, a chain of linear layers collapses to a linear transformation.

PyTorch Linear's default reset uses Kaiming-uniform machinery with `a=sqrt(5)`, producing the familiar bound `1/sqrt(fan_in)`; its bias is uniform within that bound. Embedding initializes weights from a standard normal, with a padding row zeroed when configured. Xavier initialization balances input/output variance; Kaiming initialization accounts for rectifier nonlinearities. Random initialization breaks symmetry. Inspect these distributions in the small module; do not reinitialize pretrained Qwen. Its checkpoint represents previous learning, and transfer learning starts there.

`nn.Embedding(V,D)` maps `[B,T]` IDs to `[B,T,D]` vectors by table lookup. Gradients accumulate into selected embedding rows when the table is trainable. In this LoRA project, Qwen's embedding table is frozen; the small MLP and embedding example expose the underlying mechanics separately.

## Dataset to loss

A Dataset defines examples and indexing; Hugging Face Datasets also provides column transforms and disk-backed tables. A DataLoader samples/shuffles examples and calls a collator to construct mini-batches. Mini-batch training estimates a population gradient from a manageable subset. Batch size controls examples per forward call; variable-length examples need padding. The attention mask hides padded input positions and `-100` labels exclude padded targets from loss.

A forward pass maps IDs → embeddings → Transformer blocks → hidden states → LM head → logits. A logit is an unnormalized score. Softmax computes `p_i=exp(z_i)/sum_j exp(z_j)`; implementations subtract the largest logit for numerical stability. Cross-entropy with a one-hot target is the negative log-likelihood `-log p_gold`. For causal language modeling, the logit at position t predicts the token at t+1. Transformers performs that shift inside its loss. Our labels remain aligned with input positions, and loss counting excludes position zero.

The prompt is visible to the model but ignored as a training target. Only assistant tokens (including the end-of-turn marker) contribute to the loss. `data/tokenize.py` verifies the exact chat prefix before masking. This avoids teaching the model to reproduce the user's question instead of answering it.

## Autograd and optimization

During forward computation, PyTorch builds a computational graph for differentiable operations involving trainable parameters. Automatic differentiation / autograd traverses this graph in reverse when `loss.backward()` is called. Backpropagation applies the chain rule, combining local derivatives to produce each parameter's gradient: the sensitivity of loss to a small parameter change. Gradients accumulate until explicitly cleared.

Gradient descent updates `theta <- theta - learning_rate * gradient`. Full-batch gradient descent uses all examples. Stochastic gradient descent (SGD) traditionally uses one example; practical SGD optimizers often receive mini-batch gradients. Adam tracks moving first and second gradient moments and applies coordinate-wise scaling. AdamW decouples weight decay from the adaptive gradient update. In Adam, coupled L2 regularization changes the gradient that enters moment estimates; AdamW shrinks weights separately. This implementation supports SGD, Adam and AdamW, defaulting to AdamW.

The learning rate controls update size. A learning-rate scheduler can warm up the rate and then decrease it linearly or with a cosine. `scheduler.step()` follows an optimizer update, never every micro-batch. Our logs record the LR used for the update. Weight decay is configurable regularization, but it does not guarantee improved generalization.

Gradient accumulation combines several micro-batches before one update. We weight each loss by its number of supervised tokens divided by the window's total supervised tokens. This matches a single token-averaged batch even when answers differ in length or the final window is short. Effective nominal batch size is micro-batch size × accumulation steps on one device. Gradient clipping limits the joint norm before updating; telemetry records the norm before clipping. CUDA FP16 gradients are unscaled before clipping.

## Reliability and learning dynamics

The training loop enables dropout with `model.train()`. The validation loop uses `model.eval()` and `torch.no_grad()`; it measures held-out loss without changing parameters or recording an autograd graph. Dropout randomly suppresses activations while training; LoRA dropout regularizes adapter inputs. Validation restores the previous model mode.

Checkpointing stores the adapter, optimizer moments, scheduler position, epoch/batch cursor, global step, best validation loss, patience counter, configuration fingerprint and random-number states. It does not duplicate frozen base weights. A checkpoint is published only after its files are written. Resume must use the same source/config/data identity. Epoch-addressed shuffling lets the loader skip completed batches without changing the dropout RNG.

Early stopping ends training when validation loss fails to improve for a configured number of evaluations. Best-checkpoint selection uses validation, never test. A lower training loss with a rising validation loss suggests overfitting; high training and validation losses can suggest underfitting. The bias-variance tradeoff connects limited capacity/strong constraints with bias, and sensitivity to small datasets with variance. More data, regularization and LoRA rank affect this balance; the curves are evidence, not automatic diagnoses. Hyperparameter tuning compares settings on validation, after which the test remains a final check.

Try modifying one factor at a time. The learning goal is to predict the result from the mechanism before observing it.
