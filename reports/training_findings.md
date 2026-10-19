# Main training findings

The rank-8 Q/K/V/O adapter completed 531 updates / three full epochs, selected epoch three by lowest full-validation loss, and stopped normally. Validation CE was 5.5334 before training, 2.1217 at epoch one, 2.0366 at epoch two and 2.0328 at epoch three. The large first-epoch reduction and small final-epoch gain suggest diminishing improvement on the reference-likelihood objective. They do not prove semantic mastery or absence of overfitting.

The model made 4,236 example exposures and processed 27,651 supervised assistant tokens. These are training exposures, not unique-example counts. The unique training population is 1,412 examples / 1,231 normalized terms. The last optimizer window contains four examples; token-weighted accumulation handles it without treating it as a full eight-example window.

Maximum pre-clipping gradient norm was 39.611. All observed losses and norms were finite. Clipping at 1.0 limits the applied joint gradient norm; a large unclipped value alone does not establish exploding gradients. The first update used a small warmup LR, and the final updates approached zero under cosine decay.

Runtime was 921.48 seconds (15.36 minutes). Sampled MPS tensor allocation peaked at 1.143 GB; final MPS driver allocation was 3.418 GB. The driver includes allocator/framework memory beyond live tensors. These measurements are not interchangeable with CPU RSS or total system memory and cannot establish a QLoRA memory advantage.

The saved best adapter contains 1,081,344 trainable parameters. The 494,032,768 base parameters stayed frozen. Gradient smoke evidence directly checks sampled real base weights and adapter changes; tests check the full parameter set on a small real Qwen architecture. Reload tests compare actual checkpoint-adapter logits, and separate loop tests exercise optimizer/scheduler/RNG continuation.

Only validation informed model selection. Final test outputs have not been used to change the training configuration.
