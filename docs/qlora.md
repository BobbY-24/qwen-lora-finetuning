# Quantization and optional QLoRA

Ordinary LoRA freezes higher-precision base weights and trains adapters. QLoRA combines a quantized frozen base (typically 4-bit NF4) with higher-precision LoRA adapters. Quantization approximates numbers with fewer representable values. Scales and metadata carry overhead, so nominal bits per weight are not total runtime memory. Activations, gradients, optimizer states and caches also matter.

FP32 has 32 bits per value and broad precision. FP16 has 16 bits, a narrower exponent range, and can require gradient scaling to avoid underflow. BF16 retains FP32's exponent width but fewer mantissa bits. Mixed precision uses different types for different operations or parameters. Here the base is BF16 and PEFT adapters normally FP32; this is not 4-bit training. CPU defaults to FP32. CUDA chooses BF16 where supported; explicit CUDA FP16 uses GradScaler and unscales gradients before clipping. The Apple BF16 path is validated by real forward/backward tests.

INT8 loading stores base weights with 8-bit representations. NF4 uses a nonuniform 4-bit codebook designed around normally distributed weights. Double quantization compresses quantization constants. `model/qlora.py` provides bitsandbytes 4-bit NF4 and 8-bit loading configuration; `prepare_model_for_kbit_training` prepares the frozen base before attaching LoRA. Gradients update adapters, not quantized base storage.

This implementation gates bitsandbytes training to CUDA. The present M3 environment has no CUDA or bitsandbytes, so QLoRA performance and memory comparisons are **not run**. A tested guard rejects MPS rather than falling back silently. The CUDA integration test is skipped on this machine; it must pass on the intended CUDA installation before launching a quantized experiment. Do not infer a hardware speedup from configuration support.

Gradient accumulation reduces activation memory per forward pass while approximating a larger token-averaged batch. It does not shrink the base model. Accelerate is installed as a PEFT dependency and provides useful device/mixed-precision/distributed utilities, but this single-device educational loop uses native PyTorch so accumulation and optimizer mechanics remain visible. No distributed training is needed.

References: [QLoRA paper](https://arxiv.org/abs/2305.14314), [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
