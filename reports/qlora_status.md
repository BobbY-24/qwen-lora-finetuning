# QLoRA status

Not run. Observed hardware is Apple M3; CUDA is unavailable and bitsandbytes is not installed. The project provides a CUDA-only NF4/8-bit configuration and optional real checkpoint integration test. The MPS rejection test passes; the CUDA test is explicitly skipped. There is no measured quantization bit width, memory reduction, runtime comparison or QLoRA score to report.

Ordinary LoRA uses BF16 base weights and higher-precision adapters. It must not be relabeled QLoRA.
