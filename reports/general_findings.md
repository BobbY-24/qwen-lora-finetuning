# General capability interpretation

The objective short-answer suite improved from **7/20** to **15/20** under the fixed strict evaluator: 9 newly passing answers and 1 newly failing answer. This small suite does not show a general collapse, but it cannot rule out broader degradation. Some improvements are formatting compliance (for example, returning only a sorted list), and others correct the answer itself. These are authored sanity prompts, not a benchmark.

A definite instruction-following regression is `general-16`: the prompt requires exactly `CHECK PASSED`. The base returns that exact text; LoRA returns `Check passed`, violating the explicit case-sensitive copy constraint.

The five summaries remain qualitative. Assistant inspection finds that LoRA's library summary omits the reopening time and online borrowing; its battery summary says “more efficient,” which is not established by battery duration alone. The original reference information is preserved. Do not let the 20-item strict aggregate conceal these losses of detail.

The OOD set is also mixed: higher ROUGE-L accompanies lower reference-content coverage. Specific technical errors remain (for example, the fine-tuned hoisting answer discusses C++ public/private access, and its sidecar answer describes storage rather than a supporting container). No broad OOD or general-capability improvement is established.
