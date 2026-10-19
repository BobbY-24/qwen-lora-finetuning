# OOD diagnostic

Assistant-authored technical-community examples, checked for term overlap; not a human-validated benchmark.

```json
{
  "Base": {
    "condition": "zero-shot",
    "rows": 12,
    "unique_terms": 12,
    "metrics": {
      "rouge_l": 0.11509671727861777,
      "reference_content_word_coverage": 0.2557539682539683,
      "empty": 0.0,
      "response_words": 51.583333333333336,
      "repeated_trigram_fraction": 0.0043616584564860434
    },
    "assistant_loss": {
      "loss": 3.625421579464062,
      "perplexity": 37.540546147368694,
      "target_tokens": 185
    },
    "runtime_seconds": 19.05283695901744,
    "runtime_scope": "current invocation; earlier resumed generation not included",
    "peak_process_rss_bytes": 356990976,
    "mps_current_allocated_bytes": 988066048,
    "mps_driver_allocated_bytes": 2215608320,
    "token_limit_rate": 0.8333333333333334,
    "completed": 12,
    "failed": 0,
    "semantic_accuracy": null,
    "semantic_review": "requires independent human annotation"
  },
  "LoRA": {
    "condition": "LoRA",
    "rows": 12,
    "unique_terms": 12,
    "metrics": {
      "rouge_l": 0.1691176255225113,
      "reference_content_word_coverage": 0.11607142857142856,
      "empty": 0.0,
      "response_words": 16.25,
      "repeated_trigram_fraction": 0.011074800290486566
    },
    "assistant_loss": {
      "loss": 3.369641345900458,
      "perplexity": 29.06809979486123,
      "target_tokens": 185
    },
    "runtime_seconds": 7.723094250017311,
    "runtime_scope": "current invocation; earlier resumed generation not included",
    "peak_process_rss_bytes": 468582400,
    "mps_current_allocated_bytes": 992882944,
    "mps_driver_allocated_bytes": 2223996928,
    "token_limit_rate": 0.08333333333333333,
    "completed": 12,
    "failed": 0,
    "semantic_accuracy": null,
    "semantic_review": "requires independent human annotation"
  }
}
```
