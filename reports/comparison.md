# Measured comparison

Lexical proxies on frozen unseen-term test examples. Semantic accuracy requires independent review.

| Condition | Rows | ROUGE-L (row mean) | Reference-word coverage | Assistant PPL |
|---|---:|---:|---:|---:|
| Base zero-shot | 181 | 0.0352 | 0.1378 | 161.97 |
| Base few-shot | 181 | 0.0727 | 0.1021 | 41.36 |
| LoRA | 181 | 0.2967 | 0.2890 | 7.74 |

## Paired statistics

```json
{
  "Base few-shot": {
    "metric": "rouge_l",
    "base_term_macro": 0.036281943754320416,
    "tuned_term_macro": 0.07455410233978768,
    "units": 154,
    "mean_delta": 0.038272158585467266,
    "std_delta": 0.15279478121476628,
    "ci95": [
      0.01571802453523728,
      0.0639567584076964
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 0.0006999300069993001
  },
  "LoRA": {
    "metric": "rouge_l",
    "base_term_macro": 0.036281943754320416,
    "tuned_term_macro": 0.30172052346817235,
    "units": 154,
    "mean_delta": 0.265438579713852,
    "std_delta": 0.330173082453437,
    "ci95": [
      0.21401595248054925,
      0.3179591246448135
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 9.999000099990002e-05
  }
}
```

Intervals condition on the trained adapter and capture term resampling, not seed variation. P-values are exploratory and unadjusted. QLoRA was not run on MPS. No semantic or confirmatory significance claim is made.
