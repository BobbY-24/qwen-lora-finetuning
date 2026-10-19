# Actual ablations

Context inference ablation holds weights, decoding and rows fixed and removes the usage sentence. No extra context derived from definition metadata is used.

```json
{
  "Base term only": {
    "condition": "zero-shot",
    "rows": 181,
    "unique_terms": 154,
    "metrics": {
      "rouge_l": 0.0378257037572937,
      "reference_content_word_coverage": 0.11500773086733436,
      "empty": 0.0,
      "response_words": 38.32596685082873,
      "repeated_trigram_fraction": 0.005241844499787999
    },
    "assistant_loss": {
      "loss": 5.21711950746667,
      "perplexity": 184.402249019581,
      "target_tokens": 1196
    },
    "runtime_seconds": 237.74464716698276,
    "runtime_scope": "current invocation; earlier resumed generation not included",
    "peak_process_rss_bytes": 465043456,
    "mps_current_allocated_bytes": 988066048,
    "mps_driver_allocated_bytes": 2216067072,
    "token_limit_rate": 0.20994475138121546,
    "completed": 181,
    "failed": 0,
    "semantic_accuracy": null,
    "semantic_review": "requires independent human annotation"
  },
  "LoRA term only": {
    "condition": "LoRA",
    "rows": 181,
    "unique_terms": 154,
    "metrics": {
      "rouge_l": 0.16817169890253814,
      "reference_content_word_coverage": 0.14212978008883112,
      "empty": 0.0,
      "response_words": 4.7292817679558015,
      "repeated_trigram_fraction": 0.0
    },
    "assistant_loss": {
      "loss": 2.4951552824681023,
      "perplexity": 12.12361595743898,
      "target_tokens": 1196
    },
    "runtime_seconds": 47.872926499985624,
    "runtime_scope": "current invocation; earlier resumed generation not included",
    "peak_process_rss_bytes": 389382144,
    "mps_current_allocated_bytes": 992882944,
    "mps_driver_allocated_bytes": 2225504256,
    "token_limit_rate": 0.0,
    "completed": 181,
    "failed": 0,
    "semantic_accuracy": null,
    "semantic_review": "requires independent human annotation"
  }
}
```

## Predeclared training ablations and repeatability

| One-epoch condition | Rows | Trainable params | Updates | Test ROUGE-L |
|---|---:|---:|---:|---:|
| one_epoch_rank8 | 1412 | 1081344 | 177 | 0.2095 |
| one_epoch_rank4 | 1412 | 540672 | 177 | 0.2206 |
| one_epoch_quarter | 346 | 1081344 | 44 | 0.1577 |
| one_epoch_seed123 | 1412 | 1081344 | 177 | 0.2263 |

The term-only inference condition changes wording as well as removing the sentence, so it does not cleanly isolate context.

Rank comparison uses matched full data and epochs; alpha remains fixed. Data comparison keeps one epoch, so exposure and schedule change with sample size. The main selected three-epoch model is excluded from these curves.

```json
{
  "training_seeds": [
    42,
    123
  ],
  "split_seed": 42,
  "epochs": 1,
  "mean_test_rouge_l": 0.217882903471127,
  "sample_std_test_rouge_l": 0.011891988885321335,
  "limitation": "Two seeds only; no reliable seed-distribution confidence interval. Main three-epoch run has one seed."
}
```

Paired term contrasts (right condition minus left as named; exploratory, unadjusted):

```json
{
  "rank4_minus_rank8_one_epoch": {
    "metric": "rouge_l",
    "base_term_macro": 0.21050655324350695,
    "tuned_term_macro": 0.2256774246609539,
    "units": 154,
    "mean_delta": 0.015170871417446906,
    "std_delta": 0.13022714692046272,
    "ci95": [
      -0.005376213814342055,
      0.0363711483092556
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 0.1464853514648535
  },
  "quarter_minus_full_one_epoch": {
    "metric": "rouge_l",
    "base_term_macro": 0.21050655324350695,
    "tuned_term_macro": 0.15906363357662057,
    "units": 154,
    "mean_delta": -0.05144291966688636,
    "std_delta": 0.21314217556822992,
    "ci95": [
      -0.08597821140522861,
      -0.019582148277783304
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 0.0022997700229977
  },
  "base_term_only_minus_sentence": {
    "metric": "rouge_l",
    "base_term_macro": 0.036281943754320416,
    "tuned_term_macro": 0.03627899564908318,
    "units": 154,
    "mean_delta": -2.9481052372389285e-06,
    "std_delta": 0.04583179997697978,
    "ci95": [
      -0.007067470021034636,
      0.007304625253736919
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 0.9992000799920008
  },
  "lora_term_only_minus_sentence": {
    "metric": "rouge_l",
    "base_term_macro": 0.30172052346817235,
    "tuned_term_macro": 0.1700591159479693,
    "units": 154,
    "mean_delta": -0.13166140752020303,
    "std_delta": 0.30206637549748216,
    "ci95": [
      -0.18086312807779112,
      -0.08534164446943596
    ],
    "method": "paired bootstrap over normalized terms; fixed trained model, not training-seed uncertainty",
    "seed": 42,
    "repetitions": 10000,
    "exploratory_permutation_p": 9.999000099990002e-05
  }
}
```
