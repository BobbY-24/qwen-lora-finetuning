# Untouched Qwen baselines

Untouched local weights are the control. Validation baselines were executed before any fine-tuning. Final test baselines use the complete frozen test population after the protocol was fixed. Few-shot demonstrations come only from training terms.

| Population | Condition | Rows | ROUGE-L | Assistant PPL |
|---|---|---:|---:|---:|
| validation smoke | zero-shot | 16 | 0.0452 | 107.09 |
| validation smoke | two-shot | 16 | 0.0945 | 45.06 |
| final test | zero-shot | 181 | 0.0352 | 161.97 |
| final test | two-shot | 181 | 0.0727 | 41.36 |

Do not compare validation and test means as though they were the same population. Lexical scores and reference likelihood are proxies, not independent semantic correctness judgments.
