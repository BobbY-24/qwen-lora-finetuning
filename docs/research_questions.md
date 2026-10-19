# What each research question requires

| Question | Implemented evidence | Boundary on conclusions |
|---|---|---|
| RQ1: Does SFT improve contextual slang interpretation? | Frozen unseen-term zero/few-shot versus LoRA predictions, reference metrics and human review sheets | Semantic improvement requires independent correctness labels; lexical/style effects may explain proxy gains |
| RQ2: Does LoRA provide most specialization gain with few parameters? | Exact adapter/base counts and comparison to untouched base | 'Most' relative to full fine-tuning remains unanswered without a full-weight control |
| RQ3: Does QLoRA reduce memory at similar quality? | Guarded CUDA 4/8-bit configuration | Unanswered on this MPS hardware; no numerical result |
| RQ4: How much data is needed? | Nested term-level fraction configuration and plotting support for actual runs | A threshold needs multiple controlled measured fractions; a single run is insufficient |
| RQ5: Does context help? | Fixed-weight term-only versus usage-sentence inference ablation | Source Context metadata is excluded because it can reveal the answer; independently sourced extra context remains untested |
| RQ6: Does specialization transfer to unseen slang terms? | Disjoint normalized term splits and paired test metrics | Does not rule out base-model pretraining familiarity or uncaught semantic aliases |
| RQ7: Does it transfer to another community? | Authored software-community OOD diagnostics with provenance and no split overlap | Small, partly synthetic diagnostic set is not broad OOD evidence |
| RQ8: Does it degrade general capabilities? | 25 fixed before/after prompts across five categories | Limited sanity check; no sweeping claim about catastrophic forgetting |
| RQ9: How do optimization and regularization affect training? | Configurable SGD/Adam/AdamW, LR, decay, dropout, rank, precision, accumulation and target modules | Configurability alone is not an ablation result; broad sweeps are not executed by default |

The study is credible when unanswered questions stay visibly unanswered. A failed or neutral specialization result still teaches model training if the controls, measurements and limitations are clear.

The supplemental one-epoch rank-8/full-data control, rank-4/full-data run and rank-8/25%-term run are predeclared as small experiments. Compare their matched-budget rank conditions and fixed-epoch data conditions separately from the main three-epoch model. Two data sizes cannot establish a threshold or scaling law.
