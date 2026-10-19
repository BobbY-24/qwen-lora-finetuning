# Artifact contract

Processed JSONL has string `id` (unique source-row ID), integer `source_row`, string `term`, normalized string `term_key`, usage `context`, weak reference `definition`, and nullable preserved `source_context_metadata`. IDs persist through training/evaluation; rows are never identified by their current list position. Rejected raw row numbers and reason lists are saved separately. Raw values remain immutable. The dataset manifest supplies source revision, snapshot hash, split hashes/counts, tokenizer information and length distributions.

Prediction JSONL retains source fields and adds string `prediction`, integer prompt/generated token counts, boolean `hit_token_limit`, and a `scores` object. Each output directory's manifest lists every expected ID, input hash, condition, demonstrations, model/config fingerprint, adapter hash and final-test role. Reference fields are joined only after generation, whose input interface admits only term/context or a general prompt. The predictions file is atomically republished after each completed example. A crash leaves the remaining IDs pending rather than inventing failures or answers.

Summary JSON reports actual completed row counts, failure counts for the successfully completed run, metric means, assistant loss/perplexity where applicable, and named runtime/memory measurements. A failed process has no completed summary and must be inspected/resumed; an absent summary is not zero failures. `semantic_accuracy` is null until independent review. No JSON NaN/Infinity values are allowed.

Human review artifacts leave correctness and error labels null/empty, explicitly distinguishing unknown from incorrect. Proposed reference corrections require a new adjudicated dataset version; outputs never overwrite weak references. Error-analysis pairs join by ID and verify the same population/content.

Training `metrics.csv` records optimizer-step, epoch fraction, train loss, optional validation loss, LR used, pre-clipping gradient norm, cumulative examples/assistant tokens and elapsed seconds. A blank validation cell means no validation ran at that step. Run summaries distinguish nominal batch size, unique training population and repeated exposures. Each checkpoint stores adapters plus trusted local optimizer/scheduler/RNG state, not frozen base weights.

`analysis/verify_artifacts.py` reloads saved JSONL, rejects duplicate/missing IDs, checks frozen hashes, recomputes every lexical score/aggregate and verifies complete final-test coverage. Plot scripts use only actual observations and label proxy metrics, units and controls. No unrun condition receives a zero-valued data point.

Tested package versions are pinned in `requirements.txt`; project minimums follow the exercised modern APIs rather than promising compatibility with older untested Transformer releases. Environment differences still require a fresh smoke test. Optional quantized dependencies are not part of the tested MPS environment.
