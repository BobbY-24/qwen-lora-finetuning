# Analysis entrypoints

Run from the repository root using `.venv/bin/python analysis/FILE.py`. Model execution is serial and uses the same explicit training/evaluation implementations in `src` and `scripts`. `run_supplemental.py` launches the four predeclared one-epoch training configurations; `evaluate_supplemental.py` generates their fixed test outputs.

`plot_supplemental.py` uses only matched one-epoch runs for the rank and data curves. It also reports two-seed mean/sample standard deviation. `verify_artifacts.py` reloads predictions, checks complete ID joins, recomputes metrics, and checks frozen split hashes. `write_final_report.py` derives the report from saved measurements. These reporting scripts do not alter model prompts or training code.

`build_diagnostics.py` preserves the authored OOD/general set construction and filters OOD term overlap. Do not regenerate or change these frozen inputs after inspecting final results without assigning a new evaluation version.

Once all predeclared training summaries exist, `complete_protocol.py` runs the final evaluation, supplemental evaluation and derived-report steps in order, stopping on any failure. It adds no new model condition and does no hyperparameter selection. Re-running it may resume existing evaluation only when source/config/input fingerprints match.
