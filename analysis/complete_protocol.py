"""Complete the already predeclared evaluation protocol; no tuning or new conditions."""
from pathlib import Path
import json,subprocess,sys

for name in ['one_epoch_rank8','one_epoch_rank4','one_epoch_quarter','one_epoch_seed123']:
    summary=json.loads((Path('experiments/ablations')/name/'summary.json').read_text())
    if summary['status']!='completed':raise RuntimeError(f'Incomplete supplemental training: {name}')
for script in ['scripts/final_evaluations.py','analysis/evaluate_supplemental.py','scripts/build_reports.py',
               'analysis/plot_supplemental.py','analysis/stratify_results.py','analysis/verify_artifacts.py',
               'analysis/run_inventory.py','analysis/write_final_report.py']:
    print(f'Running {script}',flush=True)
    subprocess.run([sys.executable,script],check=True)
