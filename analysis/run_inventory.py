from pathlib import Path
from datetime import datetime,timezone
import json,sys
sys.path.insert(0,'src')
from community_qwen.utils.io import write_json
runs=[]
for p in sorted(Path('experiments').rglob('summary.json')):
    s=json.loads(p.read_text());kind='training' if 'trainable_parameters' in s else 'evaluation'
    env_path=p.parent/'environment.json';env=json.loads(env_path.read_text()) if env_path.exists() else {}
    runs.append({'run_id':str(p.parent),'kind':kind,'status':s.get('status','completed'),
                 'summary_file_modified_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
                 'source_sha256':env.get('source_sha256'),'runtime_seconds':s.get('runtime_seconds'),
                 'rows':s.get('rows',s.get('train_rows')),'paid_api_cost_usd':0,
                 'cost_basis':'All generation/training used local model weights; no paid API calls.'})
write_json('reports/run_inventory.json',{'runs':runs,'timestamp_basis':'Actual summary-file modification time, not a reconstructed exact launch timestamp.'})
print('Indexed',len(runs),'completed run summaries.')
