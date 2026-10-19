from pathlib import Path
import sys
sys.path.insert(0,'src')
from community_qwen.utils.io import read_jsonl,write_json
from community_qwen.evaluation.regression import check_general,summarize_general
b={r['id']:r for r in read_jsonl('experiments/baseline/general/predictions.jsonl')}
t={r['id']:r for r in read_jsonl('experiments/lora/run_001/general/predictions.jsonl')}
assert set(b)==set(t)
objective=[k for k in b if check_general(b[k]) is not None]
changed=[{'id':k,'prompt':b[k]['prompt'],'reference':b[k]['definition'],'base':b[k]['prediction'],'lora':t[k]['prediction'],
          'base_strict_correct':check_general(b[k]),'lora_strict_correct':check_general(t[k])}
         for k in objective if check_general(b[k])!=check_general(t[k])]
r={'objective_prompts':len(objective),'base_strict_correct':sum(check_general(b[k]) for k in objective),
   'lora_strict_correct':sum(check_general(t[k]) for k in objective),'improvements':sum(not x['base_strict_correct'] and x['lora_strict_correct'] for x in changed),
   'regressions':sum(x['base_strict_correct'] and not x['lora_strict_correct'] for x in changed),'changed_cases':changed}
write_json('reports/general_findings.json',r)
text=f'''# General capability interpretation

The objective short-answer suite improved from **{r['base_strict_correct']}/{r['objective_prompts']}** to **{r['lora_strict_correct']}/{r['objective_prompts']}** under the fixed strict evaluator: {r['improvements']} newly passing answers and {r['regressions']} newly failing answer. This small suite does not show a general collapse, but it cannot rule out broader degradation. Some improvements are formatting compliance (for example, returning only a sorted list), and others correct the answer itself. These are authored sanity prompts, not a benchmark.

A definite instruction-following regression is `general-16`: the prompt requires exactly `CHECK PASSED`. The base returns that exact text; LoRA returns `Check passed`, violating the explicit case-sensitive copy constraint.

The five summaries remain qualitative. Assistant inspection finds that LoRA's library summary omits the reopening time and online borrowing; its battery summary says “more efficient,” which is not established by battery duration alone. The original reference information is preserved. Do not let the 20-item strict aggregate conceal these losses of detail.

The OOD set is also mixed: higher ROUGE-L accompanies lower reference-content coverage. Specific technical errors remain (for example, the fine-tuned hoisting answer discusses C++ public/private access, and its sidecar answer describes storage rather than a supporting container). No broad OOD or general-capability improvement is established.
'''
Path('reports/general_findings.md').write_text(text)
