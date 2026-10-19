"""Selected assistant case review. Never fills independent human-label columns."""
from pathlib import Path
import sys
sys.path.insert(0,'src')
from community_qwen.utils.io import read_jsonl,write_jsonl
b={r['id']:r for r in read_jsonl('experiments/baseline/test/predictions.jsonl')}
t={r['id']:r for r in read_jsonl('experiments/lora/run_001/test/predictions.jsonl')}
# Grades: 0 wrong, 1 partial, 2 contextually adequate, None unscorable reference.
annotations=[
('hf-00130-d8b7b62132',0,2,'improvement',['wrong_sense','hallucinated_definition'],'The base invents initials/nickname and regret; LoRA correctly recognizes the joking reversal.'),
('hf-00254-fb0ddda9d8',0,2,'improvement',['hallucinated_definition'],'First Class Experience does not fit the frequency slot; Once does.'),
('hf-01344-131a537650',0,2,'improvement',['wrong_sense'],'A messaging-app interpretation misses the question marker; Quick question fits the utterance.'),
('hf-01626-66fe3ac1e1',0,2,'improvement',['wrong_sense'],'Victory Point misses praise for the opponent; Very well played fits.'),
('hf-00008-2587f51157',0,0,'both_fail',['hallucinated_definition'],'Both expansions are wrong. LoRA shares several words with the reference despite inventing Good old man.'),
('hf-00018-e2dd8db6da',0,0,'both_fail',['context_ignored','wrong_sense'],'The base treats this as financial rent; LoRA says never become part of memory, contradicting the intrusive-memory sense.'),
('hf-00093-1829a56b9c',0,0,'both_fail',['context_ignored','hallucinated_definition'],'Neither an employee reassignment nor Korean explains the manager-demanding stereotype.'),
('hf-00074-77daff0953',0,0,'both_fail',['wrong_sense','overly_generic'],'Combat is wrong here. Merely wearing a costume also misses doing/looking exceptionally well.'),
('hf-00026-6a3c249c2b',1,0,'partial_to_wrong',['overly_generic','wrong_sense'],'The base captures a fashion aspect but misses internet subculture; Elegant is not an explanation of e-girl.'),
('hf-00272-29d14498bf',1,0,'partial_to_wrong',['hallucinated_definition'],'The base captures adult-only content but invents the expansion and platform claim; Above average loses the meaning entirely.'),
('hf-00331-448fc3b8e3',2,2,'metric_artifact',['correct_paraphrase'],'The base correctly describes a forum administrator but receives zero lexical F1 because the target is the single word Administrator. LoRA exact-matches it. A +1 lexical jump is not a semantic improvement here.'),
('hf-00104-ed5feb1667',2,2,'metric_artifact',['correct_paraphrase'],'To tell the truth is an adequate contextual paraphrase of To be honest, despite a lower overlap score than the longer base response.'),
('hf-00177-e9b63402ae',2,1,'loss_of_specificity',['overly_generic'],'The base explicitly marks a subjective opinion. I think so conveys subjectivity but is a weaker substitute and not the acronym expansion; do not force this into an unequivocal correct-to-wrong label.'),
('hf-00090-acd25f533a',None,None,'reference_review',['reference_context_mismatch'],'The supplied term is I can’t even, but the usage is Ion even care. This needs reference adjudication; do not silently repair the gold or count it as a clean semantic test.'),
]
rows=[]
text='# Selected qualitative cases\n\nThese 14 cases were selected after viewing fixed predictions to illustrate failure mechanisms and metric limitations. Labels are **assistant qualitative judgments**, not independent human gold, and must not be aggregated into a semantic-accuracy estimate. A human reviewer should validate them blind to model identity.\n\nGrades: 0 wrong, 1 partial, 2 contextually adequate, null unscorable reference.\n'
for key,bg,tg,category,tags,note in annotations:
    r={'id':key,'term':b[key]['term'],'context':b[key]['context'],'reference':b[key]['definition'],
       'base':b[key]['prediction'],'lora':t[key]['prediction'],'assistant_base_grade':bg,'assistant_lora_grade':tg,
       'assistant_category':category,'assistant_tags':tags,'assistant_notes':note,'reviewer':'assistant; non-independent, selected qualitative sample',
       'independent_human_label':None}
    rows.append(r)
    text+=f'\n## {r["term"]}: {category}\n\nID: `{key}`. Usage: {r["context"]}\n\nReference: {r["reference"]}\n\nBase: {r["base"]}\n\nLoRA: {r["lora"]}\n\nJudgment ({bg} → {tg}): {note}\n'
text+='\n## Unassigned claims\n\nNo unambiguous fully-correct-to-wrong slang example was assigned in this small selected review; partial-to-wrong and loss-of-specificity cases are labeled separately. No memorization or excessive-slang-style claim is established. Term disjointness does not rule out pretraining familiarity. Human annotation of the full or a prespecified random sample is still needed.\n'
write_jsonl('reports/assistant_case_review.jsonl',rows)
Path('reports/qualitative_cases.md').write_text(text)
