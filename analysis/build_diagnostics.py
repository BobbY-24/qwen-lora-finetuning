from pathlib import Path
import json, hashlib, re

def dump(path,rows):
 Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

ood=[
('detached HEAD','I checked out the old commit and now I am in detached HEAD.','Git HEAD points directly to a commit rather than to a branch.','https://git-scm.com/docs/git-checkout'),
('cherry-pick','Please cherry-pick the bug fix onto the release branch.','Apply the changes introduced by a selected existing commit to the current branch.','https://git-scm.com/docs/git-cherry-pick'),
('fast-forward','Git merged the branch with a fast-forward.','Move the branch pointer forward to the descendant commit without creating a merge commit.','https://git-scm.com/docs/git-merge'),
('rebase','Rebase your feature branch onto the current main branch.','Replay the branch commits on a new base commit.','https://git-scm.com/docs/git-rebase'),
('CrashLoopBackOff','The pod keeps entering CrashLoopBackOff after startup.','A container repeatedly fails and Kubernetes delays its next restart attempt.','https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/'),
('sidecar','We added a sidecar to collect the application logs.','A supporting container runs alongside the main application container in a pod.','https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/'),
('taint','The node has a taint so ordinary workloads do not land there.','A node property repels pods unless their tolerations match.','https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/'),
('backpressure','The slow consumer applies backpressure to the stream.','The consumer signals that it cannot keep up, limiting how quickly data is produced.','https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Concepts'),
('closure','The callback remembers count because it forms a closure.','A function retains access to variables in the lexical environment where it was created.','https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Closures'),
('hoisting','The declaration behaves this way because of hoisting.','JavaScript makes certain declarations available before their textual position during execution setup.','https://developer.mozilla.org/en-US/docs/Glossary/Hoisting'),
('idempotent','Retrying this request is safe because the operation is idempotent.','Repeating the same operation has the same intended effect as performing it once.','https://developer.mozilla.org/en-US/docs/Glossary/Idempotent'),
('specificity','The class rule loses because the other selector has higher specificity.','CSS ranks selector weights to decide which competing style declaration applies.','https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_cascade/Specificity'),
]
allterms={r['term_key'] for f in Path('data/processed').glob('*.jsonl') if f.stem in ['train','validation','test'] for r in map(json.loads,f.read_text().splitlines())}
rows=[];excluded=[]
for i,(term,ctx,definition,source) in enumerate(ood):
 key=''.join(c for c in term.casefold() if c.isalnum())
 if key in allterms:excluded.append(term);continue
 rows.append(dict(id=f'ood-{i:02}',term=term,term_key=key,context=ctx,definition=definition,source=source,
                  provenance='Assistant-authored evaluation example and reference; independent human review pending',domain='software technical community'))
dump('data/ood/technical.jsonl',rows)
Path('data/ood/provenance.json').write_text(json.dumps({'excluded_overlap_terms':excluded,'count':len(rows),'trained_on':False,'status':'authored diagnostic set, not external benchmark'},indent=2))
general=[
('reasoning','What is 17 + 28? Answer with only the number.','45'),
('reasoning','A box has 3 red balls and 2 blue balls. How many balls are there? Answer with only the number.','5'),
('reasoning','All roses are plants. Some plants grow indoors. Does it follow that all roses grow indoors? Answer yes or no.','No'),
('reasoning','Mia is older than Leo. Leo is older than Sam. Who is youngest? Answer with a name only.','Sam'),
('reasoning','A train travels 60 kilometers in 2 hours. What is its average speed in kilometers per hour? Answer with only the number.','30'),
('factual','What is the chemical symbol for oxygen? Answer with the symbol only.','O'),
('factual','What is the capital of France? Answer with the city only.','Paris'),
('factual','Which planet is known as the Red Planet? Answer with its name only.','Mars'),
('factual','How many sides does a triangle have? Answer with only the number.','3'),
('factual','At standard atmospheric pressure, at what Celsius temperature does pure water freeze? Answer with only the number.','0'),
('summarization','Summarize in one sentence: The library will close Monday for repairs. It will reopen Tuesday at 9 am. Online borrowing remains available.','The library closes Monday for repairs and reopens Tuesday at 9 am, while online borrowing remains available.'),
('summarization','Summarize in one sentence: Lena missed the bus. She walked to work. She arrived ten minutes late.','Lena walked to work after missing the bus and arrived ten minutes late.'),
('summarization','Summarize in one sentence: The garden received no rain for a week. Volunteers watered the plants. The plants survived.','Volunteers watered the garden during a dry week, keeping the plants alive.'),
('summarization','Summarize in one sentence: The software update fixes a login bug. It adds no new features. Users must restart the app.','The update fixes a login bug without adding features and requires restarting the app.'),
('summarization','Summarize in one sentence: The team tested two batteries. Battery A lasted six hours. Battery B lasted nine hours.','Battery B lasted nine hours in testing, compared with six hours for battery A.'),
('instruction','Return these words in alphabetical order, separated by commas: pear, apple, banana.','apple, banana, pear'),
('instruction','Return exactly this text and nothing else: CHECK PASSED','CHECK PASSED'),
('instruction','Convert hello world to uppercase. Output only the converted text.','HELLO WORLD'),
('instruction','Extract only the email address: Contact Ana at ana@example.com tomorrow.','ana@example.com'),
('instruction','Output a JSON object with a single key count whose value is the integer 3.','{"count": 3}'),
('coding','What does Python print for len([1, 2, 3])? Answer with only the number.','3'),
('coding','What does Python print for sorted([3, 1, 2])? Output only the list.','[1, 2, 3]'),
('coding','What is the value of 7 % 3 in Python? Answer with only the number.','1'),
('coding','What does Python print for "hello".upper()? Output only the resulting text.','HELLO'),
('coding','In Python, what is list(range(3))? Output only the resulting list.','[0, 1, 2]'),
]
dump('data/ood/general.jsonl',[dict(id=f'general-{i:02}',category=cat,prompt=p,definition=gold,provenance='Assistant-authored sanity prompt; not a benchmark') for i,(cat,p,gold) in enumerate(general)])
