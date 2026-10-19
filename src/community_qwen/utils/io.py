"""Small, explicit provenance and serialization helpers."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    temp.replace(path)


def code_identity():
    files = {str(p): p.read_text() for base in ['src', 'scripts'] for p in sorted(Path(base).rglob('*.py'))}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'git_commit': commit.stdout.strip() or None, 'source_sha256': digest(files)}
