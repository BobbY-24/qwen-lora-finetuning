# General capability sanity check

Strict output matching on 20 short-answer prompts; five summaries require review. No generated code is executed.

```json
{
  "Base": {
    "reasoning": {
      "correct": 1,
      "total": 5,
      "strict_accuracy": 0.2
    },
    "factual": {
      "correct": 3,
      "total": 5,
      "strict_accuracy": 0.6
    },
    "instruction": {
      "correct": 3,
      "total": 5,
      "strict_accuracy": 0.6
    },
    "coding": {
      "correct": 0,
      "total": 5,
      "strict_accuracy": 0.0
    }
  },
  "LoRA": {
    "reasoning": {
      "correct": 4,
      "total": 5,
      "strict_accuracy": 0.8
    },
    "factual": {
      "correct": 5,
      "total": 5,
      "strict_accuracy": 1.0
    },
    "instruction": {
      "correct": 3,
      "total": 5,
      "strict_accuracy": 0.6
    },
    "coding": {
      "correct": 3,
      "total": 5,
      "strict_accuracy": 0.6
    }
  }
}
```
