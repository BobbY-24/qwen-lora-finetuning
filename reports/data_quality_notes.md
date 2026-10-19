# Data quality and task validity

Training-only diagnostics: 1260/1412 rows have all-uppercase term strings; 16 usage sentences do not contain the normalized term as a literal substring. These are review flags, not exclusions. Inflections and variants can explain absence. The corpus mixes acronyms, older internet language, and current slang; its title should not be mistaken for demographic validation.

Descriptions are preserved as weak targets. For example, the training reference for PDA says “Personal display of affection”; a careful reviewer may dispute that expansion. We do not silently replace it or rewrite test references. Other records define multiple senses while the usage sentence picks one, so lexical agreement can reward an overly broad answer. The training examples for “Hot girl summer” and “Sturdy” include long cultural descriptions, while many acronym targets have only a few words; this affects loss weighting and output-style learning.

The dataset Context column often paraphrases the answer. It is excluded from all prompts. Example sentences could still contain subtle explanatory clues; a structural check cannot prove absence of all semantic leakage. Prefix/assistant masks are mechanically tested, but human contextual-reference review remains necessary.

## Observed training-term tokenization

```json
{
  "1 token": [
    {
      "term": "W",
      "token_ids": [
        54
      ],
      "pieces": [
        "W"
      ]
    },
    {
      "term": "L",
      "token_ids": [
        43
      ],
      "pieces": [
        "L"
      ]
    },
    {
      "term": "Stan",
      "token_ids": [
        50959
      ],
      "pieces": [
        "Stan"
      ]
    },
    {
      "term": "Drag",
      "token_ids": [
        17906
      ],
      "pieces": [
        "Drag"
      ]
    },
    {
      "term": "Bet",
      "token_ids": [
        56199
      ],
      "pieces": [
        "Bet"
      ]
    }
  ],
  "3+ tokens": [
    {
      "term": "L+ratio",
      "token_ids": [
        43,
        10,
        45358
      ],
      "pieces": [
        "L",
        "+",
        "ratio"
      ]
    },
    {
      "term": "Cheugy",
      "token_ids": [
        26843,
        768,
        88
      ],
      "pieces": [
        "Che",
        "ug",
        "y"
      ]
    },
    {
      "term": "Glow up",
      "token_ids": [
        38,
        10303,
        705
      ],
      "pieces": [
        "G",
        "low",
        "Ġup"
      ]
    },
    {
      "term": "Sip tea",
      "token_ids": [
        50,
        573,
        15243
      ],
      "pieces": [
        "S",
        "ip",
        "Ġtea"
      ]
    },
    {
      "term": "Iykyk",
      "token_ids": [
        40,
        88,
        7891,
        74
      ],
      "pieces": [
        "I",
        "y",
        "ky",
        "k"
      ]
    }
  ],
  "2 tokens": [
    {
      "term": "Dank",
      "token_ids": [
        35,
        1180
      ],
      "pieces": [
        "D",
        "ank"
      ]
    },
    {
      "term": "TFW",
      "token_ids": [
        10808,
        54
      ],
      "pieces": [
        "TF",
        "W"
      ]
    },
    {
      "term": "Woke",
      "token_ids": [
        54,
        4740
      ],
      "pieces": [
        "W",
        "oke"
      ]
    },
    {
      "term": "Bop",
      "token_ids": [
        33,
        453
      ],
      "pieces": [
        "B",
        "op"
      ]
    },
    {
      "term": "Smol",
      "token_ids": [
        10673,
        337
      ],
      "pieces": [
        "Sm",
        "ol"
      ]
    }
  ]
}
```

The actual token IDs and piece strings above show why vocabulary entries are not words. Tokenization length can influence the difficulty of unfamiliar terms even when the output target is short.
