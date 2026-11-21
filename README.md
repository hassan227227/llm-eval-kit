# LLM Eval Kit

![tests](https://github.com/hassan227227/llm-eval-kit/actions/workflows/ci.yml/badge.svg)

A small, dependable harness for answering the question every LLM team asks: **did this prompt change make the assistant better or worse?** It runs two prompt versions over a dataset, grades every answer, and writes a comparison report.

```
dataset.jsonl ──► prompt v1 ──► answers ──┐
              └─► prompt v2 ──► answers ──┤
                                          ▼
                    deterministic checks (must / must-not include)
                    + LLM judge (structured verdict vs reference, 1-5)
                                          ▼
                     report.md: pass rate · mean score · tokens · cases that changed
```

## How grading works

- **Deterministic checks** catch hard requirements cheaply: an answer about lease length must contain "36"; an answer about approval must never say "guarantee".
- **LLM-as-judge** compares the answer with a reference answer and returns a validated `Verdict(score, reason)` through structured outputs, so the judge can't return malformed JSON.
- **A case passes** only if both agree: all checks pass and the judge scores 4 or 5.
- **The report** shows pass rate, mean score and token usage side by side, then lists exactly which cases were fixed or regressed, with the judge's reason.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...         # Windows: set ANTHROPIC_API_KEY=...

python run_eval.py --a prompts/v1.txt --b prompts/v2.txt --cases datasets/leasing_support.jsonl
cat results/report.md

pytest -q                            # offline: the model client is faked in tests
```

The sample dataset is 8 customer-support questions for a fictional leasing company, including an out-of-scope question the assistant should decline. `v1` is a one-line prompt; `v2` grounds the assistant in the policy and sets scope and length.

## Dataset format

One JSON object per line:

```json
{"id": "terms", "input": "What lease lengths do you offer?", "reference": "24, 36 or 48 months.", "must_include": ["24", "36", "48"], "must_not_include": []}
```

## Files

| File | What it does |
|---|---|
| `evalkit.py` | Dataset loading, checks, generation, judge, summary and report |
| `run_eval.py` | CLI to compare two prompt files |
| `test_evalkit.py` | Offline tests with a fake model client |
| `.github/workflows/ci.yml` | Runs the tests on every push |

## Next steps

- Run cases concurrently and cache answers by prompt hash
- Repeat each case N times to measure variance
- Use it to evaluate [docqa-agent](https://github.com/hassan227227/docqa-agent) and [loan-agent](https://github.com/hassan227227/loan-agent)

## Author

Hassan Raza · [Portfolio](https://hassan227227.github.io) · [LinkedIn](https://www.linkedin.com/in/iamhassan890)
