"""Compare two prompt versions on a dataset and write a Markdown report.

Usage:
  python run_eval.py --a prompts/v1.txt --b prompts/v2.txt --cases datasets/leasing_support.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anthropic

from evalkit import compare_report, evaluate, load_cases, save_results, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="baseline system prompt file")
    p.add_argument("--b", required=True, help="candidate system prompt file")
    p.add_argument("--cases", required=True, help="JSONL dataset")
    p.add_argument("--out", default="results", help="output folder")
    args = p.parse_args()

    cases = load_cases(args.cases)
    client = anthropic.Anthropic()
    name_a, name_b = Path(args.a).stem, Path(args.b).stem
    res_a = evaluate(client, Path(args.a).read_text(encoding="utf-8"), cases)
    res_b = evaluate(client, Path(args.b).read_text(encoding="utf-8"), cases)

    out = Path(args.out)
    save_results(res_a, out / f"{name_a}.json")
    save_results(res_b, out / f"{name_b}.json")
    report = compare_report(name_a, res_a, name_b, res_b)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(name_a, summarize(res_a))
    print(name_b, summarize(res_b))


if __name__ == "__main__":
    main()
