"""A small, dependable eval harness for LLM prompts.

1. Run a prompt version over a dataset of cases.
2. Grade every answer two ways: deterministic checks (must / must-not include)
   and an LLM judge that returns a structured verdict against the reference.
3. Compare prompt versions in a Markdown report: pass rate, mean score,
   tokens, and the cases that changed.

The model client is injected, so everything except the API call is unit-tested offline.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"
JUDGE_MODEL = "claude-opus-5-5"
PASS_SCORE = 4

JUDGE_SYSTEM = (
    "You grade an assistant's answer against a reference answer. "
    "Score 5 = fully correct and complete; 4 = correct with minor omissions; "
    "3 = partly correct; 2 = mostly wrong; 1 = wrong or harmful. "
    "Judge facts, not style. Give a one-sentence reason."
)


class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


@dataclass
class Case:
    id: str
    input: str
    reference: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


@dataclass
class Result:
    case_id: str
    answer: str
    checks_passed: bool
    failed_checks: list[str]
    judge_score: int
    judge_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def passed(self) -> bool:
        return self.checks_passed and self.judge_score >= PASS_SCORE


class Client(Protocol):  # the parts of anthropic.Anthropic we use
    messages: Any


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(Case(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"{path}:{n}: invalid case ({e})") from e
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def run_checks(answer: str, case: Case) -> list[str]:
    """Return the names of failed deterministic checks (case-insensitive)."""
    low = answer.lower()
    failed = [f"missing: {s}" for s in case.must_include if s.lower() not in low]
    failed += [f"forbidden: {s}" for s in case.must_not_include if s.lower() in low]
    return failed


def generate(client: Client, system_prompt: str, case: Case) -> tuple[str, int, int]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system_prompt,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": case.input}],
    )
    if resp.stop_reason == "refusal":
        return "[refused]", resp.usage.input_tokens, resp.usage.output_tokens
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def judge(client: Client, case: Case, answer: str) -> Verdict:
    resp = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=JUDGE_SYSTEM,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": f"Question:\n{case.input}\n\nReference answer:\n{case.reference}\n\nAssistant answer:\n{answer}",
        }],
        output_format=Verdict,
    )
    if resp.parsed_output is None:
        return Verdict(score=1, reason="Judge returned no verdict.")
    return resp.parsed_output


def evaluate(client: Client, system_prompt: str, cases: list[Case]) -> list[Result]:
    results = []
    for case in cases:
        answer, tin, tout = generate(client, system_prompt, case)
        failed = run_checks(answer, case)
        verdict = judge(client, case, answer)
        results.append(Result(case.id, answer, not failed, failed, verdict.score, verdict.reason, tin, tout))
    return results


def summarize(results: list[Result]) -> dict:
    if not results:
        return {"cases": 0, "pass_rate": 0.0, "mean_score": 0.0, "tokens": 0}
    return {
        "cases": len(results),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3),
        "mean_score": round(statistics.mean(r.judge_score for r in results), 2),
        "tokens": sum(r.input_tokens + r.output_tokens for r in results),
    }


def compare_report(name_a: str, a: list[Result], name_b: str, b: list[Result]) -> str:
    sa, sb = summarize(a), summarize(b)
    lines = [
        f"# Eval report: {name_a} vs {name_b}",
        "",
        "| Metric | " + name_a + " | " + name_b + " |",
        "|---|---|---|",
        f"| Pass rate | {sa['pass_rate']:.0%} | {sb['pass_rate']:.0%} |",
        f"| Mean judge score (1-5) | {sa['mean_score']} | {sb['mean_score']} |",
        f"| Tokens | {sa['tokens']} | {sb['tokens']} |",
        "",
    ]
    by_id = {r.case_id: r for r in b}
    changed = [(r, by_id[r.case_id]) for r in a if r.case_id in by_id and r.passed != by_id[r.case_id].passed]
    lines.append("## Cases that changed")
    if not changed:
        lines.append("None.")
    for ra, rb in changed:
        arrow = "fixed" if rb.passed else "regressed"
        lines.append(f"- **{ra.case_id}** {arrow}: {name_a} {ra.judge_score}/5, {name_b} {rb.judge_score}/5. {rb.judge_reason}")
    return "\n".join(lines) + "\n"


def save_results(results: list[Result], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
