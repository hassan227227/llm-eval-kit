from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from evalkit import Case, Result, Verdict, compare_report, evaluate, load_cases, run_checks, summarize


class FakeMessages:
    """Stands in for client.messages: canned answers and judge scores."""

    def __init__(self, answers: dict[str, str], scores: dict[str, int]):
        self.answers, self.scores = answers, scores

    def create(self, **kw):
        q = kw["messages"][0]["content"]
        return NS(stop_reason="end_turn", content=[NS(type="text", text=self.answers[q])],
                  usage=NS(input_tokens=10, output_tokens=5))

    def parse(self, **kw):
        content = kw["messages"][0]["content"]
        q = content.split("Question:\n", 1)[1].split("\n\n", 1)[0]
        return NS(parsed_output=Verdict(score=self.scores[q], reason="ok"))


def client(answers, scores):
    return NS(messages=FakeMessages(answers, scores))


CASES = [
    Case("c1", "How long are leases?", "24, 36 or 48 months.", must_include=["36"]),
    Case("c2", "Can I end early?", "After 12 months, fee is 3 payments.", must_include=["12"], must_not_include=["anytime"]),
]


def test_load_sample_dataset():
    cases = load_cases(Path(__file__).parent / "datasets" / "leasing_support.jsonl")
    assert len(cases) >= 6 and len({c.id for c in cases}) == len(cases)


def test_load_rejects_duplicates(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"id":"a","input":"q","reference":"r"}\n{"id":"a","input":"q","reference":"r"}\n')
    with pytest.raises(ValueError):
        load_cases(f)


def test_run_checks_case_insensitive():
    assert run_checks("You can end it ANYTIME", CASES[1]) == ["missing: 12", "forbidden: anytime"]
    assert run_checks("after 12 months", CASES[1]) == []


def test_evaluate_combines_checks_and_judge():
    c = client({"How long are leases?": "36 months", "Can I end early?": "anytime"},
               {"How long are leases?": 5, "Can I end early?": 2})
    res = evaluate(c, "system", CASES)
    assert [r.passed for r in res] == [True, False]
    assert res[1].failed_checks == ["missing: 12", "forbidden: anytime"]
    assert summarize(res) == {"cases": 2, "pass_rate": 0.5, "mean_score": 3.5, "tokens": 30}


def test_judge_score_below_threshold_fails_even_if_checks_pass():
    r = Result("x", "36", True, [], 3, "partial", 1, 1)
    assert not r.passed


def test_compare_report_lists_fixed_cases():
    a = [Result("c1", "", True, [], 5, "", 1, 1), Result("c2", "", False, ["m"], 2, "", 1, 1)]
    b = [Result("c1", "", True, [], 5, "", 1, 1), Result("c2", "", True, [], 5, "now correct", 1, 1)]
    report = compare_report("v1", a, "v2", b)
    assert "| Pass rate | 50% | 100% |" in report
    assert "**c2** fixed" in report
