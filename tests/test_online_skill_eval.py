import pytest

from agents import online_skill_eval as ev


def rule_ids(instructions, **kwargs):
    skill = {"name": "demo", "description": "", "instructions": instructions}
    return {rule["rule_id"] for rule in ev._compile_eval_rules(skill, **kwargs)}


@pytest.mark.parametrize(
    ("instructions", "expected"),
    [
        ("Always cite sources for factual claims.", "must_cite_sources"),
        ("回答时请引用来源。", "must_cite_sources"),
        ("Give the conclusion first, then details.", "lead_with_conclusion"),
        ("先给结论，再展开。", "lead_with_conclusion"),
        ("Return the result as JSON.", "json_parseable"),
        ("请使用结构化输出。", "json_parseable"),
        ("Present results in a markdown table.", "markdown_table"),
        ("用表格展示对比。", "markdown_table"),
        ("Do not fabricate numbers.", "uncertainty_marked"),
        ("不要编造数据。", "uncertainty_marked"),
    ],
)
def test_rules_compile_from_english_and_chinese_instructions(instructions, expected):
    assert expected in rule_ids(instructions)


@pytest.mark.parametrize("instructions", ["Answer in no more than 2 paragraphs.", "回答不超过 2 段。"])
def test_paragraph_limit_is_parsed_in_both_languages(instructions):
    skill = {"name": "demo", "instructions": instructions}
    (rule,) = [r for r in ev._compile_eval_rules(skill) if r["rule_id"] == "paragraph_limit"]
    assert rule["params"]["max_paragraphs"] == 2


def test_llm_rules_are_opt_in():
    assert "no_unfounded_claims" not in rule_ids("Do not hallucinate.")
    assert "no_unfounded_claims" in rule_ids("Do not hallucinate.", include_llm_rules=True)


def evaluate(mode, text, **params):
    return ev._evaluate_rule({"rule_id": mode, "params": {"mode": mode, **params}}, text)["passed"]


def test_programmatic_rule_checks():
    assert evaluate("json_parseable", '{"ok": true}')
    assert not evaluate("json_parseable", "ok: true")
    assert evaluate("mentions_sources", "See https://example.com")
    assert evaluate("mentions_sources", "来源：官方文档")
    assert not evaluate("mentions_sources", "Trust me.")
    assert evaluate("max_paragraphs", "one\n\ntwo", max_paragraphs=2)
    assert not evaluate("max_paragraphs", "one\n\ntwo\n\nthree", max_paragraphs=2)
    assert evaluate("markdown_table", "| a | b |\n|---|---|\n| 1 | 2 |")
    assert evaluate("lead_with_conclusion", "结论：可以上线。\n\n" + "细节 " * 80)
    assert not evaluate("lead_with_conclusion", "Some long preamble " * 20)


def test_promotion_requires_beating_the_champion():
    candidate = {"average_score": 0.9, "hard_failures": 0}

    first = ev._promotion_decision(status="healthy", candidate=candidate, champion={})
    assert first["promoted"] and first["reason"] == "first healthy candidate for this lineage"

    champion = {"summary": {"average_score": 0.85, "hard_failures": 0}}
    assert ev._promotion_decision(status="healthy", candidate=candidate, champion=champion)["promoted"]

    worse = {"average_score": 0.95, "hard_failures": 2}
    assert not ev._promotion_decision(status="healthy", candidate=worse, champion=champion)["promoted"]

    for status in ("unobserved", "incubating", "pruned", "watch"):
        assert not ev._promotion_decision(status=status, candidate=candidate, champion={})["promoted"]


def test_evaluation_runs_on_an_empty_project(isolated_env):
    report = ev.evaluate_online_skill_evolution(write_report=False, write_artifacts=False)
    assert isinstance(report, dict)
    assert isinstance(ev.format_online_skill_eval(report), str)
