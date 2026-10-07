from agents.frontmatter import format_frontmatter, parse_frontmatter


def test_parses_meta_and_body():
    result = parse_frontmatter("---\nname: demo\ntype: user\n---\n\nHello world")
    assert result.meta == {"name": "demo", "type": "user"}
    assert result.body == "Hello world"


def test_value_may_contain_colons():
    result = parse_frontmatter("---\nurl: https://example.com:8080\n---\nbody")
    assert result.meta["url"] == "https://example.com:8080"


def test_missing_frontmatter_returns_whole_content_as_body():
    result = parse_frontmatter("just text")
    assert result.meta == {}
    assert result.body == "just text"


def test_unclosed_frontmatter_is_treated_as_body():
    content = "---\nname: demo\nno closing delimiter"
    result = parse_frontmatter(content)
    assert result.meta == {}
    assert result.body == content


def test_format_round_trips():
    meta = {"name": "demo", "description": "a test"}
    result = parse_frontmatter(format_frontmatter(meta, "Body text"))
    assert result.meta == meta
    assert result.body == "Body text"
