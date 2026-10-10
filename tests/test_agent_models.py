from agents import agent


def test_known_and_unknown_context_windows():
    assert agent._get_context_windows("gpt-4o") == 128000
    assert agent._get_context_windows("some-new-model") == 200000


def test_max_output_tokens_by_model_family():
    assert agent._get_max_output_tokens("claude-opus-4-6") == 64000
    assert agent._get_max_output_tokens("claude-sonnet-4-6") == 32000
    assert agent._get_max_output_tokens("claude-haiku-4-5-20251001") == 32000
    assert agent._get_max_output_tokens("deepseek-chat") == 16384


def test_tools_convert_to_openai_function_format():
    tools = [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}]
    assert agent._to_openai_tools(tools) == [
        {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}}
    ]


def test_sanitize_replaces_invalid_surrogates_recursively():
    value = {"text": "ok\ud800", "items": ["a\udfff", ("b",)]}
    cleaned = agent._sanitize_for_utf8(value)
    cleaned_text = repr(cleaned)
    assert "\\ud800" not in cleaned_text and "\\udfff" not in cleaned_text
    assert cleaned["items"][1] == ("b",)
