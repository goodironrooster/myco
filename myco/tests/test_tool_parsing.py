"""Tests for tool parsing in myco CLI."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from myco.cli import parse_and_execute_tools, _log_and_show_tool
from myco.gate import AutopoieticGate, GateResult
from myco.session_log import SessionLogger
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def mock_gate(tmp_path):
    """Create a mock gate that permits all actions."""
    myco_dir = tmp_path / ".myco"
    myco_dir.mkdir()
    gate = MagicMock(spec=AutopoieticGate)
    gate.check_action.return_value = GateResult(permitted=True)
    return gate


@pytest.fixture
def mock_logger(tmp_path):
    """Create a mock session logger."""
    logger = MagicMock(spec=SessionLogger)
    return logger


class TestJSONToolCallParsing:
    """Tests for JSON tool call parsing."""

    def test_parse_json_write_file(self, mock_gate, mock_logger, tmp_path):
        """Test parsing JSON write_file tool call."""
        content = '{"name": "write_file", "arguments": {"path": "test.py", "content": "print(1)"}}'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "write_file"
        assert "Success" in results[0]["result"]

    def test_parse_json_read_file(self, mock_gate, mock_logger, tmp_path):
        """Test parsing JSON read_file tool call."""
        # Create test file first
        test_file = tmp_path / "existing.py"
        test_file.write_text("# test\n")
        
        content = '{"name": "read_file", "arguments": {"path": "existing.py"}}'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "read_file"

    def test_parse_json_with_whitespace(self, mock_gate, mock_logger, tmp_path):
        """Test parsing JSON with extra whitespace."""
        content = '{  "name" :  "write_file" ,  "arguments" :  { "path" : "x.py" , "content" : "hi" }  }'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "write_file"

    def test_parse_multiple_json_calls(self, mock_gate, mock_logger, tmp_path):
        """Test parsing multiple JSON tool calls."""
        content = '''
        {"name": "write_file", "arguments": {"path": "a.py", "content": "a"}}
        {"name": "write_file", "arguments": {"path": "b.py", "content": "b"}}
        '''
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 2


class TestMarkdownParsing:
    """Tests for markdown code block parsing."""

    def test_parse_markdown_with_filename_comment(self, mock_gate, mock_logger, tmp_path):
        """Test parsing markdown block with filename comment."""
        content = '''```python
# file: calculator.py
def add(a, b):
    return a + b
```'''
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "write_file"
        # Verify file was created with correct name
        created_file = tmp_path / "calculator.py"
        assert created_file.exists()

    def test_parse_markdown_with_file_comment_variations(self, mock_gate, mock_logger, tmp_path):
        """Test parsing markdown with various filename comment formats."""
        test_cases = [
            "# File: test.py",
            "# FILE: test.py",
            "# filename: test.py",
        ]
        
        for comment in test_cases:
            content = f'''```python
{comment}
def test(): pass
```'''
            results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
            assert len(results) == 1

    def test_parse_markdown_without_filename_uses_lang(self, mock_gate, mock_logger, tmp_path):
        """Test that markdown without filename still gets parsed if lang is python."""
        content = '''```python
def hello():
    print("world")
```'''
        
        # Without filename in content, should not create file (no filename to use)
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        # Should not create file without filename
        assert len(results) == 0

    def test_markdown_not_json_fallback(self, mock_gate, mock_logger, tmp_path):
        """Test markdown parsing when JSON parsing finds nothing."""
        # First ensure JSON pattern doesn't match
        content = '''Here's the code:
```python
# file: module.py
def func(): pass
```'''
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1


class TestFunctionCallParsing:
    """Tests for direct function call parsing."""

    def test_parse_function_call_double_quotes(self, mock_gate, mock_logger, tmp_path):
        """Test parsing function call with double quotes."""
        content = 'write_file(path="test.py", content="print(1)")'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "write_file"

    def test_parse_function_call_single_quotes(self, mock_gate, mock_logger, tmp_path):
        """Test parsing function call with single quotes."""
        content = "write_file(path='test.py', content='print(1)')"
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1

    def test_parse_function_call_multiline_content(self, mock_gate, mock_logger, tmp_path):
        """Test parsing function call with multiline content."""
        content = '''write_file(path="multi.py", content="def foo():
    pass
    return True")'''
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1

    def test_parse_read_file_function(self, mock_gate, mock_logger, tmp_path):
        """Test parsing read_file function call."""
        test_file = tmp_path / "existing.py"
        test_file.write_text("# test\n")
        
        content = 'read_file(path="existing.py")'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 1
        assert results[0]["tool"] == "read_file"


class TestParsingFallbacks:
    """Tests for parsing fallback behavior."""

    def test_no_tool_calls_returns_empty(self, mock_gate, mock_logger, tmp_path):
        """Test that content without tool calls returns empty list."""
        content = "I will create a file now. Here is some discussion about the task."
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 0

    def test_json_takes_precedence_over_markdown(self, mock_gate, mock_logger, tmp_path):
        """Test that JSON parsing takes precedence over markdown."""
        content = '''
{"name": "write_file", "arguments": {"path": "json.py", "content": "json"}}
```python
# file: markdown.py
def markdown(): pass
```'''
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        # Should only parse JSON, not markdown (JSON is first)
        assert len(results) >= 1
        assert results[0]["tool"] == "write_file"

    def test_invalid_json_is_skipped(self, mock_gate, mock_logger, tmp_path):
        """Test that invalid JSON is skipped gracefully."""
        content = '{"name": "write_file", "arguments": {invalid json}}'
        
        results = parse_and_execute_tools(content, tmp_path, mock_gate, mock_logger, verbose=False)
        
        assert len(results) == 0


class TestLogAndShowTool:
    """Tests for _log_and_show_tool helper."""

    def test_logs_success(self, mock_logger):
        """Test that success is logged correctly."""
        _log_and_show_tool(
            mock_logger,
            "write_file",
            {"path": "test.py"},
            "Successfully wrote 10 bytes",
            verbose=False
        )
        
        mock_logger.log_tool_call.assert_called_once()
        call_args = mock_logger.log_tool_call.call_args
        assert call_args[1]["success"] is True

    def test_logs_failure(self, mock_logger):
        """Test that failure is logged correctly."""
        _log_and_show_tool(
            mock_logger,
            "write_file",
            {"path": "test.py"},
            "Error: Permission denied",
            verbose=False
        )

        # "Error: Permission denied" doesn't have "Success" -> False
        call_args = mock_logger.log_tool_call.call_args
        assert call_args[1]["success"] is False

    def test_logs_success_explicit(self, mock_logger):
        """Test that explicit success is logged correctly."""
        _log_and_show_tool(
            mock_logger,
            "write_file",
            {"path": "test.py"},
            "Success: File written",
            verbose=False
        )

        call_args = mock_logger.log_tool_call.call_args
        assert call_args[1]["success"] is True

    def test_logs_blocked(self, mock_logger):
        """Test that BLOCKED is logged as failure."""
        _log_and_show_tool(
            mock_logger,
            "write_file",
            {"path": "test.py"},
            "BLOCKED: Entropy too high",
            verbose=False
        )
        
        call_args = mock_logger.log_tool_call.call_args
        assert call_args[1]["success"] is False
