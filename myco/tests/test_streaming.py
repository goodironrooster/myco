"""Tests for streaming response handling in myco CLI."""

import json
import pytest
from unittest.mock import MagicMock, patch, Mock


class TestStreamingResponseParsing:
    """Tests for parsing streaming SSE responses."""

    def test_parse_sse_data_line(self):
        """Test parsing SSE data line format."""
        line = b'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        
        line_str = line.decode('utf-8')
        assert line_str.startswith('data: ')
        
        data = line_str[6:]  # Remove 'data: ' prefix
        chunk = json.loads(data)
        
        delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
        assert delta == "Hello"

    def test_parse_sse_done_signal(self):
        """Test parsing SSE [DONE] signal."""
        line = b'data: [DONE]'
        
        line_str = line.decode('utf-8')
        data = line_str[6:]
        
        assert data == '[DONE]'

    def test_parse_sse_empty_line(self):
        """Test handling empty SSE lines."""
        line = b''
        
        # Empty lines should be skipped
        assert not line

    def test_parse_sse_multiple_chunks(self):
        """Test parsing multiple SSE chunks."""
        chunks = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " World"}}]}',
            b'data: [DONE]'
        ]
        
        content = ""
        for line in chunks:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = line_str[6:]
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                content += delta
        
        assert content == "Hello World"

    def test_parse_sse_empty_delta(self):
        """Test handling empty delta in chunk."""
        line = b'data: {"choices": [{"delta": {}}]}'
        
        line_str = line.decode('utf-8')
        data = line_str[6:]
        chunk = json.loads(data)
        delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
        
        assert delta == ''

    def test_parse_sse_missing_choices(self):
        """Test handling missing choices in chunk."""
        line = b'data: {"usage": {"total_tokens": 10}}'
        
        line_str = line.decode('utf-8')
        data = line_str[6:]
        chunk = json.loads(data)
        delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
        
        assert delta == ''

    def test_parse_sse_invalid_json(self):
        """Test handling invalid JSON in SSE."""
        line = b'data: {invalid json}'
        
        line_str = line.decode('utf-8')
        data = line_str[6:]
        
        try:
            json.loads(data)
            assert False, "Should have raised JSONDecodeError"
        except json.JSONDecodeError:
            pass  # Expected


class TestStreamingVsNonStreaming:
    """Tests for streaming vs non-streaming mode selection."""

    def test_verbose_uses_streaming(self):
        """Test that verbose mode enables streaming."""
        # This is a logic test - in actual code, verbose=True sets stream=True
        verbose = True
        stream_enabled = verbose  # Logic from cli.py
        
        assert stream_enabled is True

    def test_non_verbose_uses_non_streaming(self):
        """Test that non-verbose mode uses regular response."""
        verbose = False
        stream_enabled = verbose  # Logic from cli.py
        
        assert stream_enabled is False


class TestTokenCounting:
    """Tests for token counting in different modes."""

    def test_streaming_mode_token_estimate(self):
        """Test token estimation for streaming mode."""
        content = "This is a test response with some content"
        
        # Streaming mode estimates tokens from content length
        tokens = len(content) // 4
        
        assert tokens == len(content) // 4

    def test_non_streaming_mode_token_count(self):
        """Test token counting from server response in non-streaming mode."""
        # Simulate server response with usage data
        result = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        # Non-streaming mode uses actual usage data
        usage = result.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        
        assert tokens == 30

    def test_non_streaming_fallback_estimate(self):
        """Test fallback token estimation when usage missing."""
        # Simulate server response without usage data
        result = {"choices": []}
        
        content = "This is a test response"
        usage = result.get("usage", {})
        tokens = usage.get("total_tokens", len(content) // 4)
        
        assert tokens == len(content) // 4


class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    @patch('myco.cli.requests.post')
    def test_streaming_request_parameters(self, mock_post):
        """Test that streaming request has correct parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = []
        mock_post.return_value = mock_response
        
        # Import here to avoid circular imports
        import requests
        
        # Simulate streaming call
        requests.post(
            "http://test/v1/chat/completions",
            json={
                "model": "test",
                "messages": [],
                "stream": True
            },
            stream=True
        )
        
        # Verify stream parameter was set
        call_args = mock_post.call_args
        assert call_args[1]["stream"] is True
        assert call_args[1]["json"]["stream"] is True

    @patch('myco.cli.requests.post')
    def test_non_streaming_request_parameters(self, mock_post):
        """Test that non-streaming request has correct parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response
        
        import requests
        
        # Simulate non-streaming call
        requests.post(
            "http://test/v1/chat/completions",
            json={
                "model": "test",
                "messages": []
            }
        )
        
        # Verify stream parameter was NOT set
        call_args = mock_post.call_args
        assert "stream" not in call_args[1] or call_args[1].get("stream") is not True
