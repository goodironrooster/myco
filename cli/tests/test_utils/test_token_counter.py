"""Tests for token counting utilities."""

import pytest

from cli.utils.token_counter import (
    TokenCounter,
    TokenCount,
    estimate_tokens,
    count_words,
)


class TestTokenCount:
    """Test TokenCount dataclass."""

    def test_default_values(self):
        """Test default token count values."""
        count = TokenCount()
        assert count.prompt_tokens == 0
        assert count.completion_tokens == 0
        assert count.total_tokens == 0

    def test_custom_values(self):
        """Test token count with custom values."""
        count = TokenCount(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert count.prompt_tokens == 100
        assert count.completion_tokens == 50
        assert count.total_tokens == 150

    def test_human_readable_empty(self):
        """Test human readable format with zero values."""
        count = TokenCount()
        assert count.human_readable == "0 tokens"

    def test_human_readable_partial(self):
        """Test human readable format with partial values."""
        count = TokenCount(prompt_tokens=100, completion_tokens=50)
        assert "prompt: 100" in count.human_readable
        assert "completion: 50" in count.human_readable

    def test_human_readable_full(self):
        """Test human readable format with all values."""
        count = TokenCount(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        readable = count.human_readable
        assert "prompt: 1,000" in readable
        assert "completion: 500" in readable
        assert "total: 1,500" in readable


class TestTokenCounter:
    """Test TokenCounter functionality."""

    def test_start_session(self):
        """Test starting a new session."""
        counter = TokenCounter()
        counter.start_session("session-123")

        assert counter._current_session == "session-123"
        assert "session-123" in counter._session_tokens

    def test_record_usage(self):
        """Test recording token usage."""
        counter = TokenCounter()
        counter.start_session("test-session")

        result = counter.record_usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_record_usage_accumulates(self):
        """Test that token usage accumulates."""
        counter = TokenCounter()
        counter.start_session("test-session")

        counter.record_usage(100, 50, 150)
        counter.record_usage(200, 100, 300)

        result = counter.get_current_tokens()
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 150
        assert result.total_tokens == 450

    def test_record_usage_calculates_total(self):
        """Test that total is calculated if not provided."""
        counter = TokenCounter()
        counter.start_session("test-session")

        result = counter.record_usage(
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert result.total_tokens == 150

    def test_get_session_tokens(self):
        """Test getting tokens for specific session."""
        counter = TokenCounter()

        counter.start_session("session-1")
        counter.record_usage(100, 50)

        counter.start_session("session-2")
        counter.record_usage(200, 100)

        tokens1 = counter.get_session_tokens("session-1")
        tokens2 = counter.get_session_tokens("session-2")

        assert tokens1.prompt_tokens == 100
        assert tokens2.prompt_tokens == 200

    def test_get_session_tokens_not_found(self):
        """Test getting tokens for non-existent session."""
        counter = TokenCounter()
        result = counter.get_session_tokens("nonexistent")
        assert result is None

    def test_get_current_tokens(self):
        """Test getting current session tokens."""
        counter = TokenCounter()
        counter.start_session("current-test")
        counter.record_usage(500, 250)

        result = counter.get_current_tokens()
        assert result.prompt_tokens == 500
        assert result.completion_tokens == 250

    def test_get_current_tokens_no_session(self):
        """Test getting tokens without active session."""
        counter = TokenCounter()
        result = counter.get_current_tokens()
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_get_lifetime_tokens(self):
        """Test getting lifetime token count."""
        counter = TokenCounter()

        counter.start_session("session-1")
        counter.record_usage(100, 50)

        counter.start_session("session-2")
        counter.record_usage(200, 100)

        lifetime = counter.get_lifetime_tokens()

        assert lifetime.prompt_tokens == 300
        assert lifetime.completion_tokens == 150
        assert lifetime.total_tokens == 450

    def test_reset_session(self):
        """Test resetting session token count."""
        counter = TokenCounter()
        counter.start_session("reset-test")
        counter.record_usage(1000, 500)

        counter.reset_session()

        result = counter.get_current_tokens()
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_reset_session_by_id(self):
        """Test resetting specific session."""
        counter = TokenCounter()

        counter.start_session("session-1")
        counter.record_usage(100, 50)

        counter.start_session("session-2")
        counter.record_usage(200, 100)

        counter.reset_session("session-1")

        tokens1 = counter.get_session_tokens("session-1")
        tokens2 = counter.get_session_tokens("session-2")

        assert tokens1.prompt_tokens == 0
        assert tokens2.prompt_tokens == 200

    def test_reset_all(self):
        """Test resetting all token counts."""
        counter = TokenCounter()

        counter.start_session("session-1")
        counter.record_usage(100, 50)

        counter.start_session("session-2")
        counter.record_usage(200, 100)

        counter.reset_all()

        assert counter._session_tokens == {}
        assert counter._current_session is None
        assert counter.get_lifetime_tokens().total_tokens == 0

    def test_default_session(self):
        """Test recording without starting session uses default."""
        counter = TokenCounter()
        counter.record_usage(100, 50)

        result = counter.get_session_tokens("default")
        assert result is not None
        assert result.prompt_tokens == 100


class TestEstimateTokens:
    """Test token estimation functions."""

    def test_estimate_empty(self):
        """Test estimating empty string."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_short(self):
        """Test estimating short text."""
        # "Hello" = 5 chars, should be at least 1 token
        assert estimate_tokens("Hello") >= 1

    def test_estimate_medium(self):
        """Test estimating medium length text."""
        text = "This is a test sentence with about twenty words in it total"
        estimate = estimate_tokens(text)
        # Rough estimate: ~60 chars / 4 = ~15 tokens
        assert estimate > 10
        assert estimate < 25

    def test_estimate_whitespace(self):
        """Test that extra whitespace doesn't affect estimate."""
        text1 = estimate_tokens("Hello world")
        text2 = estimate_tokens("  Hello   world  ")
        assert text1 == text2

    def test_estimate_long_text(self):
        """Test estimating long text."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        estimate = estimate_tokens(text)
        # ~45 chars * 100 = 4500 chars / 4 = ~1125 tokens
        assert estimate > 1000
        assert estimate < 1300


class TestCountWords:
    """Test word counting function."""

    def test_count_empty(self):
        """Test counting empty string."""
        assert count_words("") == 0
        assert count_words(None) == 0
        assert count_words("   ") == 0

    def test_count_single(self):
        """Test counting single word."""
        assert count_words("Hello") == 1

    def test_count_multiple(self):
        """Test counting multiple words."""
        assert count_words("Hello world test") == 3

    def test_count_punctuation(self):
        """Test counting with punctuation."""
        # Punctuation attached to words counts as part of word
        assert count_words("Hello, world! Test.") == 3

    def test_count_whitespace(self):
        """Test that extra whitespace doesn't affect count."""
        text1 = count_words("one two three")
        text2 = count_words("  one   two  three  ")
        assert text1 == text2 == 3

    def test_count_newlines(self):
        """Test counting across lines."""
        text = "Line one\nLine two\nLine three"
        assert count_words(text) == 6
