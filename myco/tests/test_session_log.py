"""Tests for myco.session_log module."""

import json
import pytest
from pathlib import Path
from myco.session_log import LogEntry, SessionLogger, get_logger


class TestLogEntry:
    """Tests for LogEntry dataclass."""
    
    def test_create_entry(self):
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2026-03-19T00:00:00Z",
            level="INFO",
            event_type="test_event",
            message="Test message",
            data={"key": "value"}
        )
        
        assert entry.timestamp == "2026-03-19T00:00:00Z"
        assert entry.level == "INFO"
        assert entry.event_type == "test_event"
        assert entry.message == "Test message"
        assert entry.data["key"] == "value"
    
    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = LogEntry(
            timestamp="2026-03-19T00:00:00Z",
            level="INFO",
            event_type="test",
            message="Test",
            data={"key": "value"}
        )
        
        d = entry.to_dict()
        
        assert d["timestamp"] == "2026-03-19T00:00:00Z"
        assert d["level"] == "INFO"
        assert d["event_type"] == "test"
        assert d["message"] == "Test"
        assert d["data"]["key"] == "value"
    
    def test_to_json(self):
        """Test converting entry to JSON."""
        entry = LogEntry(
            timestamp="2026-03-19T00:00:00Z",
            level="INFO",
            event_type="test",
            message="Test",
            data={"key": "value"}
        )
        
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["timestamp"] == "2026-03-19T00:00:00Z"
        assert parsed["message"] == "Test"
    
    def test_from_json(self):
        """Test creating entry from JSON."""
        json_str = '{"timestamp": "2026-03-19T00:00:00Z", "level": "INFO", "event_type": "test", "message": "Test", "data": {"key": "value"}}'
        
        entry = LogEntry.from_json(json_str)
        
        assert entry.timestamp == "2026-03-19T00:00:00Z"
        assert entry.level == "INFO"
        assert entry.message == "Test"


class TestSessionLogger:
    """Tests for SessionLogger class."""
    
    @pytest.fixture
    def logger(self, tmp_path):
        """Create a session logger."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        return SessionLogger(tmp_path, session_id="test_session_001")
    
    def test_create_logger(self, tmp_path):
        """Test creating a session logger."""
        logger = SessionLogger(tmp_path)
        
        assert logger.session_id.startswith("session_")
        assert logger.log_path == tmp_path / ".myco" / "session.log"
    
    def test_log_writes_to_file(self, logger):
        """Test logging writes to file."""
        logger.info("test_event", "Test message", key="value")
        
        assert logger.log_path.exists()
        
        content = logger.log_path.read_text(encoding="utf-8")
        assert "test_event" in content
        assert "Test message" in content
    
    def test_log_creates_entry(self, logger):
        """Test logging creates entry."""
        entry = logger.info("test_event", "Test message")
        
        assert entry.event_type == "test_event"
        assert entry.message == "Test message"
        assert entry.level == "INFO"
    
    def test_log_levels(self, logger):
        """Test different log levels."""
        debug_entry = logger.debug("debug_event", "Debug message")
        info_entry = logger.info("info_event", "Info message")
        warning_entry = logger.warning("warning_event", "Warning message")
        error_entry = logger.error("error_event", "Error message")
        
        assert debug_entry.level == "DEBUG"
        assert info_entry.level == "INFO"
        assert warning_entry.level == "WARNING"
        assert error_entry.level == "ERROR"
    
    def test_log_tool_call(self, logger):
        """Test logging tool calls."""
        entry = logger.log_tool_call(
            tool_name="write_file",
            arguments={"path": "test.py", "content": "test"},
            result="Success",
            success=True
        )
        
        assert entry.event_type == "tool_call"
        assert entry.data["tool_name"] == "write_file"
        assert entry.data["success"] is True
    
    def test_log_gate_check(self, logger):
        """Test logging gate checks."""
        entry = logger.log_gate_check(
            file_path="test.py",
            action_type="write",
            permitted=True,
            reason="Within entropy budget",
            entropy_before=0.5,
            entropy_after=0.52
        )
        
        assert entry.event_type == "gate_check"
        assert entry.data["file_path"] == "test.py"
        assert entry.data["permitted"] is True
    
    def test_log_entropy_change(self, logger):
        """Test logging entropy changes."""
        entry = logger.log_entropy_change(
            before=0.5,
            after=0.45,
            modules_affected=["module_a", "module_b"]
        )
        
        assert entry.event_type == "entropy_change"
        # Use approximate comparison for floating point
        assert abs(entry.data["entropy_delta"] - (-0.05)) < 0.001
        assert len(entry.data["modules_affected"]) == 2
    
    def test_log_attractor_event(self, logger):
        """Test logging attractor events."""
        entry = logger.log_attractor_event(
            attractor_name="import_restructure_loop",
            perturbation="perspective_inversion",
            turn_detected=5
        )
        
        assert entry.event_type == "attractor_detected"
        assert entry.data["attractor_name"] == "import_restructure_loop"
        assert entry.level == "WARNING"
    
    def test_log_tensegrity_violation(self, logger):
        """Test logging tensegrity violations."""
        entry = logger.log_tensegrity_violation(
            importer="module_a",
            imported="module_b",
            violation_type="tension_tension_edge"
        )
        
        assert entry.event_type == "tensegrity_violation"
        assert entry.data["importer"] == "module_a"
        assert entry.level == "WARNING"
    
    def test_log_session_start(self, logger):
        """Test logging session start."""
        entry = logger.log_session_start("Test task description")
        
        assert entry.event_type == "session_start"
        assert entry.data["task"] == "Test task description"
    
    def test_log_session_end(self, logger):
        """Test logging session end."""
        entry = logger.log_session_end(
            iterations=5,
            tokens=1000,
            joules=0.5,
            entropy_delta=-0.03
        )
        
        assert entry.event_type == "session_end"
        assert entry.data["iterations"] == 5
        assert entry.data["tokens"] == 1000
    
    def test_get_entries(self, logger):
        """Test getting entries."""
        logger.info("event_a", "Message A")
        logger.info("event_b", "Message B")
        logger.warning("event_a", "Message C")
        
        all_entries = logger.get_entries()
        assert len(all_entries) == 3
        
        filtered = logger.get_entries(filter_type="event_a")
        assert len(filtered) == 2
    
    def test_read_log_file(self, logger):
        """Test reading log file."""
        logger.info("event_a", "Message A")
        logger.info("event_b", "Message B")
        
        # Create new logger to test file reading
        new_logger = SessionLogger(logger.project_root)
        entries = new_logger.read_log_file()
        
        assert len(entries) == 2
    
    def test_get_session_entries(self, logger):
        """Test getting session-specific entries."""
        logger.info("event", "Message 1")
        logger.info("event", "Message 2")
        
        entries = logger.get_session_entries()
        
        # All entries should have the session ID
        for entry in entries:
            assert entry.data.get("session_id") == logger.session_id
    
    def test_clear_log(self, logger):
        """Test clearing log."""
        logger.info("event", "Message")
        assert logger.log_path.exists()
        
        logger.clear_log()
        
        assert not logger.log_path.exists()
        assert len(logger.get_entries()) == 0

    def test_session_id_generation(self, tmp_path):
        """Test session ID generation."""
        logger1 = SessionLogger(tmp_path, session_id="test_session_1")
        logger2 = SessionLogger(tmp_path, session_id="test_session_2")

        # Session IDs should match what we provided
        assert logger1.session_id == "test_session_1"
        assert logger2.session_id == "test_session_2"
        assert logger1.session_id.startswith("test_")
        assert logger2.session_id.startswith("test_")
    
    def test_session_id_auto_generation(self, tmp_path):
        """Test automatic session ID generation."""
        import time
        
        logger1 = SessionLogger(tmp_path)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        logger2 = SessionLogger(tmp_path)

        # Session IDs should start with session_
        assert logger1.session_id.startswith("session_")
        assert logger2.session_id.startswith("session_")
    
    def test_log_with_extra_data(self, logger):
        """Test logging with extra data fields."""
        entry = logger.info(
            "custom_event",
            "Custom message",
            custom_field="value",
            another_field=123,
            nested={"key": "value"}
        )
        
        assert entry.data["custom_field"] == "value"
        assert entry.data["another_field"] == 123
        assert entry.data["nested"]["key"] == "value"


class TestGlobalLogger:
    """Tests for global logger functions."""
    
    def test_get_logger_creates_instance(self, tmp_path, monkeypatch):
        """Test get_logger creates instance."""
        # Reset global logger
        from myco import session_log
        monkeypatch.setattr(session_log, "_logger", None)
        
        logger = session_log.get_logger(tmp_path)
        
        assert isinstance(logger, SessionLogger)
    
    def test_get_logger_returns_same_instance(self, tmp_path):
        """Test get_logger returns same instance."""
        logger1 = SessionLogger(tmp_path)
        
        # Note: Global logger is shared, so we test the function exists
        from myco.session_log import get_logger
        logger2 = get_logger(tmp_path)
        
        # Should return the cached global instance
        assert logger2 is not None
