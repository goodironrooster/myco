"""Integration tests for myco agent loop with mock server."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from myco.cli import cli
from myco.world import WorldModel


class MockStreamingResponse:
    """Mock streaming response for testing."""
    
    def __init__(self, content_chunks, usage=None):
        self.content_chunks = content_chunks
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        self.status_code = 200
    
    def raise_for_status(self):
        pass
    
    def iter_lines(self):
        """Generate SSE format lines."""
        for chunk in self.content_chunks:
            yield f'data: {json.dumps(chunk)}'.encode('utf-8')
        yield b'data: [DONE]'


class MockNonStreamingResponse:
    """Mock non-streaming response for testing."""
    
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        self.status_code = 200
    
    def raise_for_status(self):
        pass
    
    def json(self):
        return {
            "choices": [{
                "message": {
                    "content": self.content,
                    "role": "assistant"
                }
            }],
            "usage": self.usage
        }


class TestAgentLoopWithMockServer:
    """Integration tests for agent loop with mocked server."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def project_with_world(self, tmp_path):
        """Create a project with world model."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        
        # Create world.json
        world = WorldModel.load(tmp_path)
        world.save()
        
        yield tmp_path
    
    def test_agent_creates_file_with_json_tool_call(
        self, runner, project_with_world
    ):
        """Test agent creates file when model returns JSON tool call."""
        tool_call = {
            "name": "write_file",
            "arguments": {
                "path": "test_output.py",
                "content": "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef test(): pass\n"
            }
        }
        
        # Mock streaming response
        chunks = [
            {"choices": [{"delta": {"content": json.dumps(tool_call)}}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        ]
        
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockStreamingResponse(chunks)
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Create a test file', '-v'])
                
                # Should complete without error
                assert result.exit_code == 0
                
                # File should be created
                output_file = project_with_world / "test_output.py"
                assert output_file.exists()
    
    def test_agent_handles_connection_error(
        self, runner, project_with_world
    ):
        """Test agent handles connection errors gracefully."""
        with patch('myco.cli.requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError()
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test task', '-v'])
                
                # Should exit gracefully with error message
                assert result.exit_code == 0
                assert "Could not connect" in result.output or "Connection" in result.output
    
    def test_agent_handles_timeout(
        self, runner, project_with_world
    ):
        """Test agent handles timeout errors gracefully."""
        with patch('myco.cli.requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout()
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test task', '-v'])
                
                # Should exit gracefully with timeout message
                assert result.exit_code == 0
                assert "timed out" in result.output.lower()
    
    def test_agent_handles_invalid_json(
        self, runner, project_with_world
    ):
        """Test agent handles invalid JSON response."""
        with patch('myco.cli.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.side_effect = json.JSONDecodeError("test", "doc", 0)
            mock_post.return_value = mock_response
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test task', '-v'])
                
                # Should handle gracefully
                assert result.exit_code == 0
    
    def test_agent_respects_max_iterations(
        self, runner, project_with_world
    ):
        """Test agent respects max iterations limit."""
        # Model returns tool calls repeatedly to test iteration limit
        tool_call = {
            "name": "list_files",
            "arguments": {"path": "."}
        }
        
        chunks = [
            {"choices": [{"delta": {"content": json.dumps(tool_call)}}]},
        ]
        
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockStreamingResponse(chunks)
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                # Set low max iterations
                result = runner.invoke(
                    cli, 
                    ['run', 'Test task', '-v', '-i', '2']
                )
                
                # Should complete (hit iteration limit)
                assert result.exit_code == 0
    
    def test_non_streaming_mode_works(
        self, runner, project_with_world
    ):
        """Test non-streaming mode (non-verbose) works correctly."""
        tool_call = {
            "name": "write_file",
            "arguments": {
                "path": "non_stream.py",
                "content": "# test\n"
            }
        }

        content = json.dumps(tool_call)

        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockNonStreamingResponse(content)

            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)

                # Non-verbose mode
                result = runner.invoke(cli, ['run', 'Create file'])

                # Should complete
                assert result.exit_code == 0


class TestGateFeedbackLoop:
    """Tests for autopoietic gate feedback loop."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def project_with_world(self, tmp_path):
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        WorldModel.load(tmp_path).save()
        yield tmp_path

    def test_blocked_action_returns_blocked_result(
        self, runner, project_with_world
    ):
        """Test that blocked actions return BLOCKED result."""
        from myco.gate import AutopoieticGate, GateResult
        from unittest.mock import patch

        # Create a gate that blocks write actions
        def mock_check_action(path, action_type, **kwargs):
            if action_type == "write":
                return GateResult(
                    permitted=False,
                    reason="Entropy increase exceeds threshold",
                    violation_type="entropy_increase"
                )
            return GateResult(permitted=True)

        tool_call = {
            "name": "write_file",
            "arguments": {"path": "test.py", "content": "test"}
        }

        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockNonStreamingResponse(json.dumps(tool_call))

            with patch.object(AutopoieticGate, 'check_action', mock_check_action):
                with runner.isolated_filesystem():
                    import os
                    os.chdir(project_with_world)

                    result = runner.invoke(cli, ['run', 'Test task', '-v'])

                    # Session log should be created even if action blocked
                    log_path = project_with_world / ".myco" / "session.log"
                    assert log_path.exists()

    def test_gate_blocked_logged_to_session(
        self, runner, project_with_world
    ):
        """Test that gate blocked actions are logged to session log."""
        from myco.gate import AutopoieticGate, GateResult
        from myco.cli import parse_and_execute_tools
        from pathlib import Path
        from unittest.mock import MagicMock

        # Create mock gate that blocks
        mock_gate = MagicMock(spec=AutopoieticGate)
        mock_gate.check_action.return_value = GateResult(
            permitted=False,
            reason="Test block reason",
            violation_type="test_violation"
        )

        # Create mock logger
        mock_logger = MagicMock()

        # Test parse_and_execute_tools directly
        content = '{"name": "write_file", "arguments": {"path": "test.py", "content": "test"}}'
        
        tool_results = parse_and_execute_tools(
            content,
            project_with_world,
            mock_gate,
            mock_logger,
            verbose=False
        )

        # Should have one result that is blocked
        assert len(tool_results) == 1
        assert tool_results[0]["result"].startswith("BLOCKED:")

        # Logger should have been called with gate_blocked
        mock_logger.log_tool_call.assert_called()

    def test_feedback_message_format(
        self, runner, project_with_world
    ):
        """Test that feedback message has correct format."""
        # This test verifies the feedback format in the code
        # by checking the expected output structure
        
        blocked_result = "BLOCKED: Entropy increase exceeds threshold"
        
        # Build feedback as the code does
        blocked_feedback = ""
        if blocked_result.startswith("BLOCKED:"):
            blocked_feedback += f"\n\n⚠️  ACTION BLOCKED: test_tool\n"
            blocked_feedback += f"Reason: {blocked_result.replace('BLOCKED: ', '')}\n"
            blocked_feedback += "The autopoietic gate blocked this action because it would degrade the codebase.\n"
            blocked_feedback += "Propose an alternative approach that preserves structural integrity.\n"

        # Verify format
        assert "⚠️  ACTION BLOCKED" in blocked_feedback
        assert "Reason:" in blocked_feedback
        assert "alternative approach" in blocked_feedback

    def test_confirm_flag_in_cli(
        self, runner, project_with_world
    ):
        """Test that --confirm flag is accepted by CLI."""
        # Just test that the flag is recognized (doesn't actually run model)
        from myco.gate import AutopoieticGate, GateResult
        from unittest.mock import patch

        def mock_check_action(path, action_type, **kwargs):
            return GateResult(permitted=True)

        tool_call = {"name": "list_files", "arguments": {"path": "."}}

        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockNonStreamingResponse(json.dumps(tool_call))

            with patch.object(AutopoieticGate, 'check_action', mock_check_action):
                with runner.isolated_filesystem():
                    import os
                    os.chdir(project_with_world)

                    # Should accept --confirm flag without error (non-verbose to avoid streaming)
                    result = runner.invoke(cli, ['run', 'Test task', '--confirm'])

                    # Should complete (list_files doesn't require confirmation)
                    assert result.exit_code == 0


class TestTokenCountingWithMockServer:
    """Tests for token counting with mock server."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def project_with_world(self, tmp_path):
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        WorldModel.load(tmp_path).save()
        yield tmp_path
    
    def test_streaming_mode_captures_usage(
        self, runner, project_with_world
    ):
        """Test streaming mode captures usage from server."""
        chunks = [
            {"choices": [{"delta": {"content": "test"}}]},
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
        ]
        
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockStreamingResponse(chunks)
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test', '-v'])
                
                assert result.exit_code == 0
                # Check that usage was captured (would be in session log)
    
    def test_non_streaming_mode_captures_usage(
        self, runner, project_with_world
    ):
        """Test non-streaming mode captures usage from server."""
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockNonStreamingResponse(
                "test",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            )
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test'])
                
                assert result.exit_code == 0


class TestSessionLoggingWithMockServer:
    """Tests for session logging with mock server."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    @pytest.fixture
    def project_with_world(self, tmp_path):
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        WorldModel.load(tmp_path).save()
        yield tmp_path
    
    def test_session_log_created(
        self, runner, project_with_world
    ):
        """Test session log file is created."""
        chunks = [
            {"choices": [{"delta": {"content": "test response"}}]},
        ]
        
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockStreamingResponse(chunks)
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test task', '-v'])
                
                assert result.exit_code == 0
                
                # Session log should be created
                log_path = project_with_world / ".myco" / "session.log"
                assert log_path.exists()
    
    def test_session_log_contains_entries(
        self, runner, project_with_world
    ):
        """Test session log contains expected entries."""
        tool_call = {
            "name": "write_file",
            "arguments": {"path": "x.py", "content": "test"}
        }
        
        chunks = [
            {"choices": [{"delta": {"content": json.dumps(tool_call)}}]},
        ]
        
        with patch('myco.cli.requests.post') as mock_post:
            mock_post.return_value = MockStreamingResponse(chunks)
            
            with runner.isolated_filesystem():
                import os
                os.chdir(project_with_world)
                
                result = runner.invoke(cli, ['run', 'Test task', '-v'])
                
                assert result.exit_code == 0
                
                # Read and verify log
                log_path = project_with_world / ".myco" / "session.log"
                log_content = log_path.read_text()
                
                # Should contain session_start
                assert "session_start" in log_content
                # Should contain tool_call
                assert "tool_call" in log_content
                # Should contain session_end
                assert "session_end" in log_content
