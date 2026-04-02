"""Server management core logic."""

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from ..utils.logging import LogConfig


@dataclass
class ServerStatus:
    """Status of the llama.cpp server."""

    running: bool
    pid: Optional[int] = None
    url: Optional[str] = None
    model: Optional[str] = None
    message: str = ""


class ServerManager:
    """Manages the llama.cpp server process."""

    def __init__(
        self,
        server_exe: Path,
        host: str = "127.0.0.1",
        port: int = 1234,
    ):
        """Initialize server manager.

        Args:
            server_exe: Path to llama-server.exe
            host: Server host
            port: Server port
        """
        self.server_exe = server_exe
        self.host = host
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._pid_file = Path.home() / ".gguf-cli" / "server.pid"
        self.logger = LogConfig.get_logger("gguf.server")

    @property
    def base_url(self) -> str:
        """Return the server base URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        """Return the server API URL."""
        return f"{self.base_url}/v1"

    def start(
        self,
        model_path: Path,
        context_length: int = 8192,
        threads: Optional[int] = None,
        background: bool = True,
        quiet: bool = True,  # MYCO: silent by default
        gpu_layers: Optional[int] = None,
        batch_size: int = 256,  # MYCO: optimized for 40+ tok/s
        flash_attn: bool = True,  # MYCO: flash attention for speed
    ) -> ServerStatus:
        """Start the llama.cpp server.

        Args:
            model_path: Path to the GGUF model file
            context_length: Context window size
            threads: Number of threads (None = auto)
            background: If True, run in background

        Returns:
            ServerStatus indicating result
        """
        # Check if already running
        if self.is_running():
            return ServerStatus(
                running=True,
                pid=self._get_stored_pid(),
                url=self.base_url,
                message="Server already running",
            )

        # Validate server executable
        if not self.server_exe.exists():
            return ServerStatus(
                running=False,
                message=f"Server executable not found: {self.server_exe}",
            )

        # Validate model
        if not model_path.exists():
            return ServerStatus(
                running=False,
                message=f"Model file not found: {model_path}",
            )

        # Build command
        cmd = [
            str(self.server_exe),
            "-m",
            str(model_path),
            "--port",
            str(self.port),
            "--host",
            self.host,
            "-c",
            str(context_length),
        ]

        # MYCO vision: silent by default, verbose on demand
        if quiet:
            cmd.append("--log-disable")

        if threads:
            cmd.extend(["-t", str(threads)])

        # GPU acceleration - MYCO optimized for 40+ tokens/second
        if gpu_layers is not None:
            cmd.extend(["--gpu-layers", str(gpu_layers)])

        # Batch size for CUDA (optimal: 256 for 40+ tok/s)
        if batch_size:
            cmd.extend(["-b", str(batch_size)])

        # Flash attention for speed
        if flash_attn:
            cmd.extend(["-fa", "on"])

        self.logger.info(f"Starting server: {' '.join(cmd)}")

        try:
            if background:
                # Windows-specific: use CREATE_NEW_PROCESS_GROUP
                # Don't pipe output in background mode to avoid blocking
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                self._store_pid(self._process.pid)

                # Give process time to start or fail
                time.sleep(3)

                # Check if process is still running
                if self._process.poll() is not None:
                    # Process exited - likely an error
                    return ServerStatus(
                        running=False,
                        pid=self._process.pid,
                        message="Server exited immediately. Check model compatibility with llama.cpp version.",
                    )

                # Wait for server to become healthy
                if self._wait_for_server(timeout=60):
                    return ServerStatus(
                        running=True,
                        pid=self._process.pid,
                        url=self.base_url,
                        model=model_path.name,
                        message="Server started successfully",
                    )
                else:
                    return ServerStatus(
                        running=False,
                        pid=self._process.pid,
                        message="Server process running but health check failed",
                    )
            else:
                # Foreground mode - capture output for error reporting
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._store_pid(self._process.pid)

                # Quick check: if process exits immediately, there was a startup error
                time.sleep(2)
                if self._process and self._process.poll() is not None:
                    # Process exited early - try to get error output
                    try:
                        stderr = (
                            self._process.stderr.read().decode("utf-8", errors="ignore")
                            if self._process.stderr
                            else ""
                        )
                        if stderr:
                            # Check for known errors
                            if "unknown model architecture" in stderr.lower():
                                # Extract architecture name
                                import re

                                arch_match = re.search(
                                    r"unknown model architecture: '(\w+)'", stderr
                                )
                                arch_name = arch_match.group(1) if arch_match else "unknown"
                                return ServerStatus(
                                    running=False,
                                    pid=self._process.pid,
                                    message=f"Model architecture '{arch_name}' not supported by this llama.cpp version. "
                                    f"Build from source: github.com/ggml-org/llama.cpp",
                                )
                            if "failed to load model" in stderr.lower():
                                return ServerStatus(
                                    running=False,
                                    pid=self._process.pid,
                                    message="Failed to load model (file may be corrupted or incompatible)",
                                )
                            # Extract last error line
                            error_lines = [l for l in stderr.splitlines() if "error" in l.lower()]
                            if error_lines:
                                return ServerStatus(
                                    running=False,
                                    pid=self._process.pid,
                                    message=f"Server exited: {error_lines[-1][:200]}",
                                )
                    except Exception:
                        pass

                    return ServerStatus(
                        running=False,
                        pid=self._process.pid,
                        message="Server exited immediately (check model compatibility)",
                    )

                # Wait for server to start
                if self._wait_for_server(timeout=60):
                    return ServerStatus(
                        running=True,
                        pid=self._process.pid,
                        url=self.base_url,
                        model=model_path.name,
                        message="Server started successfully",
                    )
                else:
                    return ServerStatus(
                        running=False,
                        pid=self._process.pid,
                        message="Server failed to start (timeout)",
                    )

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return ServerStatus(
                running=False,
                message=f"Failed to start server: {e}",
            )

    def stop(self) -> ServerStatus:
        """Stop the running server.

        Returns:
            ServerStatus indicating result
        """
        pid = self._get_stored_pid()

        if pid is None:
            # Try to find by port
            pid = self._find_process_by_port()

        if pid is None:
            return ServerStatus(
                running=False,
                message="No running server found",
            )

        try:
            import ctypes

            ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, pid)  # CTRL_C_EVENT
        except Exception:
            try:
                process = subprocess.Popen(
                    ["taskkill", "/F", "/PID", str(pid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                process.communicate()
            except Exception as e:
                return ServerStatus(
                    running=True,
                    pid=pid,
                    message=f"Failed to stop server: {e}",
                )

        self._clear_pid_file()
        time.sleep(1)

        if not self.is_running():
            return ServerStatus(
                running=False,
                message="Server stopped successfully",
            )

        return ServerStatus(
            running=True,
            pid=pid,
            message="Server may still be running",
        )

    def status(self) -> ServerStatus:
        """Check server status.

        Returns:
            ServerStatus with current state
        """
        pid = self._get_stored_pid()

        if pid is None:
            pid = self._find_process_by_port()

        if pid is None:
            return ServerStatus(
                running=False,
                message="Server not running",
            )

        # Check if process exists
        try:
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x0001, False, pid)
            if not handle:
                self._clear_pid_file()
                return ServerStatus(
                    running=False,
                    message="Server process not found",
                )
            ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass

        # Check health endpoint
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return ServerStatus(
                    running=True,
                    pid=pid,
                    url=self.base_url,
                    message="Server is healthy",
                )
        except requests.RequestException:
            pass

        return ServerStatus(
            running=True,
            pid=pid,
            url=self.base_url,
            message="Server process exists but health check failed",
        )

    def is_running(self) -> bool:
        """Check if server is currently running."""
        status = self.status()
        return status.running

    def _wait_for_server(self, timeout: int = 30, interval: float = 1.0) -> bool:
        """Wait for server to become healthy.

        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if server became healthy
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    self.logger.info("Server is healthy")
                    return True
            except requests.RequestException:
                pass

            time.sleep(interval)

        self.logger.warning("Server health check timed out")
        return False

    def _store_pid(self, pid: int) -> None:
        """Store PID to file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(pid))

    def _get_stored_pid(self) -> Optional[int]:
        """Get stored PID from file."""
        if self._pid_file.exists():
            try:
                return int(self._pid_file.read_text().strip())
            except ValueError:
                self._clear_pid_file()
        return None

    def _clear_pid_file(self) -> None:
        """Clear the PID file."""
        if self._pid_file.exists():
            self._pid_file.unlink()

    def _find_process_by_port(self) -> Optional[int]:
        """Find process using the server port."""
        try:
            # Use netstat to find process using the port
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.splitlines():
                if f":{self.port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        return int(parts[-1])
        except Exception:
            pass

        return None
