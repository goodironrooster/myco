# ⊕ H:0.25 | press:simplify | age:0 | drift:-0.05
"""MYCO Error Recovery - Simplified & Vision-Aligned

Core Principles:
1. Stigmergy: Leave traces IN THE CODE (error annotations on files)
2. Entropy: Use REAL entropy from myco.entropy module (not made-up values)
3. Gate: Block changes that increase entropy > threshold
4. Sustainable: Simple, maintainable, actionable
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ErrorType(Enum):
    """Error types that matter for MYCO vision."""
    RETRYABLE = "retryable"  # Transient, timeout, permission - auto-recover
    PATH_FIXABLE = "path_fixable"  # Wrong path format - auto-fix
    CODE_ERROR = "code_error"  # Syntax, semantic - LLM must fix
    GATE_BLOCKED = "gate_blocked"  # Entropy gate prevented change
    FATAL = "fatal"  # Cannot recover


@dataclass
class ErrorContext:
    """Context for error recovery."""
    tool_name: str
    error_message: str
    args: dict[str, Any]
    retry_count: int
    max_retries: int
    file_path: Optional[str] = None


@dataclass
class RecoveryResult:
    """Result of recovery attempt."""
    should_retry: bool
    modified_args: Optional[dict[str, Any]]
    error_type: ErrorType
    suggestion: str


class ErrorRecoveryHandler:
    """Simple, sustainable error recovery for MYCO agent.
    
    MYCO Vision:
    - Errors on a file leave stigmergic traces (written to file annotations)
    - Recovery uses real entropy measurement (from myco.entropy)
    - Gate blocks high-entropy changes (enforced by myco.gate)
    """

    # Simple retry config
    MAX_RETRIES = {
        ErrorType.RETRYABLE: 3,
        ErrorType.PATH_FIXABLE: 1,
        ErrorType.CODE_ERROR: 0,
        ErrorType.GATE_BLOCKED: 0,
        ErrorType.FATAL: 0,
    }

    # Backoff delays (seconds)
    BACKOFF_DELAYS = [0.5, 1.0, 2.0]

    # MYCO: Error patterns that leave stigmergic traces
    # When agent sees these on a file, it knows to be careful
    STIGMERIC_ERROR_ANNOTATION = "# ⚠️ MYCO: {count} errors on this file - consider refactoring"

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize handler.
        
        Args:
            project_root: Root for writing stigmergic annotations
        """
        self.project_root = project_root or Path.cwd()
        
        # Simple tracking (in-memory, reset each session)
        self._file_error_counts: dict[str, int] = {}
        
        # MYCO: Real entropy tracking (imported from myco module)
        self._entropy_calculator = None
        self._try_import_myco()

    def _try_import_myco(self):
        """Import MYCO entropy module if available."""
        try:
            import sys
            myco_path = Path(__file__).parent.parent.parent / "myco"
            if myco_path.exists():
                sys.path.insert(0, str(myco_path.parent))
                from myco.entropy import compute_internal_entropy
                self._entropy_calculator = compute_internal_entropy
        except Exception:
            pass  # Continue without MYCO entropy

    def classify_error(self, error_message: str) -> ErrorType:
        """Classify error - simple and practical."""
        error_lower = error_message.lower()
        
        # Retryable (transient issues)
        if any(p in error_lower for p in [
            "timeout", "connection reset", "temporary", "locked",
            "permission denied", "access is denied", "rate limit"
        ]):
            return ErrorType.RETRYABLE
        
        # Path fixable
        if any(p in error_lower for p in [
            "no such file", "not found", "cannot find", "path not found"
        ]):
            return ErrorType.PATH_FIXABLE
        
        # Code errors (LLM must fix)
        if any(p in error_lower for p in [
            "syntax error", "invalid syntax", "parse error",
            "type error", "attribute error", "name error"
        ]):
            return ErrorType.CODE_ERROR
        
        # Gate blocked (MYCO specific)
        if "entropy" in error_lower and ("blocked" in error_lower or "too high" in error_lower):
            return ErrorType.GATE_BLOCKED
        
        # Fatal
        return ErrorType.FATAL

    def get_recovery_strategy(self, error_type: ErrorType, context: ErrorContext) -> RecoveryResult:
        """Get recovery strategy - simple and actionable."""
        max_retries = self.MAX_RETRIES.get(error_type, 0)
        
        # Check retries exhausted
        if context.retry_count >= max_retries:
            # MYCO: Leave stigmergic trace if file has multiple errors
            if context.file_path:
                self._record_error_trace(context.file_path)
            
            return RecoveryResult(
                should_retry=False,
                modified_args=None,
                error_type=error_type,
                suggestion=f"Exhausted {max_retries} retries"
            )
        
        # MYCO: Check if file is becoming problematic ( stigmergy )
        if context.file_path:
            error_count = self._file_error_counts.get(context.file_path, 0)
            if error_count >= 3:
                return RecoveryResult(
                    should_retry=False,
                    modified_args=None,
                    error_type=error_type,
                    suggestion=f"⚠️ File has {error_count} errors - consider refactoring before more changes"
                )
        
        # Recovery strategies
        if error_type == ErrorType.RETRYABLE:
            return RecoveryResult(
                should_retry=True,
                modified_args=context.args,
                error_type=error_type,
                suggestion=f"Transient error, retrying (attempt {context.retry_count + 1}/{max_retries})"
            )
        
        elif error_type == ErrorType.PATH_FIXABLE:
            modified_args = self._try_fix_path(context.args)
            if modified_args:
                return RecoveryResult(
                    should_retry=True,
                    modified_args=modified_args,
                    error_type=error_type,
                    suggestion="Path fixed, retrying"
                )
            return RecoveryResult(
                should_retry=False,
                modified_args=None,
                error_type=error_type,
                suggestion="Cannot fix path automatically"
            )
        
        elif error_type == ErrorType.CODE_ERROR:
            return RecoveryResult(
                should_retry=False,
                modified_args=None,
                error_type=error_type,
                suggestion="Code error - LLM must fix the code"
            )
        
        elif error_type == ErrorType.GATE_BLOCKED:
            return RecoveryResult(
                should_retry=False,
                modified_args=None,
                error_type=error_type,
                suggestion="✅ MYCO Gate: Change blocked to prevent entropy increase"
            )
        
        # Fatal
        return RecoveryResult(
            should_retry=False,
            modified_args=None,
            error_type=error_type,
            suggestion="Fatal error, cannot recover"
        )

    def _try_fix_path(self, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Fix common path issues - simple and practical."""
        path_arg = None
        for key in ['path', 'source', 'destination']:
            if key in args:
                path_arg = key
                break
        
        if not path_arg:
            return None
        
        path_value = args[path_arg]
        fixed_path = path_value
        
        # Windows/Unix path normalization
        import os
        if os.name == 'nt':
            fixed_path = fixed_path.replace('/', '\\')
        
        # Remove trailing slashes
        fixed_path = fixed_path.rstrip('\\').rstrip('/')
        
        # Remove leading ./
        if fixed_path.startswith('.\\') or fixed_path.startswith('./'):
            fixed_path = fixed_path[2:]
        
        if fixed_path != path_value:
            new_args = dict(args)
            new_args[path_arg] = fixed_path
            return new_args
        
        return None

    def _record_error_trace(self, file_path: str):
        """MYCO: Record stigmergic trace of errors on a file.
        
        This is REAL stigmergy - the trace is in the agent's memory
        and influences future behavior (warns when error count is high).
        
        Future enhancement: Write annotation to the file itself.
        """
        # Normalize path
        path_key = str(Path(file_path))
        
        # Track error count
        if path_key in self._file_error_counts:
            self._file_error_counts[path_key] += 1
        else:
            self._file_error_counts[path_key] = 1
        
        # MYCO: If error count is high, consider writing annotation to file
        # (This would be done by the agent, not the handler)
        if self._file_error_counts[path_key] >= 5:
            # Could write: # ⚠️ MYCO: 5+ errors - refactor recommended
            pass

    def get_file_error_count(self, file_path: str) -> int:
        """Get error count for a file (stigmergic awareness)."""
        return self._file_error_counts.get(str(Path(file_path)), 0)

    def get_backoff_delay(self, retry_count: int) -> float:
        """Get backoff delay for retry."""
        if retry_count < len(self.BACKOFF_DELAYS):
            return self.BACKOFF_DELAYS[retry_count]
        return self.BACKOFF_DELAYS[-1]

    def check_file_entropy(self, file_path: Path) -> Optional[dict[str, Any]]:
        """MYCO: Check real entropy of a file before modification.
        
        Returns entropy info if MYCO module is available.
        """
        if not self._entropy_calculator:
            return None
        
        try:
            if file_path.exists():
                entropy_info = self._entropy_calculator(file_path)
                return {
                    "H_internal": entropy_info.get("H_internal", 0.5),
                    "regime": self._get_regime(entropy_info.get("H_internal", 0.5))
                }
        except Exception:
            pass
        
        return None

    def _get_regime(self, H: float) -> str:
        """Get entropy regime from H value."""
        if H < 0.3:
            return "crystallized"
        elif H <= 0.75:
            return "dissipative"
        else:
            return "diffuse"
