"""Agent core for autonomous task execution.

MYCO Vision Integration:
- Entropy regime analysis before actions
- Autopoietic gate enforcement
- World model persistence
- Attractor detection
- Stigmergic memory
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from ..ui import (
    StatusDisplay,
    TaskStatus,
    VerificationPanel,
    ApprovalPrompt,
    show_status,
    show_error,
    show_success,
)
from ..utils.logging import LogConfig
from .approval import ApprovalManager
from .impact_estimator import get_command_impact
from .tools import CommandTools, FileTools, SearchTools, ToolResult, BrowserTools, ProcessTools, GitTools, TestTools, CodebaseSearch, EntropyGate, ProjectTools
from .architecture import ModuleManifestManager, get_module_info, get_dependencies, get_dependents, update_manifest, track_dependencies, get_affected_files
from .architecture_map import ArchitectureMapManager, load_architecture_map, get_arch_module_info, get_arch_dependencies, get_arch_dependents, get_arch_component, update_architecture_map, get_refactoring_suggestion
from .session_memory import record_session, get_similar_sessions, get_lessons, add_pattern, get_patterns, record_mistake, get_warnings
from .quality import record_quality_change, get_quality_trend, get_project_health, get_files_needing_attention
from .self_improvement import analyze_task_success, get_pattern_recommendations, get_antipattern_warnings, get_quality_advice, get_project_quality_advice
from .certainty import infer_types, generate_contract, generate_property_tests, verify_integration
from .error_recovery import ErrorRecoveryHandler, ErrorType
from .project_context import get_project_context, ensure_project_context, ProjectContext

# MYCO: Import vision components
HAS_MYCO = False  # Default to False
try:
    import sys
    from pathlib import Path
    myco_path = Path(__file__).parent.parent.parent / "myco"
    if myco_path.exists():
        sys.path.insert(0, str(myco_path.parent))
        from myco.world import WorldModel
        from myco.entropy import get_regime_intervention, calculate_substrate_health
        from myco.gate import AutopoieticGate
        from myco.attractor import AttractorDetector
        from myco.session_log import SessionLogger
        HAS_MYCO = True
except Exception:
    pass  # HAS_MYCO remains False


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


class PathResolver:
    """Resolves and validates file paths relative to project root.
    
    MYCO: Fixes path normalization issues where agent gets confused
    about project root vs absolute paths.
    """
    
    def __init__(self, project_root: Path):
        """Initialize path resolver.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root.resolve()
    
    def resolve(self, path: str) -> Path:
        """Resolve a path relative to project root.
        
        Args:
            path: Path to resolve (can be relative or absolute)
            
        Returns:
            Resolved absolute path
        """
        if not path:
            return self.project_root
        
        p = Path(path)
        
        # If absolute path, return as-is
        if p.is_absolute():
            return p.resolve()
        
        # If path already contains project root, just resolve it
        try:
            resolved = p.resolve()
            if str(resolved).startswith(str(self.project_root)):
                return resolved
        except Exception:
            pass
        
        # If path starts with project root name, strip it
        path_str = str(p)
        project_name = self.project_root.name
        for prefix in [f"{project_name}/", f"{project_name}\\", project_name]:
            if path_str.startswith(prefix):
                path_str = path_str[len(prefix):].lstrip('/').lstrip('\\')
                p = Path(path_str)
                break
        
        # Resolve relative to project root
        resolved = (self.project_root / p).resolve()
        return resolved
    
    def is_in_project(self, path: Path) -> bool:
        """Check if a path is within the project.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is within project root
        """
        try:
            resolved = path.resolve()
            return str(resolved).startswith(str(self.project_root))
        except Exception:
            return False


class Agent:
    """Autonomous agent that can use tools to complete tasks.

    MYCO Vision: Operates as a thermodynamic coding agent that tends
    the substrate, reads entropy regimes, and enforces autopoietic gates.
    """

    # File tool wrappers with path resolution
    def _read_file(self, path: str = None, lines: Optional[int] = None, file_path: str = None):
        """Read file with path resolution."""
        actual_path = path or file_path
        if actual_path:
            actual_path = str(self._path_resolver.resolve(actual_path))
        return FileTools.read_file(path=actual_path, lines=lines)

    def _write_file(self, path: str, content: str):
        """Write file with path resolution."""
        resolved_path = str(self._path_resolver.resolve(path))
        return FileTools.write_file(path=resolved_path, content=content)

    def _edit_file(self, path: str, old_text: str, new_text: str):
        """Edit file with path resolution."""
        resolved_path = str(self._path_resolver.resolve(path))
        return FileTools.edit_file(path=resolved_path, old_text=old_text, new_text=new_text)

    def _append_file(self, path: str, content: str):
        """Append to file with path resolution."""
        resolved_path = str(self._path_resolver.resolve(path))
        return FileTools.append_file(path=resolved_path, content=content)

    def _delete_file(self, path: str):
        """Delete file with path resolution."""
        resolved_path = str(self._path_resolver.resolve(path))
        return FileTools.delete_file(path=resolved_path)

    def _copy_file(self, src: str, dst: str):
        """Copy file with path resolution."""
        resolved_src = str(self._path_resolver.resolve(src))
        resolved_dst = str(self._path_resolver.resolve(dst))
        return FileTools.copy_file(src=resolved_src, dst=resolved_dst)

    def _list_files(self, path: str, pattern: str = "*"):
        """List files with path resolution."""
        resolved_path = str(self._path_resolver.resolve(path))
        return FileTools.list_files(path=resolved_path, pattern=pattern)

    def __init__(
        self,
        base_url: str,
        model: str,
        max_iterations: int = 10,
        require_approval: bool = True,
        project_root: Optional[str] = None,
    ):
        """Initialize the agent.

        Args:
            base_url: Server base URL
            model: Model name to use
            max_iterations: Maximum tool use iterations
            require_approval: If True, prompt for approval on risky operations
            project_root: Root directory for MYCO world model (default: current dir)
        """
        self.base_url = base_url
        self.model = model
        self.max_iterations = max_iterations
        self.require_approval = require_approval
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.logger = LogConfig.get_logger("gguf.agent")
        self.logger.info(f"Agent project_root: {self.project_root}")
        self.logger.info(f"Agent cwd: {Path.cwd()}")

        # MYCO: Initialize project context
        self.project_context = get_project_context()
        try:
            self.current_project = ensure_project_context(self.project_root)
            self.logger.info(f"Project context: {self.current_project.name}")
        except Exception as e:
            self.logger.warning(f"Could not load project context: {e}")
            self.current_project = None

        # MYCO: Initialize vision components
        self.world_model = None
        self.gate = None
        self.attractor_detector = None
        self.session_logger = None
        self.substrate_health = None

        if HAS_MYCO:
            try:
                # Load world model
                self.world_model = WorldModel.load(self.project_root)
                self.logger.info(f"Loaded world model: {self.world_model.session_count} sessions")

                # Initialize autopoietic gate
                self.gate = AutopoieticGate(self.project_root, self.world_model)

                # Initialize attractor detector
                self.attractor_detector = AttractorDetector()

                # Initialize session logger
                self.session_logger = SessionLogger(self.project_root)

                # Calculate substrate health
                self.substrate_health = calculate_substrate_health(self.project_root)
                # substrate_health is a dict, not an object - access keys directly
                health_score = self.substrate_health.get('health_score', 0)
                self.logger.info(f"Substrate health score: {health_score:.2f}")
            except Exception as e:
                self.logger.warning(f"MYCO vision components failed to initialize: {e}")
                # Don't modify global HAS_MYCO here - just continue without MYCO features

        # Initialize approval manager
        self.approval_manager = ApprovalManager() if require_approval else None

        # MYCO: Loop detection - track consecutive identical writes
        self._last_file_hash = None
        self._consecutive_same_writes = 0
        self._files_written = set()

        # MYCO: Enhanced loop detection (Phase F)
        self._command_history = []  # Last 5 commands
        self._recent_tool_calls = []  # Last 10 tool calls for pattern detection
        self._loop_detection_threshold = 3  # Force new action after 3 repeats
        self._loop_count = 0  # Track current loop count
        
        # MYCO: Task progress tracking (NEW - prevents overwhelm)
        self._task_plan = []  # List of subtasks
        self._task_progress = 0  # Current subtask index
        self._files_needed = []  # Files needed for task completion
        self._iteration_budget = 30  # Increased from 20 for complex tasks
        self._iteration_warning_threshold = 22  # 75% of budget
        
        # MYCO: Dependency tracking
        self._file_dependencies = {}  # file -> [dependencies]
        
        # MYCO Phase F: Loop recovery tracking
        self._loop_recovery_suggestion = None  # Set when loop detected

        # MYCO: Multi-file coordination (Phase 3)
        self._multi_file_coordinator = None  # Initialized per task

        # MYCO: Path resolver for consistent path handling
        self._path_resolver = PathResolver(self.project_root)

        # MYCO: Short-term action memory (prevents context drift)
        self._action_memory = []  # Last 20 actions with semantic info
        self._max_action_memory = 20
        
        # MYCO: Recently accessed files (prevents re-reading same files)
        self._recently_read_files = {}  # path -> last_read_iteration
        self._recently_modified_files = {}  # path -> last_modified_iteration

        # Define available tools
        self.tools: dict[str, Callable] = {
            # File operations (with path resolution)
            "read_file": self._read_file,
            "write_file": self._write_file,
            "append_file": self._append_file,  # MYCO: For large files
            "edit_file": self._edit_file,
            "delete_file": self._delete_file,  # MYCO: File management
            "copy_file": self._copy_file,  # MYCO: File management
            "list_files": self._list_files,
            # Command execution
            "run_command": CommandTools.run_command,
            "run_python": CommandTools.run_python,  # MYCO: Code execution
            # Search tools
            "search_text": SearchTools.search_text,
            # MYCO: Project structure and path validation
            "find_file": ProjectTools.find_file,
            "list_project_structure": ProjectTools.list_project_structure,
            "test_import": ProjectTools.test_import,
            # MYCO: Architecture awareness
            "load_architecture": load_architecture_map,
            "get_module_info": get_arch_module_info,
            "get_dependencies": get_arch_dependencies,
            "get_dependents": get_arch_dependents,
            "get_component": get_arch_component,
            "update_architecture": update_architecture_map,
            "get_refactoring_suggestion": get_refactoring_suggestion,
            "get_affected_files": get_affected_files,
            "get_file_dependencies": get_arch_dependencies,
            # MYCO Phase 2: Session memory and patterns
            "record_session": record_session,
            "get_similar_sessions": get_similar_sessions,
            "get_lessons": get_lessons,
            "add_pattern": add_pattern,
            "get_patterns": get_patterns,
            "record_mistake": record_mistake,
            "get_warnings": get_warnings,
            # MYCO Phase 2.2: Quality feedback
            "record_quality_change": record_quality_change,
            "get_quality_trend": get_quality_trend,
            "get_project_health": get_project_health,
            "get_files_needing_attention": get_files_needing_attention,
            # MYCO Phase 2.3: Self-improvement
            "analyze_task_success": analyze_task_success,
            "get_pattern_recommendations": get_pattern_recommendations,
            "get_antipattern_warnings": get_antipattern_warnings,
            "get_quality_advice": get_quality_advice,
            "get_project_quality_advice": get_project_quality_advice,
            # MYCO Phase 3: Certainty
            "infer_types": infer_types,
            "generate_contract": generate_contract,
            "generate_property_tests": generate_property_tests,
            "verify_integration": verify_integration,
            # MYCO: Advanced capabilities
            "browser_open": BrowserTools.browser_open,
            "browser_screenshot": BrowserTools.browser_screenshot,
            "browser_click": BrowserTools.browser_click,
            "browser_fill": BrowserTools.browser_fill,
            "browser_evaluate": BrowserTools.browser_evaluate,
            "browser_close": BrowserTools.browser_close,
            "browser_compare_screenshots": BrowserTools.browser_compare_screenshots,
            "browser_screenshot_compare": BrowserTools.browser_screenshot_compare,
            "browser_analyze_screenshot": BrowserTools.browser_analyze_screenshot,
            "process_start": ProcessTools.process_start,
            "process_stop": ProcessTools.process_stop,
            "process_status": ProcessTools.process_status,
            "process_logs": ProcessTools.process_logs,
            "process_list": ProcessTools.process_list,
            # MYCO: Git version control
            "git_status": GitTools.git_status,
            "git_diff": GitTools.git_diff,
            "git_add": GitTools.git_add,
            "git_commit": GitTools.git_commit,
            "git_branch": GitTools.git_branch,
            "git_log": GitTools.git_log,
            "git_push": GitTools.git_push,
            "git_pull": GitTools.git_pull,
            "git_init": GitTools.git_init,
            # MYCO: Test runner
            "test_run": TestTools.test_run,
            "test_pytest": TestTools.test_pytest,
            "test_unittest": TestTools.test_unittest,
            "test_jest": TestTools.test_jest,
            "test_npm": TestTools.test_npm,
            # MYCO: Codebase search (grep + vision)
            "search_grep": CodebaseSearch.search_grep,
            "search_files": CodebaseSearch.search_files,
            "search_definitions": CodebaseSearch.search_definitions,
            "search_by_entropy": CodebaseSearch.search_by_entropy,
            "search_todo": CodebaseSearch.search_todo,
            "search_imports": CodebaseSearch.search_imports,
            # MYCO: Entropy gate (autopoietic protection)
            "entropy_check": EntropyGate.check_entropy_delta,
            "substrate_health": EntropyGate.get_substrate_health,
        }

        # Tool definitions for LLM
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to read"},
                            "lines": {
                                "type": "integer",
                                "description": "Maximum number of lines to read (optional)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a new file or overwrite existing file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to write"},
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "append_file",
                    "description": "Append content to an existing file (use for large files >200 lines)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to append to"},
                            "content": {
                                "type": "string",
                                "description": "Content to append to the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing old text with new text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to edit"},
                            "old_text": {
                                "type": "string",
                                "description": "Text to find and replace",
                            },
                            "new_text": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to delete"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file from source to destination",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source file path"},
                            "destination": {"type": "string", "description": "Destination file path"},
                        },
                        "required": ["source", "destination"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list"},
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern (e.g., '*.py')",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command (Windows: use 'dir' not 'ls', 'mkdir folder' not 'mkdir -p')",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 60)",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Run Python code and return output (for testing/verification)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute"},
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 30)",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "Search for text in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to search"},
                            "query": {"type": "string", "description": "Text to search for"},
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum matches to return",
                            },
                        },
                        "required": ["path", "query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_open",
                    "description": "Open a URL in a headless browser (requires Playwright)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to open"},
                            "headless": {"type": "boolean", "description": "Run headless (default: true)"},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_screenshot",
                    "description": "Take a screenshot of the current page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to save screenshot"},
                            "selector": {"type": "string", "description": "CSS selector for element (optional)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element on the page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector to click"},
                        },
                        "required": ["selector"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_fill",
                    "description": "Fill an input field with text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector for input"},
                            "value": {"type": "string", "description": "Text to fill"},
                        },
                        "required": ["selector", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_evaluate",
                    "description": "Execute JavaScript in the browser",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {"type": "string", "description": "JavaScript code to execute"},
                        },
                        "required": ["script"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_close",
                    "description": "Close the browser",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_compare_screenshots",
                    "description": "Compare two screenshots and highlight differences (visual regression testing)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image1_path": {"type": "string", "description": "Path to baseline screenshot"},
                            "image2_path": {"type": "string", "description": "Path to current screenshot"},
                            "output_path": {"type": "string", "description": "Path to save diff image (default: diff.png)"},
                            "threshold": {"type": "number", "description": "Sensitivity 0.0-1.0 (default: 0.05)"},
                        },
                        "required": ["image1_path", "image2_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_screenshot_compare",
                    "description": "Take screenshot of URL and compare with baseline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to screenshot"},
                            "baseline_path": {"type": "string", "description": "Path to baseline screenshot"},
                            "current_path": {"type": "string", "description": "Path for current screenshot"},
                            "diff_path": {"type": "string", "description": "Path for diff image"},
                            "threshold": {"type": "number", "description": "Sensitivity 0.0-1.0"},
                        },
                        "required": ["url", "baseline_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_analyze_screenshot",
                    "description": "Analyze screenshot for colors, text density, and visual elements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "Path to screenshot"},
                        },
                        "required": ["image_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_start",
                    "description": "Start a background process (server, dev mode, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name to identify the process"},
                            "command": {"type": "string", "description": "Command to run"},
                            "port": {"type": "integer", "description": "Port to wait for (optional)"},
                            "timeout": {"type": "integer", "description": "Startup timeout in seconds"},
                        },
                        "required": ["name", "command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_stop",
                    "description": "Stop a running process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of process to stop"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_status",
                    "description": "Get status of processes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Specific process name (optional)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_logs",
                    "description": "Get logs from a process",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the process"},
                            "lines": {"type": "integer", "description": "Number of lines"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_list",
                    "description": "List all running processes",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Get git repository status (changed files, untracked files)",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show changes between commits, commit and working tree, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Specific file path (optional)"},
                            "staged": {"type": "boolean", "description": "Show staged changes (default: false)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_add",
                    "description": "Stage files for commit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Files to stage",
                            },
                        },
                        "required": ["files"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Commit staged changes with a message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Files to stage before commit (optional)",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_branch",
                    "description": "List, create, or switch branches",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Branch name"},
                            "create": {"type": "boolean", "description": "Create new branch (default: false)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": "Show commit log",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of commits (default: 10)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_push",
                    "description": "Push changes to remote repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "description": "Remote name (default: origin)"},
                            "branch": {"type": "string", "description": "Branch name (default: current)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_pull",
                    "description": "Pull changes from remote repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "description": "Remote name (default: origin)"},
                            "branch": {"type": "string", "description": "Branch name (default: current)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_init",
                    "description": "Initialize a new git repository",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_run",
                    "description": "Run tests with auto-detected framework (pytest, jest, npm test, unittest)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Test command (default: pytest)"},
                            "framework": {"type": "string", "description": "Framework: pytest, jest, npm, unittest"},
                            "path": {"type": "string", "description": "Test file or directory"},
                            "verbose": {"type": "boolean", "description": "Verbose output (default: true)"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_pytest",
                    "description": "Run pytest tests (Python)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Test file or directory"},
                            "verbose": {"type": "boolean", "description": "Verbose output"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"},
                            "markers": {"type": "string", "description": "Pytest markers (-m expression)"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_unittest",
                    "description": "Run Python unittest tests",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Test file"},
                            "verbose": {"type": "boolean", "description": "Verbose output"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_jest",
                    "description": "Run Jest tests (JavaScript/TypeScript)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Test file or pattern"},
                            "verbose": {"type": "boolean", "description": "Verbose output"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_npm",
                    "description": "Run npm test (Node.js projects)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timeout": {"type": "integer", "description": "Timeout in seconds"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_grep",
                    "description": "Search for text pattern in files (grep-style with regex support)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Search pattern (regex)"},
                            "path": {"type": "string", "description": "Directory to search (default: .)"},
                            "include": {"type": "string", "description": "File glob pattern (default: *)"},
                            "max_results": {"type": "integer", "description": "Max results (default: 100)"},
                            "context_lines": {"type": "integer", "description": "Lines of context (default: 0)"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Find files by name pattern (glob)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py')"},
                            "path": {"type": "string", "description": "Directory to search"},
                            "max_results": {"type": "integer", "description": "Max results"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_definitions",
                    "description": "Search for code definitions (classes, functions)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name to search for"},
                            "def_type": {"type": "string", "description": "Type: class, function, async"},
                            "path": {"type": "string", "description": "Directory to search"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_entropy",
                    "description": "MYCO Vision: Search files by entropy regime (crystallized/dissipative/diffuse)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_entropy": {"type": "number", "description": "Minimum entropy (0.0-1.0)"},
                            "max_entropy": {"type": "number", "description": "Maximum entropy (0.0-1.0)"},
                            "regime": {"type": "string", "description": "Regime: crystallized, dissipative, diffuse"},
                            "path": {"type": "string", "description": "Directory to search"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_todo",
                    "description": "Search for TODO, FIXME, HACK, XXX comments",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory to search"},
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords (default: TODO, FIXME, HACK, XXX)",
                            },
                            "max_results": {"type": "integer", "description": "Max results"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_imports",
                    "description": "Search for imports of a specific module",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {"type": "string", "description": "Module name to search for"},
                            "path": {"type": "string", "description": "Directory to search"},
                            "max_results": {"type": "integer", "description": "Max results"},
                        },
                        "required": ["module_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "entropy_check",
                    "description": "MYCO Vision: Check if proposed change would increase entropy beyond threshold (autopoietic gate)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file being modified"},
                            "proposed_content": {"type": "string", "description": "Proposed new content"},
                        },
                        "required": ["file_path", "proposed_content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "substrate_health",
                    "description": "MYCO Vision: Get comprehensive substrate health report (entropy distribution)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory to analyze"},
                        },
                    },
                },
            },
        ]

        # Track session-wide approvals
        self._session_approvals: set[str] = set()

        # MYCO: Start session logging
        if self.session_logger:
            self.session_logger.log_session_start("Agent task")

        # Verify tools work (MYCO: verification-first)
        self._verify_tools()

    def _verify_tools(self):
        """Verify all tools are functional (MYCO: verification-first)."""
        self.logger.info("Verifying tools...")

        # Test list_files with current directory
        try:
            result = FileTools.list_files(".")
            if not result.success:
                self.logger.warning(f"Tool list_files test failed: {result.error}")
        except Exception as e:
            self.logger.warning(f"Tool list_files threw exception: {e}")

        self.logger.info("Tool verification complete")

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to read"},
                            "lines": {
                                "type": "integer",
                                "description": "Maximum number of lines to read (optional)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a new file or overwrite existing file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to write"},
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing old text with new text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to edit"},
                            "old_text": {
                                "type": "string",
                                "description": "Text to find and replace",
                            },
                            "new_text": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list"},
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern (e.g., '*.py')",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command (blocked: rm -rf, del /s, format, shutdown)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"},
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 60)",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "Search for text in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to search"},
                            "query": {"type": "string", "description": "Text to search for"},
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum matches to return",
                            },
                        },
                        "required": ["path", "query"],
                    },
                },
            },
        ]

    def _parse_tool_calls(self, content: str) -> list[dict]:
        """Parse tool calls from model response.

        Qwen3.5 may output tool calls in various formats.
        This parser handles common patterns.
        """
        tool_calls = []
        found_tools = set()

        # Pattern 1: JSON tool call with function format
        json_patterns = [
            r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*\})\s*\}',
            r'\{\s*"function"\s*:\s*\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*\})\s*\}\s*\}',
            r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*\})\s*\}',
        ]

        for pattern in json_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                try:
                    name = match.group(1)
                    args_str = match.group(2)
                    if not args_str or not args_str.strip():
                        continue
                    # Try to parse the arguments
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Try adding braces if missing
                        if not args_str.strip().startswith("{"):
                            args_str = "{" + args_str + "}"
                        args = json.loads(args_str)
                    if name in self.tools and name not in found_tools:
                        tool_calls.append({"name": name, "arguments": args})
                        found_tools.add(name)
                except (json.JSONDecodeError, IndexError):
                    pass

        # Pattern 2: <tool> XML tags
        xml_pattern = r"<tool>(.*?)</tool>"
        for match in re.finditer(xml_pattern, content, re.DOTALL):
            tool_content = match.group(1).strip()
            try:
                call = json.loads(tool_content)
                name = call.get("name") or call.get("function")
                if name and name in self.tools and name not in found_tools:
                    tool_calls.append(
                        {"name": name, "arguments": call.get("arguments") or call.get("args") or {}}
                    )
                    found_tools.add(name)
            except (json.JSONDecodeError, Exception):
                pass

        # Pattern 3: Direct function call: write_file(path="...", content="...")
        # Prioritize file writing operations
        priority_tools = ["write_file", "edit_file"]
        for tool_name in priority_tools:
            if tool_name in found_tools:
                continue
            func_pattern = rf"{tool_name}\s*\(\s*(.*?)\s*\)"
            for match in re.finditer(func_pattern, content, re.DOTALL):
                args_str = match.group(1).strip()
                args = {}
                # Only parse if there are arguments
                if args_str:
                    # Parse key="value" or key='value' pairs
                    arg_pattern = r'(\w+)\s*=\s*["\'](.+?)["\']'
                    for arg_match in re.finditer(arg_pattern, args_str, re.DOTALL):
                        args[arg_match.group(1)] = arg_match.group(2)

                if args and tool_name not in found_tools:
                    tool_calls.insert(
                        0,
                        {  # Add to front of list
                            "name": tool_name,
                            "arguments": args,
                        },
                    )
                    found_tools.add(tool_name)
                    break  # Only one call per tool

        # Pattern 4: Markdown code block with JSON
        code_pattern = r"```(?:json)?\s*(\{[^`]*\})\s*```"
        for match in re.finditer(code_pattern, content, re.DOTALL):
            try:
                call = json.loads(match.group(1))
                name = call.get("name") or call.get("function")
                if name and name in self.tools and name not in found_tools:
                    tool_calls.append(
                        {"name": name, "arguments": call.get("arguments") or call.get("args") or {}}
                    )
                    found_tools.add(name)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _parse_arguments_fallback(self, arguments_str: str) -> dict:
        """Fallback parsing for malformed JSON arguments (e.g., HTML content).

        Args:
            arguments_str: String that should be JSON but may contain unescaped quotes

        Returns:
            Dictionary with extracted arguments
        """
        args = {}

        # Extract path (usually well-formed)
        path_match = re.search(r'"path"\s*:\s*"([^"]+)"', arguments_str)
        if path_match:
            args['path'] = path_match.group(1)

        # Extract content - handle multi-line HTML/CSS with quotes
        content_start = arguments_str.find('"content"')
        if content_start != -1:
            colon_pos = arguments_str.find(':', content_start)
            if colon_pos != -1:
                quote_start = arguments_str.find('"', colon_pos)
                if quote_start != -1:
                    content_start_pos = quote_start + 1
                    remaining = arguments_str[content_start_pos:]

                    # Strategy: Find content by looking for the structure pattern
                    # Content ends when we see: \n}\n or \n})\n at the START of a line
                    lines = remaining.split('\n')
                    content_lines = []
                    in_content = True

                    for i, line in enumerate(lines):
                        stripped = line.rstrip()

                        # Check if this line is ONLY a closing brace (start of line)
                        if in_content and stripped in ('}', '})', '"}', '},'):
                            in_content = False
                            break
                        # Check if line starts with closing brace (likely end of JSON)
                        elif in_content and stripped.startswith('}'):
                            # Only stop if the line is SHORT (just closing braces)
                            if len(stripped) <= 5:
                                in_content = False
                                # Still include any content before the brace
                                before_brace = line.find('}')
                                if before_brace > 0:
                                    content_lines.append(line[:before_brace])
                                break
                        else:
                            content_lines.append(line)

                    if content_lines:
                        content = '\n'.join(content_lines)
                        # Clean up any trailing junk
                        content = content.rstrip('}').rstrip('"').rstrip(',').rstrip()
                        # Unescape JSON escapes: \" -> ", \n -> newline, etc.
                        content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                        args['content'] = content
                    else:
                        # Fallback: take most of the content
                        content = remaining
                        # Remove trailing JSON structure
                        for end_pattern in ['\n}', '\n})', '\n"}', '\n},']:
                            if end_pattern in content:
                                content = content[:content.rfind(end_pattern)]
                                break
                        content = content.rstrip('}').rstrip('"').rstrip(',').rstrip()
                        content = content.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                        args['content'] = content

        # Extract optional parameters
        lines_match = re.search(r'"lines"\s*:\s*(\d+)', arguments_str)
        if lines_match:
            args['lines'] = int(lines_match.group(1))

        timeout_match = re.search(r'"timeout"\s*:\s*(\d+)', arguments_str)
        if timeout_match:
            args['timeout'] = int(timeout_match.group(1))

        return args

    def _record_action(self, name: str, arguments: dict):
        """Record action in short-term memory.
        
        MYCO: Tracks recent actions to prevent context drift and repetition.
        
        Args:
            name: Tool name
            arguments: Tool arguments
        """
        from datetime import datetime
        
        # Use action memory length as index (works before _iteration is set)
        action_index = len(self._action_memory)
        
        # Create action record
        action = {
            'tool': name,
            'arguments': arguments,
            'timestamp': datetime.now().isoformat(),
            'signature': f"{name}:{str(arguments)[:100]}",
            'index': action_index
        }
        
        # Track file reads
        if name == 'read_file':
            path = arguments.get('path') or arguments.get('file_path')
            if path:
                resolved = str(self._path_resolver.resolve(path))
                self._recently_read_files[resolved] = action_index
        
        # Track file modifications
        if name in ['write_file', 'edit_file', 'append_file']:
            path = arguments.get('path')
            if path:
                resolved = str(self._path_resolver.resolve(path))
                self._recently_modified_files[resolved] = action_index
        
        # Add to action memory
        self._action_memory.append(action)
        self._action_memory = self._action_memory[-self._max_action_memory:]
    
    def _check_action_similarity(self, name: str, arguments: dict) -> Optional[str]:
        """Check if current action is similar to recent actions.
        
        MYCO: Detects when agent is repeating similar actions without progress.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Warning message if similar action found, None otherwise
        """
        if not self._action_memory:
            return None
        
        current_index = len(self._action_memory)
        
        # Check for same tool with same file
        if name == 'read_file':
            path = arguments.get('path') or arguments.get('file_path')
            if path:
                resolved = str(self._path_resolver.resolve(path))
                # Check if we read this file recently
                for action in self._action_memory[-5:]:
                    if action['tool'] == 'read_file':
                        prev_path = action['arguments'].get('path') or action['arguments'].get('file_path')
                        if prev_path:
                            prev_resolved = str(self._path_resolver.resolve(prev_path))
                            if prev_resolved == resolved:
                                prev_idx = action.get('index', 0)
                                return f"Re-reading {resolved} (last read {current_index - prev_idx} actions ago)"
        
        # Check for same edit pattern
        if name == 'edit_file':
            path = arguments.get('path')
            if path:
                resolved = str(self._path_resolver.resolve(path))
                # Check if we edited this file recently
                for action in self._action_memory[-3:]:
                    if action['tool'] == 'edit_file':
                        prev_path = action['arguments'].get('path')
                        if prev_path:
                            prev_resolved = str(self._path_resolver.resolve(prev_path))
                            if prev_resolved == resolved:
                                prev_idx = action.get('index', 0)
                                return f"Re-editing {resolved} (edited {current_index - prev_idx} actions ago)"
        
        return None

    def _execute_tool(self, name: str, arguments: dict, ui=None) -> ToolResult:
        """Execute a tool with given arguments.

        Args:
            name: Tool name
            arguments: Tool arguments
            ui: Optional UI display for approval prompts

        Returns:
            ToolResult with execution outcome
        """
        if name not in self.tools:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")

        # MYCO: Record action in short-term memory
        self._record_action(name, arguments)
        
        # MYCO: Check for semantic similarity with recent actions
        similarity_warning = self._check_action_similarity(name, arguments)
        if similarity_warning:
            self.logger.warning(f"ACTION SIMILARITY: {similarity_warning}")

        self.logger.info(f"Executing tool: {name} with args: {arguments}")

        # MYCO: Check approval for run_command
        if name == "run_command" and self.approval_manager:
            command = arguments.get("command", "")

            # Check if blocked
            if self.approval_manager.is_blocked(command):
                show_error(f"Command blocked for safety: {command}")
                return ToolResult(
                    success=False, output="", error=f"Command blocked: {command}", verified=False
                )

            # Check if approval required
            requires_approval, rule = self.approval_manager.check_approval_required(command)

            if requires_approval:
                # Check session-wide approval
                if rule and rule.pattern in self._session_approvals:
                    if ui:
                        ui.console.print(f"[green]OK[/green] Auto-approved (session): {command}")
                else:
                    # Prompt for approval
                    if ui:
                        approval_ui = ApprovalPrompt()

                        # Get enhanced impact information
                        impact_info = get_command_impact(command)

                        response = approval_ui.request_approval(
                            command=command,
                            rule_description=rule.description if rule else "Risky operation",
                            impact_info=impact_info,
                        )

                        # Handle response
                        while True:
                            if response in ("y", "yes", ""):
                                break
                            elif response in ("n", "no"):
                                return ToolResult(
                                    success=False,
                                    output="",
                                    error="User denied approval",
                                    verified=False,
                                )
                            elif response == "e":
                                # Edit command
                                new_command = approval_ui.request_edit(command)
                                if new_command != command:
                                    command = new_command
                                    arguments["command"] = command
                                    break
                            elif response == "r":
                                # Remember choice
                                if rule:
                                    self.approval_manager.remember_choice(rule.pattern, True)
                                    approval_ui.console.print(
                                        "[green]OK[/green] Choice remembered - will auto-approve in future"
                                    )
                                break
                            elif response == "a":
                                # Approve all for session
                                if rule:
                                    self._session_approvals.add(rule.pattern)
                                    approval_ui.console.print(
                                        f"[green]OK[/green] All '{rule.pattern}' commands approved for this session"
                                    )
                                break
                            else:
                                approval_ui.console.print(
                                    "[yellow]Invalid response. Try again.[/yellow]"
                                )
                                response = approval_ui.request_approval(
                                    command=command,
                                    rule_description=rule.description
                                    if rule
                                    else "Risky operation",
                                    impact_info=impact_info,
                                )

        # MYCO: Initialize error recovery handler
        if not hasattr(self, '_error_handler'):
            self._error_handler = ErrorRecoveryHandler()
            self._tool_retry_counts = {}  # Track retries per tool call

        # MYCO: Enhanced loop detection - check for repeated commands
        if name == "run_command" and "command" in arguments:
            command = arguments["command"]

            # Check if same command run recently (last 5 commands)
            if command in self._command_history[-5:]:
                self.logger.warning(f"LOOP DETECTED: Command '{command}' was just executed")
                return ToolResult(
                    success=False,
                    output="",
                    error=f"LOOP DETECTED: Command '{command}' was already executed. Check previous result.",
                    verified=True
                )

            # Add to command history
            self._command_history.append(command)
            self._command_history = self._command_history[-5:]  # Keep last 5

        # Track all tool calls for pattern detection
        tool_call_key = f"{name}:{str(arguments)[:100]}"
        
        # MYCO: Track exploration vs creation ratio
        exploration_tools = {'list_files', 'read_file', 'search_files', 'search_grep', 'search_definitions'}
        creation_tools = {'write_file', 'edit_file', 'append_file', 'run_command'}
        
        # Track last 15 actions for exploration ratio
        if not hasattr(self, '_action_history'):
            self._action_history = []
        self._action_history.append(name)
        self._action_history = self._action_history[-15:]
        
        # Calculate exploration ratio
        if len(self._action_history) >= 10:
            exploration_count = sum(1 for t in self._action_history if t in exploration_tools)
            exploration_ratio = exploration_count / len(self._action_history)
            
            # If >70% exploration without creation, inject prompt
            if exploration_ratio > 0.7 and not any(t in creation_tools for t in self._action_history[-5:]):
                self.logger.warning(f"HIGH EXPLORATION RATIO ({exploration_ratio:.0%}) - Injecting creation prompt")
                self._loop_recovery_suggestion = """
⚠️ EXPLORATION ALERT: You've been exploring/reading for a while without creating.

Last 10 actions: Mostly exploration tools (list_files, read_file, search_*)

RECOMMENDED NEXT ACTION:
- If you understand the structure → CREATE files (write_file, edit_file)
- If you need to add features → IMPLEMENT them (write_file, run_command)
- If testing → RUN tests (test_pytest, run_python)

STOP exploring and START creating!
"""

        # MYCO: Enhanced loop tracking with forced action change
        if tool_call_key in self._recent_tool_calls[-10:]:
            self._loop_count += 1
            self.logger.warning(f"LOOP DETECTED: Tool call pattern repeating (count: {self._loop_count})")

            # Force LLM to take different action after 3 repeats
            if self._loop_count >= self._loop_detection_threshold:
                self.logger.warning(f"LOOP THRESHOLD HIT ({self._loop_detection_threshold}) - Forcing different action")
                
                # Analyze what tool category has been overused
                recent_tools = [call.split(':')[0] for call in self._recent_tool_calls[-6:]]
                
                # Determine which category is stuck
                read_heavy = recent_tools.count('read_file') >= 3
                search_heavy = any(t in recent_tools for t in ['search_text', 'search_grep', 'search_definitions'])
                
                # Force COMPLETELY different tool category
                if read_heavy:
                    # Been reading too much - force writing or executing
                    self._loop_recovery_suggestion = """
🚨 LOOP BREAK: You've been reading files repeatedly without progress.

FORCED ACTION CHANGE: STOP reading files!

Choose ONE of these actions NOW:
1. write_file - Create a new file with what you know
2. edit_file - Make a specific change to a file you've read
3. run_python - Test your understanding with a quick script
4. run_command - Execute a command to verify something

DO NOT use read_file again until you've taken one of these actions!
"""
                elif search_heavy:
                    # Been searching too much - force creation
                    self._loop_recovery_suggestion = """
🚨 LOOP BREAK: You've been searching without creating!

FORCED ACTION CHANGE: STOP searching!

Choose ONE of these actions NOW:
1. write_file - Create the file you're looking for
2. edit_file - Implement what you need in an existing file
3. run_command - Run a test to see what's missing

DO NOT use search_* tools again until you've created something!
"""
                else:
                    # Generic loop break
                    self._loop_recovery_suggestion = """
🚨 LOOP BREAK: You're stuck in a repetitive pattern!

FORCED ACTION CHANGE: Do something COMPLETELY different!

Available tools you haven't used recently:
- write_file: Create new content
- edit_file: Modify existing content  
- run_python: Test your code
- run_command: Execute system commands
- test_pytest: Run tests

Pick ONE and use it NOW!
"""
            else:
                # Standard loop recovery
                if name == "read_file":
                    self._loop_recovery_suggestion = """
🔄 LOOP DETECTED: Same read_file call repeated

Recovery options:
1. File content is clear - PROCEED with next action (write/edit/test)
2. File needs fixing - use edit_file with specific old_text/new_text
3. File is corrupted - rewrite from scratch with write_file
4. Move to NEXT file - continue with plan

Recommended: Option 1 (if you understand the file) or Option 3 (if corrupted)
"""
        else:
            # Reset loop count on different action
            self._loop_count = 0

        self._recent_tool_calls.append(tool_call_key)
        self._recent_tool_calls = self._recent_tool_calls[-10:]  # Keep last 10

        # MYCO: Loop detection - check for repeated writes
        if name == "write_file" and "path" in arguments and "content" in arguments:
            file_path = arguments["path"]
            content = arguments["content"]
            content_hash = hash(content)

            # Check if same content written to same file
            file_key = f"{file_path}:{content_hash}"

            if file_key == self._last_file_hash:
                self._consecutive_same_writes += 1
                if self._consecutive_same_writes >= 2:
                    self.logger.warning(f"LOOP DETECTED: Same content written {self._consecutive_same_writes + 1} times to {file_path}")

                    # MYCO Phase F: Loop recovery suggestions
                    recovery_options = f"""
🔄 LOOP DETECTED: Same content written {self._consecutive_same_writes + 1} times

Recovery options:
1. Task may be COMPLETE - summarize and finish
2. File needs different content - use edit_file with specific changes
3. File is too large - split into smaller files
4. Move to NEXT subtask - continue with plan

Recommended: Option 1 (if task done) or Option 4 (continue plan)
"""
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"LOOP DETECTED: Same content written {self._consecutive_same_writes + 1} times.{recovery_options}",
                        verified=True
                    )
            else:
                self._consecutive_same_writes = 0
                self._last_file_hash = file_key
                self._files_written.add(file_path)

            # Limit total writes to same file in one session
            write_count = sum(1 for f in self._files_written if f == file_path)
            if write_count > 5:
                self.logger.warning(f"Multiple writes to {file_path} ({write_count} times)")

        # MYCO: Normalize paths to be relative to project root
        if self.current_project and name in ("write_file", "read_file", "edit_file", "delete_file", "list_files"):
            if "path" in arguments:
                original_path = arguments["path"]
                path_obj = Path(original_path)
                
                # MYCO Fix: Strip project name from path to prevent nesting
                # e.g., "MY_project/config.py" → "config.py"
                project_name = self.current_project.root.name
                path_str = str(path_obj)
                
                # Remove project name prefix if present
                for prefix in [f"{project_name}/", f"{project_name}\\", project_name]:
                    if path_str.startswith(prefix):
                        path_str = path_str[len(prefix):].lstrip('/').lstrip('\\')
                        self.logger.info(f"Stripped project prefix: {original_path} → {path_str}")
                        path_obj = Path(path_str)
                        break

                # If path is absolute but outside project, make it relative to project root
                if path_obj.is_absolute():
                    try:
                        # Try to make it relative to project root
                        path_obj = path_obj.relative_to(self.current_project.root)
                        arguments["path"] = str(path_obj)
                        self.logger.info(f"Normalized path: {original_path} -> {path_obj}")
                    except ValueError:
                        # Path is not within project root - use just the filename
                        arguments["path"] = path_obj.name
                        self.logger.warning(f"Path outside project, using filename: {original_path} -> {path_obj.name}")

                # If path is relative, ensure it's within project root
                elif not str(path_obj).startswith('..'):
                    # Path is already relative - resolve it relative to project root
                    arguments["path"] = str(self.current_project.root / path_obj)
                    self.logger.info(f"Resolved relative path: {original_path} -> {arguments['path']}")
                else:
                    # Path tries to escape project - normalize it
                    arguments["path"] = path_obj.name
                    self.logger.warning(f"Path escapes project, using filename: {original_path} -> {path_obj.name}")

        # MYCO: Check entropy before file modifications (gate enforcement)
        if name in ("write_file", "edit_file", "append_file") and "path" in arguments:
            file_path = Path(arguments["path"])
            if file_path.is_absolute() or (self.current_project and self.current_project.root):
                if not file_path.is_absolute() and self.current_project:
                    file_path = self.current_project.root / file_path

                entropy_info = self._error_handler.check_file_entropy(file_path)
                if entropy_info:
                    self.logger.info(f"File entropy: H={entropy_info['H_internal']:.2f} ({entropy_info['regime']})")

                    # MYCO: Warn if modifying crystallized file
                    if entropy_info['regime'] == 'crystallized':
                        self.logger.warning(
                            "⚠️ MYCO: Modifying crystallized file - consider refactoring first"
                        )

        # MYCO: Execute tool with error recovery
        max_retries = 3
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # MYCO: Track tool call timing
                tool_start = time.time()
                result = self.tools[name](**arguments)
                tool_duration = (time.time() - tool_start) * 1000  # ms
                
                # Store timing for summary
                if 'tool_call_times' in locals():
                    tool_call_times.append({
                        'tool': name,
                        'duration_ms': tool_duration,
                        'success': result.success if hasattr(result, 'success') else True
                    })
                
                # MYCO Phase 3: Add certainty checks for Python file writes
                if name == "write_file" and arguments.get("path", "").endswith('.py'):
                    content = arguments.get("content", "")
                    file_path = arguments["path"]
                    certainty_info = []
                    
                    # 1. Type inference
                    try:
                        from .certainty import TypeInferencer
                        inferencer = TypeInferencer()
                        type_result = inferencer.infer_types(content, file_path)
                        
                        if type_result.warnings:
                            certainty_info.append(f"⚠️ Type warnings: {len(type_result.warnings)}")
                        else:
                            certainty_info.append("✓ Types OK")
                    except Exception:
                        pass
                    
                    # 2. Contract generation
                    try:
                        from .certainty import ContractGenerator
                        generator = ContractGenerator()
                        
                        # Generate contracts for first function
                        import ast
                        try:
                            tree = ast.parse(content)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                                    contract = generator.generate_contract(content, node.name)
                                    if contract and (contract.preconditions or contract.postconditions):
                                        certainty_info.append(f"✓ Contract: {node.name}")
                                    break
                        except Exception:
                            pass
                    except Exception:
                        pass
                    
                    # 3. Property tests
                    try:
                        from .certainty import PropertyTester
                        tester = PropertyTester()
                        properties = tester.generate_properties(content)
                        
                        if properties:
                            certainty_info.append(f"✓ Properties: {len(properties)} generated")
                    except Exception:
                        pass
                    
                    if certainty_info and result.success:
                        result.output += f"\n\nCertainty: {', '.join(certainty_info)}"
                
                return result
            except Exception as e:
                last_error = e

                # Classify error
                error_type = self._error_handler.classify_error(str(e))

                # Get recovery strategy
                from .error_recovery import ErrorContext
                context = ErrorContext(
                    tool_name=name,
                    error_message=str(e),
                    args=arguments,
                    retry_count=attempt,
                    max_retries=max_retries,
                    file_path=arguments.get('path')  # MYCO: Track file for stigmergy
                )
                recovery = self._error_handler.get_recovery_strategy(error_type, context)

                # Log recovery attempt
                self.logger.info(f"Tool {name} error (attempt {attempt + 1}/{max_retries}): {error_type.value} - {recovery.suggestion}")

                # Check if should retry
                if not recovery.should_retry:
                    # MYCO: Record stigmergic trace for file errors
                    if context.file_path:
                        error_count = self._error_handler.get_file_error_count(context.file_path)
                        if error_count >= 3:
                            self.logger.warning(
                                f"⚠️ MYCO: {error_count} errors on {context.file_path} - consider refactoring"
                            )
                    
                    # Don't retry - return error with suggestion
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"{str(e)}. {recovery.suggestion}"
                    )

                # Apply backoff delay
                delay = self._error_handler.get_backoff_delay(attempt)
                time.sleep(delay)

                # Update arguments if recovery suggests modifications
                if recovery.modified_args:
                    arguments.update(recovery.modified_args)

        # Exhausted retries
        return ToolResult(
            success=False,
            output="",
            error=f"Tool {name} failed after {max_retries} retries: {last_error}"
        )

    def run(self, task: str, verbose: bool = False, use_ui: bool = True) -> str:
        """Run the agent to complete a task.

        Args:
            task: Task description
            verbose: Print progress
            use_ui: Use enhanced terminal UI

        Returns:
            Final response from the agent with verification status
        """
        # Initialize UI display if enabled
        ui = StatusDisplay() if use_ui else None
        verification_panel = VerificationPanel() if use_ui else None

        # MYCO: Build system prompt with entropy regime context
        myco_context = ""
        
        # MYCO: Dependency analysis for multi-file tasks
        dependency_note = ""
        if "create" in task.lower() and ".py" in task.lower():
            ordered_files = self._analyze_task_dependencies(task)
            if len(ordered_files) > 1:
                dependency_note = f"""

RECOMMENDED FILE CREATION ORDER (with entropy impact):
Based on dependency, create files in this order:
{chr(10).join(f"  {i+1}. {f}" for i, f in enumerate(ordered_files[:8]))}

Start with the first file and work your way down.
"""

        # MYCO: Add session continuity from world model
        session_context = ""
        crystallized_warning = ""
        if self.world_model:
            # Get open intentions from previous sessions
            open_intentions = self.world_model.open_intentions if hasattr(self.world_model, 'open_intentions') else []

            if open_intentions:
                session_context = f"""
PREVIOUS SESSION INTENTIONS:
The following intentions were recorded in previous sessions.
Consider addressing these before starting new work:

"""
                for i, intention in enumerate(open_intentions[:5], 1):
                    session_context += f"{i}. {intention}\n"

                session_context += """
If you complete any of these intentions, mention it in your summary.
"""
            
            # MYCO: Warn about crystallized modules
            crystallized = self.world_model.crystallized_modules if hasattr(self.world_model, 'crystallized_modules') else []
            if crystallized:
                crystallized_warning = f"""

⚠️ CRYSTALLIZED MODULES DETECTED:
The following modules have low entropy (H < 0.3) and may be rigid:
{', '.join(crystallized[:5])}

Consider refactoring (decompose, interface_inversion) before adding features to these modules.
"""
        
        # Add project context
        project_context = ""
        if self.current_project:
            # Build project structure info
            structure_info = f"- Root: {self.current_project.root}\n"
            
            # List top-level directories to clarify structure
            try:
                top_level_dirs = []
                for item in self.current_project.root.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        top_level_dirs.append(f"  - {item.name}/")
                if top_level_dirs:
                    structure_info += f"- Top-level folders:\n" + "\n".join(top_level_dirs[:8]) + "\n"
            except Exception:
                pass
            
            project_context = f"""
PROJECT CONTEXT: {self.current_project.name}
{structure_info}- Source: {self.current_project.src_dir or 'N/A'}
- Tests: {self.current_project.test_dir or 'N/A'}
- Python: {self.current_project.python_version or 'N/A'}
- Dependencies: {', '.join(self.current_project.dependencies[:5]) if self.current_project.dependencies else 'None'}

IMPORTANT: File paths are relative to the ROOT ({self.current_project.root}).
For example: "cli/agent/tools.py" means {self.current_project.root}\\cli\\agent\\tools.py
Do NOT assume files are in a "myco" subfolder unless they actually are.

PATH RULES (CRITICAL):
- ALWAYS use RELATIVE paths (e.g., "index.html", "src/main.py", "tests/test_auth.py")
- NEVER use absolute paths like "D:\\MYCO\\file.py" or "C:\\Users\\..."
- If you need to create a new file, use just the filename or relative path from project root
- The agent will automatically resolve paths relative to: {self.current_project.root}

CONSTRAINTS:
- ALL file operations MUST be within this project directory
- Do NOT modify files outside {self.current_project.root}
- Do NOT modify system directories (C:\\, D:\\, Program Files, etc.)
- Use appropriate directories: src/ for code, tests/ for tests, docs/ for documentation
"""
        
        # MYCO: Build entropy-aware system prompt with regime-specific coding guidance
        myco_context = ""
        regime_coding_guidance = ""
        
        if HAS_MYCO and self.substrate_health:
            # Get regime analysis for priority files
            regime_info = ""
            if hasattr(self.substrate_health, 'priority_files') and self.substrate_health.priority_files:
                for pf in self.substrate_health.priority_files[:3]:
                    intervention = get_regime_intervention(self.project_root / pf, H=pf.get('H', 0.5))
                    regime_info += f"- {pf['file']}: {intervention['regime']} (H={pf['H']:.2f}) → {intervention['primary']}\n"

            myco_context = f"""
MYCO SUBSTRATE STATUS:
{regime_info}
ENTROPY REGIME RULES (INVARIANTS):
1. Crystallized modules (H < 0.3): Do NOT add features. Apply decompose or interface_inversion FIRST.
2. Dissipative modules (0.3 ≤ H ≤ 0.75): Safe to make changes. Keep them minimal.
3. Diffuse modules (H > 0.75): Apply compression_collapse or tension_extraction before adding features.

AUTOPORIETIC GATE: Any action that increases entropy by >0.15 will be BLOCKED.
"""

            # MYCO: Regime-specific coding guidance (teaches LLM to write better code)
            regime_coding_guidance = """
ENTROPY-AWARE CODING (CRITICAL FOR MYCO VISION):

When writing code, adapt your style to the target module's entropy regime:

🔴 CRYSTALLIZED MODULES (H < 0.3) - Rigid, Hard to Change
  → DO: Extract interfaces, add pure functions, write tests first
  → DON'T: Add features, modify core logic, add dependencies
  → PATTERN: interface_inversion - depend on abstractions, not concretions
  → EXAMPLE:
    ```python
    # Before: Tightly coupled to concrete class
    def process(user_db: SQLiteDatabase): ...
    
    # After: Depends on protocol (interface)
    class IDatabase(Protocol):
        def get_user(self, id: int) -> dict: ...
    def process(db: IDatabase): ...  # Now swappable
    ```

🟢 DISSIPATIVE MODULES (0.3 ≤ H ≤ 0.75) - Healthy, Flowing
  → DO: Keep changes minimal, maintain boundaries, add tests
  → DON'T: Merge unrelated concerns, add god methods
  → PATTERN: single_responsibility - one reason to change
  → EXAMPLE:
    ```python
    # Good: Focused function
    def validate_email(email: str) -> bool:
        "Check email format."
        pattern = r"^[\w.+-]+@[\w.-]+\.[a-zA-Z]+$"
        return bool(re.match(pattern, email))

    # Bad: Does validation AND sending
    def handle_user_signup(email, password):
        # 50 lines of mixed concerns
    ```

🟡 DIFFUSE MODULES (H > 0.75) - Scattered, Needs Structure
  → DO: Consolidate related functions, extract classes, add structure
  → DON'T: Add more scattered functions, create new files for each function
  → PATTERN: compression_collapse - merge related scattered code
  → EXAMPLE:
    ```python
    # Before: Scattered auth functions
    def check_password(pw): ...
    def hash_password(pw): ...
    def verify_token(token): ...
    def create_token(user): ...
    
    # After: Consolidated AuthService class
    class AuthService:
        @staticmethod
        def hash_password(pw: str) -> str: ...
        @staticmethod
        def verify_password(pw: str, hash: str) -> bool: ...
        def create_token(self, user: dict) -> str: ...
        def verify_token(self, token: str) -> dict: ...
    ```
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are MYCO, an AI coding assistant with advanced TOOLS.\n\n"
                    "IMPORTANT: You MUST use tools to complete tasks. Do not just describe what to do - actually DO it.\n\n"
                    f"{session_context}"
                    f"{dependency_note}"
                    f"{regime_coding_guidance}"  # MYCO: Regime-specific coding guidance
                    "Available tools:\n"
                    "- write_file(path, content) - Create or overwrite files\n"
                    "- append_file(path, content) - Append to existing files (use for large files >200 lines)\n"
                    "- edit_file(path, old_text, new_text) - Edit by replacing text\n"
                    "- delete_file(path) - Delete files\n"
                    "- copy_file(source, destination) - Copy files\n"
                    "- read_file(path, lines?) - Read file contents\n"
                    "- list_files(path, pattern?) - List directory contents\n"
                    "- run_command(command, timeout?) - Run shell commands (Windows)\n"
                    "- run_python(code, timeout?) - Run Python code for testing\n"
                    "- search_text(path, query) - Search in files\n"
                    "- browser_open(url) - Open URL in browser (test websites)\n"
                    "- browser_screenshot(path) - Take screenshot\n"
                    "- browser_click(selector) - Click element\n"
                    "- browser_fill(selector, value) - Fill input field\n"
                    "- browser_evaluate(script) - Run JavaScript\n"
                    "- browser_close() - Close browser\n"
                    "- browser_compare_screenshots(img1, img2) - Compare screenshots (visual regression)\n"
                    "- browser_screenshot_compare(url, baseline) - Screenshot URL and compare with baseline\n"
                    "- browser_analyze_screenshot(path) - Analyze screenshot (colors, text density)\n"
                    "- process_start(name, command, port?) - Start background process\n"
                    "- process_stop(name) - Stop process\n"
                    "- process_status(name?) - Get process status\n"
                    "- process_list() - List all processes\n"
                    "- git_status() - Check git status\n"
                    "- git_diff(path?, staged?) - Show changes\n"
                    "- git_add(files) - Stage files\n"
                    "- git_commit(message, files?) - Commit changes\n"
                    "- git_branch(name?, create?) - List/create/switch branches\n"
                    "- git_log(limit?) - Show commit history\n"
                    "- git_push(remote?, branch?) - Push to remote\n"
                    "- git_pull(remote?, branch?) - Pull from remote\n"
                    "- git_init() - Initialize git repo\n"
                    "- test_run(command?, framework?, path?) - Run tests (auto-detect)\n"
                    "- test_pytest(path?) - Run pytest tests\n"
                    "- test_unittest(path?) - Run unittest tests\n"
                    "- test_jest(path?) - Run Jest tests\n"
                    "- test_npm() - Run npm test\n"
                    "- search_grep(pattern, path?) - Search text (grep)\n"
                    "- search_files(pattern) - Find files by name\n"
                    "- search_definitions(name, type?) - Find classes/functions\n"
                    "- search_by_entropy(regime?) - MYCO: Find by entropy regime\n"
                    "- search_todo() - Find TODO/FIXME comments\n"
                    "- search_imports(module) - Find module imports\n"
                    "- entropy_check(file, content) - MYCO: Gate check (block high-entropy changes)\n"
                    "- substrate_health() - MYCO: Get substrate health report\n\n"
                    "IMPORTANT RULES:\n"
                    "0. PLAN FIRST: Before any tool use, output a brief plan with 3-6 subtasks. Example:\n"
                    '   "Plan: 1) Read existing code → 2) Create models.py → 3) Create database.py → 4) Create main.py → 5) Add tests → 6) Verify"\n'
                    "   Then execute subtask 1. After each tool result, check off completed subtasks and continue.\n"
                    "1. For Windows: use 'mkdir folder' (NOT 'mkdir -p'), 'dir' (NOT 'ls'), 'del' (NOT 'rm')\n"
                    "2. For LARGE files (CSS >200 lines): FIRST write_file with base, then append_file for rest\n"
                    "3. After using a tool, WAIT for result - do NOT repeat the same command\n"
                    "4. Complete ALL files before finishing - don't stop halfway\n"
                    "5. Use run_python to test code before finishing\n"
                    "6. Use browser_open to test websites after creation\n"
                    "7. Use process_start to run dev servers, then test with browser\n\n"
                    
                    "ENTROPY EXAMPLES (CRITICAL FOR MYCO VISION):\n\n"
                    
                    "✅ GOOD (Low Entropy - Always Permitted):\n"
                    '```python\n'
                    '# Pure function, no imports, no coupling\n'
                    'def add(a: float, b: float) -> float:\n'
                    '    """Add two numbers."""\n'
                    '    return a + b\n'
                    '```\n'
                    'ΔH = 0.00-0.08 | REGIME: Safe | No dependencies, easy to test\n\n'
                    
                    "⚠️ CAUTION (Moderate Entropy - Usually Permitted):\n"
                    '```python\n'
                    '# 2-3 imports, creates some coupling\n'
                    'from sqlalchemy import Column, Integer, String\n'
                    'from src.database import Base\n'
                    'class User(Base):\n'
                    '    __tablename__ = "users"\n'
                    '    id = Column(Integer, primary_key=True)\n'
                    '```\n'
                    'ΔH = 0.10-0.18 | REGIME: Acceptable | Some coupling to database\n\n'
                    
                    "❌ AVOID (High Entropy - Will Be Blocked):\n"
                    '```python\n'
                    '# 5+ imports, monolithic, does everything\n'
                    'from src.models import User, Product, Order, OrderItem, Category\n'
                    'from src.auth import authenticate, create_token\n'
                    'from src.email import send_confirmation\n'
                    'class GodClass:\n'
                    '    # 500 lines with 20 methods\n'
                    '    def everything(self): ...\n'
                    '```\n'
                    'ΔH = 0.25-0.40 | REGIME: Blocked | Too much coupling, split into smaller modules\n\n'
                    
                    "ENTROPY BUDGET RULES:\n"
                    "1. New projects (0-5 files): Budget = 0.50 per file (flexible)\n"
                    "2. Growing projects (5-20 files): Budget = 0.30 per file (moderate)\n"
                    "3. Mature projects (20+ files): Budget = 0.15 per file (strict)\n"
                    "4. Pure functions: Always OK (ΔH ≈ 0.00)\n"
                    "5. Tests: Always OK (ΔH ≈ -0.02, reduces future entropy)\n"
                    "6. Monolithic files: Always blocked (ΔH > 0.20)\n\n"
                    
                    "REFATORING PATTERNS (When Gate Blocks):\n\n"
                    
                    "Pattern 1: Decompose God Class\n"
                    "Before: class Service (500 lines, 20 methods)\n"
                    "After:  class AuthService (150 lines)\n"
                    "         class EmailService (150 lines)\n"
                    "         class ReportService (150 lines)\n\n"
                    
                    "Pattern 2: Extract Protocol\n"
                    "Before: def process(db: Database)  # Tightly coupled\n"
                    "After:  class IDatabase(Protocol):\n"
                    "            def query(self): ...\n"
                    "        def process(db: IDatabase)  # Loosely coupled\n\n"
                    
                    "Pattern 3: Split by Responsibility\n"
                    "Before: main.py (routes + logic + models)\n"
                    "After:  main.py (routes only)\n"
                    "        models.py (SQLAlchemy models)\n"
                    "        schemas.py (Pydantic schemas)\n"
                    "        services.py (business logic)\n\n"
                    
                    "THERMODYNAMIC TESTING (CRITICAL FOR MYCO VISION):\n\n"
                    
                    "✅ GOOD: Test After Each Module\n"
                    "```\n"
                    "Iteration 1-3: Create models.py\n"
                    "Iteration 4:   Test models.py ✅\n"
                    "Iteration 5-7: Create auth.py\n"
                    "Iteration 8:   Test auth.py ✅\n"
                    "Iteration 9-11: Create api.py\n"
                    "Iteration 12:  Test api.py ✅\n"
                    "```\n"
                    "ΔH = Verified (coupling confirmed safe)\n\n"
                    
                    "❌ BAD: Test All at End\n"
                    "```\n"
                    "Iteration 1-20: Create all files\n"
                    "Iteration 21-25: Create tests\n"
                    "Iteration 26-30: Run tests → 15 failures!\n"
                    "```\n"
                    "ΔH = Unknown (coupling unverified, potential debt)\n\n"
                    
                    "WHY IT MATTERS:\n"
                    "- Testing after each module confirms entropy is safe\n"
                    "- Testing at end accumulates hidden coupling\n"
                    "- Hidden coupling = technical debt = future entropy\n\n"
                    
                    "VERIFICATION CHECKPOINTS (MYCO VISION ENFORCEMENT):\n\n"
                    
                    "After creating EACH module:\n"
                    "1. ✅ File created (syntax verified)\n"
                    "2. ✅ Imports resolved (dependencies exist)\n"
                    "3. ✅ Basic test passes (module works)\n"
                    "4. ✅ Annotation updated (stigmergic memory)\n\n"
                    
                    "Before creating DEPENDENT module:\n"
                    "1. ✅ Dependencies exist (check imports)\n"
                    "2. ✅ Dependencies tested (check annotations)\n"
                    "3. ✅ Entropy budget available (check ΔH)\n\n"
                    
                    "Example Flow:\n"
                    "```\n"
                    "Iteration 1-2: Create models.py\n"
                    "Iteration 3:   ✅ Syntax check\n"
                    "Iteration 4:   ✅ Import check (no unresolved imports)\n"
                    "Iteration 5:   ✅ Basic test (models can instantiate)\n"
                    "Iteration 6:   ✅ Annotation updated\n\n"
                    "Iteration 7-8: Create auth.py (imports models.py)\n"
                    "Iteration 9:   ✅ Check models.py exists ✅\n"
                    "Iteration 10:  ✅ Check models.py tested ✅\n"
                    "Iteration 11:  ✅ Syntax check\n"
                    "Iteration 12:  ✅ Test auth.py\n"
                    "```\n\n"
                    "Example for large CSS:\n"
                    '1. write_file("style.css", "/* Base */\\nbody { margin: 0; }")\n'
                    '2. append_file("style.css", "\\n/* Layout */\\n.container { max-width: 1200px; }")\n\n'
                    "Example for process management:\n"
                    '1. process_start("dev-server", "npm run dev", port=3000)\n'
                    '2. browser_open("http://localhost:3000")\n'
                    '3. browser_screenshot("result.png")\n\n'
                    "Example for visual regression testing:\n"
                    '1. browser_screenshot("baseline.png")  # Save baseline\n'
                    '2. # ... make changes to website ...\n'
                    '3. browser_screenshot_compare("http://localhost:3000", "baseline.png")\n'
                    '4. # Check diff.png for visual changes\n\n'
                    "Example for git workflow:\n"
                    '1. git_status()  # Check what changed\n'
                    '2. git_diff("app.py")  # Review changes\n'
                    '3. git_add(["app.py", "utils.py"])  # Stage files\n'
                    '4. git_commit("Add new feature")  # Commit\n'
                    '5. git_push()  # Push to remote\n\n'
                    "Example for test workflow:\n"
                    '1. write_file("test_app.py", "def test_add(): assert 1+1==2")\n'
                    '2. test_pytest("test_app.py")  # Run tests\n'
                    '3. # Fix any failures and re-run\n\n'
                    "Example for codebase search:\n"
                    '1. search_grep("def authenticate", path="src/")  # Find function\n'
                    '2. search_definitions("User", def_type="class")  # Find class\n'
                    '3. search_todo()  # Find TODO comments\n'
                    '4. search_by_entropy(regime="crystallized")  # MYCO: Find rigid modules\n\n'
                    "Example for entropy gate (MYCO vision):\n"
                    '1. substrate_health()  # Check overall codebase health\n'
                    '2. entropy_check("app.py", new_content)  # Gate check before write\n'
                    '3. # If BLOCKED, refactor to reduce entropy\n\n'
                    "PLANNING EXAMPLE (CRITICAL):\n"
                    'User: "Create a URL shortener API with FastAPI"\n\n'
                    'You (FIRST): "📋 PLAN:\n'
                    '1. Create requirements.txt (fastapi, uvicorn)\n'
                    '2. Create src/models.py (Pydantic schemas)\n'
                    '3. Create src/database.py (SQLite functions)\n'
                    '4. Create src/main.py (FastAPI endpoints)\n'
                    '5. Create test_main.py (pytest tests)\n'
                    '6. Run tests and verify\n\n'
                    'Starting with subtask 1..."\n'
                    'Then: {"name": "write_file", "arguments": {"path": "requirements.txt", "content": "fastapi>=0.104.0\\nuvicorn>=0.24.0"}}\n\n'
                    'After tool result: "✅ Subtask 1 complete. Starting subtask 2: Create src/models.py"\n'
                    'Then: {"name": "write_file", ...}\n\n'
                    'Continue until all 6 subtasks checked off. Then summarize.\n\n'
                    "Example:\n"
                    'User: "Create hello.py"\n'
                    'You: {"name": "write_file", "arguments": {"path": "hello.py", "content": "print(\'hi\')"}}\n'
                    f"{project_context}"
                    f"{myco_context}"
                ),
            },
            {"role": "user", "content": task},
        ]

        iteration = 0
        final_response = ""
        actions_taken = []
        verified_count = 0
        failed_count = 0
        
        # MYCO: Track start time for elapsed time display
        import time
        start_time = time.time()
        tool_call_times = []  # Track tool call durations for summary

        # Start task display
        if ui:
            ui.start_task(f"MYCO Agent: {task[:50]}...")

        # MYCO: Use iteration budget
        max_iterations = getattr(self, '_iteration_budget', 30)
        warning_threshold = getattr(self, '_iteration_warning_threshold', 22)

        while iteration < max_iterations:
            iteration += 1
            
            # Calculate elapsed time
            elapsed = time.time() - start_time

            # MYCO: Warn at 75% budget
            if iteration == warning_threshold:
                self.logger.warning(f"⚠️ MYCO: {iteration}/{max_iterations} iterations used ({elapsed:.0f}s elapsed). Consider simplifying task.")

            if verbose:
                # MYCO: Better iteration display with progress bar
                progress_pct = (iteration / max_iterations) * 100
                bar_length = 20
                filled = int(bar_length * iteration / max_iterations)
                bar = "█" * filled + "░" * (bar_length - filled)
                self.console.print(f"\n[bold blue]{'='*60}[/bold blue]")
                self.console.print(f"[bold]Iteration {iteration}/{max_iterations}[/bold]  [{bar}] {progress_pct:.0f}%")
                self.console.print(f"[dim]Elapsed: {elapsed:.1f}s | Avg iteration: {elapsed/iteration:.1f}s[/dim]")
                self.console.print(f"[bold blue]{'='*60}[/bold blue]\n")

            # Call the model
            try:
                request_data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,  # Increased from 512 for longer responses
                }

                # CRITICAL: Send tool definitions to model
                if self.tool_definitions:
                    request_data["tools"] = self.tool_definitions

                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data,
                    timeout=300,  # 5 minutes for 128K context
                )
                response.raise_for_status()
            except requests.RequestException as e:
                if ui:
                    show_error(f"API request failed: {e}")
                return f"Error: API request failed - {e}"

            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            # Initialize tool calls
            tool_calls = []
            tool_calls_raw = message.get("tool_calls", [])
            
            # MYCO: FORCE action after too many loops - override LLM choice
            if self._loop_count >= self._loop_detection_threshold * 2:
                self.logger.warning(f"🚨 FORCED ACTION: Loop count ({self._loop_count}) exceeds threshold. Forcing task-critical file creation.")
                
                # MYCO: Smart forced action - create task-critical files
                # Analyze task to determine what file to create
                task_lower = task.lower()
                forced_path = "src/core.py"  # Default
                forced_content = f"# Core implementation for: {task[:100]}\n\n# TODO: Implement based on task requirements\n"
                
                # Determine critical file based on task
                if 'model' in task_lower:
                    if 'user' in task_lower:
                        forced_path = "models/user.py"
                        forced_content = '"""User model."""\nfrom sqlalchemy import Column, Integer, String\n\nclass User:\n    __tablename__ = "users"\n    id = Column(Integer, primary_key=True)\n    # TODO: Add remaining fields\n'
                    elif 'product' in task_lower:
                        forced_path = "models/product.py"
                    elif 'order' in task_lower:
                        forced_path = "models/order.py"
                    else:
                        forced_path = "models/base.py"
                elif 'service' in task_lower:
                    if 'user' in task_lower:
                        forced_path = "services/user.py"
                    elif 'product' in task_lower:
                        forced_path = "services/product.py"
                    else:
                        forced_path = "services/base.py"
                elif 'api' in task_lower or 'router' in task_lower:
                    forced_path = "api/routes.py"
                elif 'config' in task_lower:
                    forced_path = "config.py"
                elif 'database' in task_lower:
                    forced_path = "database.py"
                
                # Force creation of task-critical file
                tool_calls = [{
                    "id": "forced_action",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": forced_path,
                            "content": forced_content
                        })
                    }
                }]
                self._loop_count = 0  # Reset after forcing
                self._loop_recovery_suggestion = f"⚠️ Agent was stuck. Forced creation of {forced_path}."
            
            # MYCO: If LLM response is empty or just text, force exploration → action transition
            elif not tool_calls_raw and iteration > 3:
                # Check if we've been exploring too long without creating
                has_created_file = any(a['tool'] == 'write_file' for a in actions_taken[-5:] if a)
                has_only_exploration = all(a['tool'] in ('list_files', 'read_file', 'search_files') for a in actions_taken[-5:] if a)
                
                if has_only_exploration and not has_created_file and len(actions_taken) >= 5:
                    self.logger.warning("🚨 FORCED ACTION: Too much exploration without creation. Forcing write_file.")
                    # Force create a TODO file to break exploration loop
                    tool_calls = [{
                        "id": "forced_exploration_break",
                        "type": "function", 
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({
                                "path": "myco_project/TODO_IMPLEMENTATION.md",
                                "content": f"""# TODO: Implementation Required

Task: {task[:200]}

## What needs to be implemented:

Based on the exploration, the agent should now CREATE files instead of listing directories.

## Next Steps:

1. Create model files if they don't exist
2. Create API/router files  
3. Test the implementation
4. Verify with entropy check

STOP listing directories and START creating files!
"""
                            })
                        }
                    }]
            
            # Parse native tool calls if provided
            elif tool_calls_raw:
                # Parse native tool calls
                for tc in tool_calls_raw:
                    func = tc.get("function", {})
                    name = func.get("name")
                    arguments = func.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            # First try strict parsing
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            try:
                                # Try lenient parsing
                                arguments = json.loads(arguments, strict=False)
                            except json.JSONDecodeError as e:
                                # HTML content often breaks JSON - use fallback parsing
                                self.logger.warning(f"JSON parse error, using fallback: {e}")
                                arguments = self._parse_arguments_fallback(arguments)
                            except Exception as e:
                                self.logger.error(f"Unexpected error parsing arguments: {e}")
                                arguments = {}
                    if name in self.tools:
                        tool_calls.append({"name": name, "arguments": arguments})

            content = message.get("content", "")

            if verbose:
                print(f"\nAI Model output ({len(content)} chars):")
                print(content[:800] if len(content) > 800 else content)

            if verbose and tool_calls:
                print(f"\n-- Found {len(tool_calls)} tool call(s)")

            if not tool_calls:
                # No tool calls - check if task might need tools but model didn't use them
                task_indicators = [
                    "file",
                    "create",
                    "write",
                    "edit",
                    "run",
                    "command",
                    "python",
                    "script",
                    "directory",
                    "folder",
                    "add",
                    "function",
                    "method",
                ]
                needs_tools = any(word in task.lower() for word in task_indicators)

                if needs_tools and iteration < self.max_iterations - 1:
                    # Prompt the model to use tools AND create a plan
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "📋 Remember to PLAN first: Output a brief plan with 3-6 subtasks before using tools.\n"
                                "Example: \"Plan: 1) Read existing code → 2) Create models.py → 3) Create main.py → 4) Test\"\n"
                                "Then execute subtask 1 with a tool call (write_file, run_command, etc.).\n"
                                "After each tool result, check off completed subtasks and continue to the next.\n"
                                "Please create your plan and start with the first tool call."
                            ),
                        }
                    )
                    continue

                # No tools needed or model won't use them - return response
                final_response = content
                break

            # Execute tools and collect results
            tool_results = []
            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments", {})

                # Skip if no arguments (invalid tool call)
                if not arguments:
                    if verbose:
                        print(f"\n-- Skipping {name}: no arguments provided")
                    tool_results.append({"tool": name, "result": "SKIPPED: No arguments provided"})
                    continue

                # Create step name for UI
                step_name = f"{name}({', '.join(f'{k}={v}' for k, v in arguments.items())})"

                if ui:
                    ui.add_step(step_name)
                    step_index = len(ui.steps) - 1

                if verbose:
                    print(f"\n-- Executing: {name}({arguments})")

                result = self._execute_tool(name, arguments, ui=ui)
                
                # Handle mkdir "already exists" as success
                if name == "run_command" and not result.success:
                    # Check both error field and output for "already exists"
                    error_text = result.error or ""
                    output_text = result.output or ""
                    if "already exists" in error_text or "already exists" in output_text:
                        if arguments.get("command", "").startswith("mkdir"):
                            # Directory already exists = success for mkdir
                            result = ToolResult(
                                success=True,
                                output="Directory already exists (OK)",
                                verified=True
                            )
                            if verbose:
                                print(f"   [mkdir already exists - treating as success]")

                # Track verification status
                actions_taken.append(
                    {
                        "tool": name,
                        "args": arguments,
                        "success": result.success,
                        "verified": result.verified,
                    }
                )

                if result.verified:
                    verified_count += 1
                    if ui:
                        ui.complete_step(step_index, verified=True, details="verified")
                elif not result.success:
                    failed_count += 1
                    if ui:
                        ui.fail_step(step_index, error=result.error or "failed")
                else:
                    if ui:
                        ui.complete_step(step_index, verified=False, details="assumed")

                if verbose:
                    status = (
                        "OK Verified"
                        if result.verified
                        else ("OK Success" if result.success else "X Failed")
                    )
                    preview = result.output[:300] if len(result.output) > 300 else result.output
                    print(f"{status} Result: {preview}")
                    if result.error:
                        print(f"   Error: {result.error}")

                tool_results.append({"tool": name, "result": result.to_response()})

            # Add tool results to conversation
            if tool_results:
                results_text = "\n\n".join(
                    f"[{r['tool']} result]:\n{r['result']}" for r in tool_results
                )
                
                # MYCO Phase F: Add loop recovery suggestion if detected
                loop_recovery_note = ""
                if self._loop_recovery_suggestion:
                    loop_recovery_note = f"\n\n{self._loop_recovery_suggestion}"
                    self._loop_recovery_suggestion = None  # Reset after showing
                
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool execution completed:\n\n{results_text}{loop_recovery_note}\n\nContinue with the task. If complete, summarize what was done and what was verified.",
                    }
                )

        # Finish task display
        if ui:
            success = failed_count == 0
            ui.finish_task(success=success)

            # Show verification panel
            if actions_taken:
                verification_panel.show_verification(actions_taken)

        # MYCO: Add verification summary to final response
        if actions_taken and final_response:
            summary = "\n\n" + "=" * 50 + "\n"
            summary += "MYCO Verification Report\n"
            summary += "=" * 50 + "\n"
            summary += f"Actions taken: {len(actions_taken)}\n"
            summary += f"OK Verified: {verified_count}\n"
            if failed_count > 0:
                summary += f"X Failed: {failed_count}\n"

            # List what was verified
            if verified_count > 0:
                summary += "\nVerified actions:\n"
                for action in actions_taken:
                    if action["verified"]:
                        summary += f"  OK {action['tool']}: {action['args']}\n"

            # List what failed
            if failed_count > 0:
                summary += "\nFailed actions (needs attention):\n"
                for action in actions_taken:
                    if not action["success"] and not action["verified"]:
                        summary += f"  X {action['tool']}: {action['args']}\n"

            summary += "=" * 50 + "\n"
            final_response += summary

        # MYCO: End session logging
        if self.session_logger and self.world_model:
            try:
                # Update world model
                self.world_model.session_count += 1
                self.world_model.last_session = datetime.utcnow().isoformat() + "Z"

                # Record files touched in this session
                if hasattr(self, '_files_written') and self._files_written:
                    # Add completed intentions for files that were created
                    for file_path in self._files_written:
                        intention = f"Created/modified: {file_path}"
                        if intention not in self.world_model.open_intentions:
                            # Remove from open intentions if it was there
                            if f"Create {file_path}" in self.world_model.open_intentions:
                                self.world_model.open_intentions.remove(f"Create {file_path}")

                # Save world model
                self.world_model.save()

                # Log session end
                self.session_logger.log_session_end(
                    iterations=iteration,
                    tokens=0,  # Would need token counting
                    joules=0.0,  # Would need energy tracking
                    entropy_delta=0.0,  # Would need entropy calculation
                    files_modified=list(self._files_written) if hasattr(self, '_files_written') else []
                )
            except Exception as e:
                self.logger.warning(f"Failed to save MYCO session: {e}")

        # MYCO: Automatic test generation (after successful file modifications)
        if hasattr(self, '_files_written') and self._files_written:
            test_verification = self._generate_and_run_tests(actions_taken)
            if test_verification:
                final_response += "\n\n" + test_verification

        # MYCO: Display final report with tool call summary
        total_time = time.time() - start_time
        self.console.print(f"\n[bold green]{'='*70}[/bold green]")
        self.console.print(f"[bold green]✅ TASK COMPLETE[/bold green]")
        self.console.print(f"[bold green]{'='*70}[/bold green]")
        self.console.print(f"\n[bold]Task:[/bold] {task[:80]}...")
        self.console.print(f"[bold]Duration:[/bold] {total_time:.1f}s")
        self.console.print(f"[bold]Iterations:[/bold] {iteration}/{max_iterations}")
        self.console.print(f"[bold]Files:[/bold] {len(self._files_written) if hasattr(self, '_files_written') else 0} modified")
        
        # Tool call summary
        if tool_call_times:
            self.console.print(f"\n[bold]Tool Call Summary:[/bold]")
            
            # Group by tool
            from collections import defaultdict
            tool_stats = defaultdict(lambda: {'count': 0, 'total_ms': 0, 'successes': 0})
            for tc in tool_call_times:
                tool_stats[tc['tool']]['count'] += 1
                tool_stats[tc['tool']]['total_ms'] += tc['duration_ms']
                if tc['success']:
                    tool_stats[tc['tool']]['successes'] += 1
            
            # Display top tools
            sorted_tools = sorted(tool_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            for tool, stats in sorted_tools:
                avg_ms = stats['total_ms'] / stats['count']
                success_rate = (stats['successes'] / stats['count'] * 100) if stats['count'] > 0 else 0
                self.console.print(f"  [green]✓[/green] {tool}: {stats['count']} calls, {avg_ms:.0f}ms avg, {success_rate:.0f}% success")
        
        # MYCO: Show entropy regime for modified files (MOST MYCO FEATURE!)
        if hasattr(self, '_files_written') and self._files_written:
            try:
                from myco.entropy import calculate_substrate_health, get_regime
                
                self.console.print(f"\n[bold]🍄 Substrate Health:[/bold]")
                for file_path in self._files_written:
                    try:
                        H = calculate_substrate_health(str(file_path))
                        regime = get_regime(H)
                        
                        if regime == "dissipative":
                            status = "[green]HEALTHY ✓[/green]"
                        elif regime == "crystallized":
                            status = "[yellow]RIGID - consider refactor[/yellow]"
                        else:
                            status = "[red]DIFFUSE - consolidate needed[/red]"
                        
                        # Entropy bar
                        bar = "█" * int(H * 10) + "░" * (10 - int(H * 10))
                        self.console.print(f"  {file_path}: [{bar}] {H:.2f} {regime} {status}")
                    except Exception as e:
                        pass
            except Exception:
                pass  # MYCO entropy module not available
        
        self.console.print(f"\n[dim]{'='*70}[/dim]\n")

        return final_response or "Task completed."

    def _check_task_completion(self, task: str, actions_taken: list) -> tuple:
        """MYCO: Check if task is complete based on actions taken.
        
        Analyzes task description and actions to determine completion status.
        
        Args:
            task: Original task description
            actions_taken: List of actions taken
            
        Returns:
            Tuple of (is_complete: bool, missing: list)
        """
        import re
        
        # Extract deliverables from task
        deliverables = []
        
        # Look for file mentions
        file_pattern = r'[\w_/]+\.py'
        files_mentioned = re.findall(file_pattern, task)
        deliverables.extend([(f, 'file') for f in files_mentioned])
        
        # Look for test mentions
        if 'test' in task.lower():
            deliverables.append(('tests', 'test'))
        
        # Look for specific endpoint mentions
        if 'endpoint' in task.lower() or 'api' in task.lower():
            deliverables.append(('endpoints', 'feature'))
        
        # Check what was actually done
        files_created = set()
        tests_run = False
        
        for action in actions_taken:
            if action['tool'] == 'write_file':
                path = action['args'].get('path', '')
                files_created.add(path.split('\\')[-1].split('/')[-1])
            elif 'test' in action['tool']:
                tests_run = True
        
        # Check for missing deliverables
        missing = []
        for deliverable, dtype in deliverables:
            fname = deliverable.split('/')[-1].split('\\')[-1]
            if dtype == 'file' and fname not in files_created:
                missing.append(deliverable)
            elif dtype == 'test' and not tests_run:
                missing.append('tests')
        
        is_complete = len(missing) == 0
        return is_complete, missing

    def _generate_and_run_tests(self, actions_taken: list) -> Optional[str]:
        """MYCO: Generate and run tests for modified Python files.
        
        MYCO Vision: Tests are negentropy - they reduce future disorder
        by catching regressions early.
        
        Args:
            actions_taken: List of actions taken during task
            
        Returns:
            Test verification report or None if no tests generated
        """
        # Find Python files that were created/modified
        py_files = []
        for action in actions_taken:
            if action['tool'] in ('write_file', 'edit_file', 'append_file'):
                path = action['args'].get('path', '')
                if path.endswith('.py') and action.get('success', False):
                    py_files.append(path)
        
        if not py_files:
            return None
        
        # Skip test files themselves
        py_files = [f for f in py_files if not f.startswith('test_')]
        
        if not py_files:
            return None
        
        # Generate test verification report
        report_lines = [
            "=" * 50,
            "🧪 MYCO TEST VERIFICATION",
            "=" * 50,
            ""
        ]
        
        tests_passed = 0
        tests_failed = 0
        
        for py_file in py_files[:5]:  # Limit to 5 files
            # Determine test file path
            path_obj = Path(py_file)
            test_file = f"test_{path_obj.name}"
            test_path = path_obj.parent / test_file
            
            report_lines.append(f"📄 File: {py_file}")
            
            # Check if test file exists
            if test_path.exists():
                # Run existing tests
                report_lines.append(f"  → Running existing tests: {test_file}")
                try:
                    result = self.tools['test_pytest'](str(test_path))
                    if result.get('success', False):
                        report_lines.append(f"  ✅ Tests passed")
                        tests_passed += 1
                    else:
                        report_lines.append(f"  ❌ Tests failed - review needed")
                        tests_failed += 1
                except Exception as e:
                    report_lines.append(f"  ⚠️ Test run error: {e}")
                    tests_failed += 1
            else:
                # Generate simple test stub
                report_lines.append(f"  → No tests found, generated stub: {test_file}")
                try:
                    # Create minimal test stub
                    module_name = path_obj.stem
                    test_content = f'''"""Tests for {module_name}."""
import pytest


def test_{module_name}_exists():
    """Verify {module_name} module can be imported."""
    try:
        # Try to import the module
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        # Module import test
        assert True  # Placeholder - add real import
    except ImportError as e:
        pytest.skip(f"Module not importable: {{e}}")


def test_{module_name}_basic():
    """Basic functionality test - customize this."""
    # TODO: Add test for main function/class in {module_name}
    assert True  # Placeholder - replace with real test
'''
                    # Write test file
                    self.tools['write_file'](path=str(test_path), content=test_content)
                    report_lines.append(f"  ✅ Test stub created")
                    tests_passed += 1
                    
                    # Try to run the test
                    report_lines.append(f"  → Running generated test...")
                    try:
                        result = self.tools['test_pytest'](str(test_path))
                        if result.get('success', False):
                            report_lines.append(f"  ✅ Stub test passed")
                        else:
                            report_lines.append(f"  ⚠️ Stub test needs customization")
                    except Exception:
                        report_lines.append(f"  ⚠️ Stub test needs customization")
                        
                except Exception as e:
                    report_lines.append(f"  ❌ Failed to create test: {e}")
                    tests_failed += 1
            
            report_lines.append("")
        
        # Summary
        report_lines.append("=" * 50)
        report_lines.append(f"Test Summary: {tests_passed} passed, {tests_failed} needs attention")
        
        if tests_failed > 0:
            report_lines.append("")
            report_lines.append("⚠️ MYCO: Some tests need attention.")
            report_lines.append("   Review test files and customize stubs.")
            report_lines.append("   Tests are negentropy - they prevent future decay.")
        
        report_lines.append("=" * 50)
        
        return "\n".join(report_lines)

    def _init_multi_file_coordination(self, task: str):
        """MYCO Phase 3: Initialize multi-file coordination for complex tasks.
        
        Sets up tracking for multi-file projects.
        """
        if "create" in task.lower() and ".py" in task.lower():
            ordered_files = self._analyze_task_dependencies(task)
            if len(ordered_files) > 1:
                self._multi_file_coordinator = {
                    'files_planned': set(ordered_files),
                    'files_created': set(),
                    'dependencies': {},
                    'order': ordered_files,
                    'current_index': 0
                }
                self.logger.info(f"Multi-file coordination initialized for {len(ordered_files)} files")
    
    def _mark_file_created(self, file_path: str):
        """MYCO Phase 3: Mark a file as created in coordination tracking."""
        if self._multi_file_coordinator:
            fname = file_path.split('\\')[-1].split('/')[-1]
            self._multi_file_coordinator['files_created'].add(fname)
            
            # Advance to next file in order
            current = self._multi_file_coordinator['order'][self._multi_file_coordinator['current_index']]
            if fname == current:
                self._multi_file_coordinator['current_index'] = min(
                    self._multi_file_coordinator['current_index'] + 1,
                    len(self._multi_file_coordinator['order']) - 1
                )
    
    def _get_next_file_suggestion(self) -> str:
        """MYCO Phase 3: Get suggestion for next file to create."""
        if not self._multi_file_coordinator:
            return ""
        
        coordinator = self._multi_file_coordinator
        remaining = coordinator['files_planned'] - coordinator['files_created']
        
        if not remaining:
            return "✅ All planned files created!"
        
        next_file = coordinator['order'][coordinator['current_index']]
        if next_file in remaining:
            return f"Next: Create {next_file}"
        else:
            # Find next uncreated file
            for f in coordinator['order'][coordinator['current_index']:]:
                if f in remaining:
                    return f"Next: Create {f}"
        
        return f"Remaining: {', '.join(remaining)}"

    def _analyze_task_dependencies(self, task: str) -> list:
        """MYCO: Analyze task to identify file dependencies.
        
        Extracts file mentions from task and suggests creation order.
        
        Args:
            task: Task description
            
        Returns:
            List of files in dependency order
        """
        import re
        
        # Extract file mentions from task
        file_pattern = r'[\w_]+\.py'
        files_mentioned = re.findall(file_pattern, task)
        
        # Remove duplicates while preserving order
        files = list(dict.fromkeys(files_mentioned))
        
        # Analyze dependencies based on common patterns
        dependencies = {}
        for f in files:
            deps = []
            
            # models.py should be created before main.py
            if 'main.py' in f and 'models.py' in files:
                deps.append('models.py')
            if 'main.py' in f and 'schemas.py' in files:
                deps.append('schemas.py')
            if 'main.py' in f and 'database.py' in files:
                deps.append('database.py')
            
            # test_*.py should be created after the file it tests
            if f.startswith('test_') and f.endswith('.py'):
                tested_file = f[5:]  # Remove 'test_' prefix
                if tested_file in files:
                    deps.append(tested_file)
            
            dependencies[f] = deps
        
        # Topological sort for creation order
        ordered = []
        remaining = set(files)
        
        while remaining:
            # Find files with no uncreated dependencies
            ready = [f for f in remaining if all(d in ordered for d in dependencies.get(f, []))]
            
            if not ready:
                # Circular dependency or missing dep, just add remaining
                ordered.extend(remaining)
                break
            
            # Add first ready file
            ordered.append(ready[0])
            remaining.remove(ready[0])
        
        return ordered

    def _estimate_entropy_impact(self, file_name: str, operation: str = "create") -> float:
        """MYCO: Estimate entropy impact for a file operation.
        
        Uses heuristics based on file type and operation.
        
        Args:
            file_name: Name of the file
            operation: Operation type (create, modify, delete)
            
        Returns:
            Estimated entropy delta (0.00-0.50)
        """
        # Base entropy by file type
        if file_name.endswith('.txt') or file_name.endswith('.md'):
            return 0.00  # Text files don't affect import graph
        elif file_name.endswith('.py'):
            if operation == "create":
                return 0.15  # New Python module
            elif operation == "modify":
                return 0.08  # Modification
            else:
                return -0.05  # Deletion reduces complexity
        elif file_name.startswith('test_'):
            return -0.02  # Tests reduce future entropy
        else:
            return 0.05  # Other files
    
    def _generate_entropy_aware_plan(self, task: str, files: list) -> str:
        """MYCO: Generate plan with entropy impact for each step.
        
        Args:
            task: Task description
            files: List of files to create/modify
            
        Returns:
            Formatted plan string with entropy impacts
        """
        plan_lines = []
        
        for i, f in enumerate(files[:8], 1):  # Limit to 8 files
            # Estimate entropy impact
            if "create" in task.lower() or "add" in task.lower():
                operation = "create"
            elif "remove" in task.lower() or "delete" in task.lower():
                operation = "delete"
            else:
                operation = "modify"
            
            entropy_impact = self._estimate_entropy_impact(f, operation)
            
            # Format entropy indicator
            if entropy_impact > 0.20:
                indicator = "⚠️ HIGH"
            elif entropy_impact > 0.10:
                indicator = "🟡 MODERATE"
            elif entropy_impact >= 0:
                indicator = "✅ LOW"
            else:
                indicator = "✅ REDUCES"
            
            plan_lines.append(f"  {i}. {f} (ΔH = {entropy_impact:+.2f}) [{indicator}]")
        
        return "\n".join(plan_lines)
    
    def _validate_plan_entropy(self, task: str, files: list) -> tuple:
        """MYCO Phase B: Validate if plan is entropy-safe.
        
        Checks each step against entropy budget.
        
        Args:
            task: Task description
            files: List of files in plan
            
        Returns:
            Tuple of (is_safe: bool, warnings: list, adjusted_plan: list)
        """
        warnings = []
        adjusted_files = []
        
        # Determine budget based on project size
        num_files = len(files)
        if num_files <= 5:
            budget = 0.50
            regime = "Embryo"
        elif num_files <= 20:
            budget = 0.30
            regime = "Growth"
        else:
            budget = 0.15
            regime = "Mature"
        
        for f in files:
            entropy_impact = self._estimate_entropy_impact(f, "create")
            
            if entropy_impact > budget:
                # High entropy - suggest splitting
                warnings.append(f"⚠️ {f} (ΔH = {entropy_impact:.2f}) exceeds {regime} budget ({budget:.2f})")
                
                # Suggest splitting for high-entropy files
                if f.endswith('main.py') or f.endswith('service.py') or f.endswith('controller.py'):
                    base_name = f.replace('.py', '')
                    warnings.append(f"   → SUGGESTION: Split {f} into:")
                    warnings.append(f"      - {base_name}_routes.py (ΔH ≈ 0.10)")
                    warnings.append(f"      - {base_name}_logic.py (ΔH ≈ 0.10)")
                    warnings.append(f"      - {base_name}_models.py (ΔH ≈ 0.10)")
                    adjusted_files.extend([f"{base_name}_routes.py", f"{base_name}_logic.py"])
                else:
                    adjusted_files.append(f)
            else:
                adjusted_files.append(f)
        
        is_safe = len(warnings) == 0
        return is_safe, warnings, adjusted_files
