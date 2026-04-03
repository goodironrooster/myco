"""Interactive agent mode with MYCO Vision integration and full toolset.

MYCO Vision Features:
- Entropy regime analysis before file writes
- Stigmergic annotations on Python files
- World model persistence
- Autopoietic gate enforcement
- Substrate health monitoring

Full Agent Tools:
- File operations (read, write, edit, delete, copy, append)
- Search tools (text, grep, files, definitions, entropy, todo, imports)
- Browser automation (Playwright)
- Process management
- Git operations
- Testing tools
"""

import json
import time
import warnings
from pathlib import Path

# Suppress SyntaxWarnings from docstring code examples
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

import click
import requests
from rich.console import Console

console = Console()

from ..ui import StatusDisplay, VerificationPanel, ApprovalPrompt
from ..ui import show_status, show_error, show_success
from ..ui import (
    EntropyStatusBar, get_status_bar,
    ConversationMemory, get_conversation_memory,
    EntropyVisualizer, get_entropy_visualizer,
    TensionMap,
    TrajectoryDisplay,
)
from ..agent.approval import ApprovalManager
from ..agent.tools import (
    CommandTools, FileTools, SearchTools, ToolResult,
    BrowserTools, ProcessTools, GitTools, TestTools, CodebaseSearch
)
from ..agent.impact_estimator import get_command_impact
from ..utils.config import Config
from ..utils.logging import LogConfig

# MYCO Vision imports
HAS_MYCO = False
try:
    myco_path = Path(__file__).parent.parent.parent / "myco"
    if myco_path.exists():
        import sys
        sys.path.insert(0, str(myco_path.parent))
        from myco.world import WorldModel
        from myco.entropy import calculate_substrate_health, get_regime_intervention
        from myco.gate import AutopoieticGate
        from myco.stigma import StigmergicAnnotation, StigmaReader
        
        # Helper to get regime from H value
        def get_regime(H: float) -> str:
            """Get entropy regime from H value."""
            if H < 0.3:
                return "crystallized"
            elif H > 0.75:
                return "diffuse"
            else:
                return "dissipative"
        
        HAS_MYCO = True
        console.print("[green]✓ MYCO Vision enabled[/green]\n")
except Exception as e:
    console.print(f"[dim]MYCO Vision: Not available ({e})[/dim]\n")


def _merge_tool_calls(fragments: list[dict]) -> list[dict]:
    """Merge streamed tool call fragments into complete tool calls.

    llama.cpp sends tool calls incrementally:
      fragment 0: {"index": 0, "id": "call_abc", "type": "function", "function": {"name": "write", "arguments": ""}}
      fragment 1: {"index": 0, "function": {"arguments": "{\"path\":"}}
      fragment 2: {"index": 0, "function": {"arguments": "\"hello.py\"}"}}

    Returns list of complete tool call dicts.
    """
    merged: dict[int, dict] = {}
    for frag in fragments:
        idx = frag.get("index", 0)
        if idx not in merged:
            merged[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}

        if "id" in frag:
            merged[idx]["id"] = frag["id"]
        if "type" in frag:
            merged[idx]["type"] = frag["type"]

        func = frag.get("function", {})
        if "name" in func:
            merged[idx]["function"]["name"] = func["name"]
        if "arguments" in func:
            merged[idx]["function"]["arguments"] += func["arguments"]

    return list(merged.values())


@click.command()
@click.option(
    "--model",
    "-m",
    "model_name",
    default=None,
    help="Model name (default: from server)",
)
@click.option(
    "--no-approval",
    is_flag=True,
    help="Disable approval prompts",
)
@click.pass_context
def interactive(ctx, model_name, no_approval):
    """Start interactive agent mode with MYCO Vision and full toolset.

    This mode combines chat with agent capabilities and MYCO vision:
    - Chat normally with the AI
    - The AI can use ALL agent tools (files, search, browser, git, tests)
    - MYCO Vision: Entropy checks before file modifications
    - MYCO Vision: Stigmergic annotations on Python files
    - You approve risky commands before execution

    \b
    Commands:
      help          - Show available commands
      status        - Show current task status (includes entropy)
      clear         - Clear conversation history
      exit/quit     - End session

    \b
    Example:
      gguf interactive
    """
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)
    base_url = f"http://{host}:{port}"
    project_root = Path.cwd()

    # Initialize MYCO Vision components
    world_model = None
    substrate_health_data = None
    if HAS_MYCO:
        try:
            world_model = WorldModel.load(project_root)
            substrate_health_data = calculate_substrate_health(project_root)
            # Handle both dict and object returns
            if isinstance(substrate_health_data, dict):
                file_count = len(substrate_health_data.get('files', []))
            else:
                file_count = len(substrate_health_data.files) if hasattr(substrate_health_data, 'files') else 0
            logger.info(f"MYCO Vision: World model loaded, {file_count} files tracked")
        except Exception as e:
            logger.warning(f"MYCO Vision: Could not initialize ({e})")

    approval_manager = ApprovalManager() if not no_approval else None

    # Check server
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            raise Exception("Server unhealthy")
    except requests.RequestException:
        click.echo(click.style("✗ ", fg="red") + f"Server not running at {base_url}")
        click.echo("Start the server with: myco")
        raise SystemExit(1)

    # Get model
    if not model_name:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            models = response.json().get("data", [])
            if models:
                model_name = models[0].get("id")
        except Exception:
            pass

    model_name = model_name or "Qwen3.5-9B-Q4_0"

    # Complete tool definitions - ALL agent tools
    tool_definitions = [
        # File operations
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "lines": {"type": "integer", "description": "Max lines to read (optional)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to file (MYCO: Auto-checks entropy budget for .py files)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append content to end of file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content to append"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit file by replacing text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "old_text": {"type": "string", "description": "Text to replace"},
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
                        "path": {"type": "string", "description": "Path to file to delete"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "copy_file",
                "description": "Copy a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source file path"},
                        "dst": {"type": "string", "description": "Destination file path"},
                    },
                    "required": ["src", "dst"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"},
                        "pattern": {"type": "string", "description": "Glob pattern (optional)"},
                    },
                    "required": ["path"],
                },
            },
        },
        # Command execution
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Run Python code snippet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            },
        },
        # Search tools
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": "Search text in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File/dir to search"},
                        "query": {"type": "string", "description": "Text to find"},
                        "max_results": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["path", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_grep",
                "description": "Search for regex pattern in files (grep-style)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search"},
                        "path": {"type": "string", "description": "Directory to search"},
                        "include": {"type": "string", "description": "Glob pattern (e.g., '*.py')"},
                        "max_results": {"type": "integer", "description": "Max results"},
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
                    "required": ["file_pattern"],
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
    ]

    # Add MYCO Vision tools
    if HAS_MYCO:
        tool_definitions.extend([
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
                    "name": "check_entropy",
                    "description": "MYCO Vision: Check entropy regime of a Python file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to Python file"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "substrate_health",
                    "description": "MYCO Vision: Get overall project health report",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_annotation",
                    "description": "MYCO Vision: Read stigmergic annotation from a Python file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to Python file"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "entropy_check",
                    "description": "MYCO Vision: Check if proposed change would increase entropy beyond threshold",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to file"},
                            "proposed_content": {"type": "string", "description": "Proposed new content"},
                        },
                        "required": ["file_path", "proposed_content"],
                    },
                },
            },
        ])

    # Add browser tools
    tool_definitions.extend([
        {
            "type": "function",
            "function": {
                "name": "browser_open",
                "description": "Open a URL in browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to open"},
                        "name": {"type": "string", "description": "Session name"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_screenshot",
                "description": "Take browser screenshot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Session name"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_click",
                "description": "Click element in browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector"},
                        "name": {"type": "string", "description": "Session name"},
                    },
                    "required": ["selector"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_fill",
                "description": "Fill input field in browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector"},
                        "value": {"type": "string", "description": "Value to fill"},
                        "name": {"type": "string", "description": "Session name"},
                    },
                    "required": ["selector", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browser_close",
                "description": "Close browser session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Session name"},
                    },
                },
            },
        },
    ])

    # Add process tools
    tool_definitions.extend([
        {
            "type": "function",
            "function": {
                "name": "process_start",
                "description": "Start a background process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                        "name": {"type": "string", "description": "Process name"},
                    },
                    "required": ["command", "name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "process_stop",
                "description": "Stop a background process",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Process name"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "process_status",
                "description": "Get process status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Process name"},
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
    ])

    # Add Git tools
    tool_definitions.extend([
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Show Git repository status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository path"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Show Git diff",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository path"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_add",
                "description": "Git add files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}, "description": "Files to add"},
                        "path": {"type": "string", "description": "Repository path"},
                    },
                    "required": ["files"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Git commit changes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "path": {"type": "string", "description": "Repository path"},
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_branch",
                "description": "Create or switch to branch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "branch": {"type": "string", "description": "Branch name"},
                        "create": {"type": "boolean", "description": "Create if not exists"},
                    },
                    "required": ["branch"],
                },
            },
        },
    ])

    # Add testing tools
    tool_definitions.extend([
        {
            "type": "function",
            "function": {
                "name": "test_run",
                "description": "Run tests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_file": {"type": "string", "description": "Test file or directory"},
                        "framework": {"type": "string", "description": "Test framework: pytest, unittest, jest, npm"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "test_pytest",
                "description": "Run pytest tests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Test file or directory"},
                        "args": {"type": "string", "description": "Additional pytest arguments"},
                    },
                },
            },
        },
    ])

    # Tools mapping - ALL agent tools
    tools = {
        # File operations
        "read_file": FileTools.read_file,
        "write_file": FileTools.write_file,
        "append_file": FileTools.append_file,
        "edit_file": FileTools.edit_file,
        "delete_file": FileTools.delete_file,
        "copy_file": FileTools.copy_file,
        "list_files": FileTools.list_files,
        # Commands
        "run_command": CommandTools.run_command,
        "run_python": CommandTools.run_python,
        # Search
        "search_text": SearchTools.search_text,
        "search_grep": CodebaseSearch.search_grep,
        "search_files": CodebaseSearch.search_files,
        "search_definitions": CodebaseSearch.search_definitions,
        "search_todo": CodebaseSearch.search_todo,
        "search_imports": CodebaseSearch.search_imports,
        # Browser
        "browser_open": BrowserTools.browser_open,
        "browser_screenshot": BrowserTools.browser_screenshot,
        "browser_click": BrowserTools.browser_click,
        "browser_fill": BrowserTools.browser_fill,
        "browser_close": BrowserTools.browser_close,
        "browser_evaluate": BrowserTools.browser_evaluate,
        # Process
        "process_start": ProcessTools.process_start,
        "process_stop": ProcessTools.process_stop,
        "process_status": ProcessTools.process_status,
        "process_list": ProcessTools.process_list,
        "process_logs": ProcessTools.process_logs,
        # Git
        "git_status": GitTools.git_status,
        "git_diff": GitTools.git_diff,
        "git_add": GitTools.git_add,
        "git_commit": GitTools.git_commit,
        "git_branch": GitTools.git_branch,
        "git_log": GitTools.git_log,
        # Tests
        "test_run": TestTools.test_run,
        "test_pytest": TestTools.test_pytest,
        "test_unittest": TestTools.test_unittest,
        "test_jest": TestTools.test_jest,
        "test_npm": TestTools.test_npm,
    }

    # Add MYCO Vision tools
    if HAS_MYCO:
        from myco.entropy import ImportGraphBuilder, EntropyCalculator
        
        def _check_entropy(path: str):
            """Check entropy of a file. Only Python files have structural entropy."""
            try:
                file_path = Path(path)
                if not file_path.exists():
                    return ToolResult(False, "", f"File not found: {path}")
                if not path.endswith('.py'):
                    return ToolResult(
                        True,
                        f"Non-Python file: {path}\n"
                        f"Entropy tracking is only for .py files.\n"
                        f"This file has no structural entropy budget.",
                        verified=True
                    )

                # Try to read stigmergic annotation from sidecar
                try:
                    from myco.stigma import load_annotations
                    annotations = load_annotations(project_root)
                    rel_path = str(file_path.relative_to(project_root))
                    if rel_path in annotations:
                        ann = annotations[rel_path].current
                        return ToolResult(
                            True,
                            f"H={ann.H:.3f} | Regime: {get_regime(ann.H)}\n"
                            f"Press: {ann.press} | Drift: {ann.drift:+.3f}\n"
                            f"Age: {ann.age} session(s)",
                            verified=True
                        )
                except Exception:
                    pass

                # Calculate entropy from import graph
                try:
                    builder = ImportGraphBuilder(project_root)
                    builder.scan()
                    calc = EntropyCalculator(builder)
                    module_name = builder._path_to_module_name(file_path)
                    h = calc.calculate_module_entropy(module_name)
                    regime = get_regime(h)
                    return ToolResult(
                        True,
                        f"H={h:.3f} | Regime: {regime}\n(Calculated on-demand)",
                        verified=True
                    )
                except Exception as e:
                    return ToolResult(False, "", f"Error calculating entropy: {e}")
            except Exception as e:
                return ToolResult(False, "", f"Error: {e}")
        
        def _substrate_health_report():
            """Get substrate health report."""
            try:
                health = calculate_substrate_health(project_root)
                # Handle dict return
                if isinstance(health, dict):
                    files = health.get('files', [])
                    avg_entropy = health.get('avg_entropy', 0)
                    crystallized = health.get('crystallized_count', 0)
                    dissipative = health.get('dissipative_count', 0)
                    diffuse = health.get('diffuse_count', 0)
                else:
                    files = health.files
                    avg_entropy = health.avg_entropy
                    crystallized = health.crystallized_count
                    dissipative = health.dissipative_count
                    diffuse = health.diffuse_count
                
                report = [
                    f"Project Health Report",
                    f"=" * 40,
                    f"Total files: {len(files)}",
                    f"Average entropy: {avg_entropy:.3f}",
                    f"",
                    f"Entropy Distribution:",
                    f"  Crystallized (H<0.3): {crystallized}",
                    f"  Dissipative (0.3-0.7): {dissipative}",
                    f"  Diffuse (H>0.7): {diffuse}",
                    f"",
                    f"At-risk files (high entropy):",
                ]
                
                for f in files[:10]:
                    if isinstance(f, dict):
                        entropy = f.get('entropy', 0)
                        path = f.get('path', '')
                    else:
                        entropy = f.entropy if hasattr(f, 'entropy') else 0
                        path = f.path if hasattr(f, 'path') else str(f)
                    if entropy > 0.7:
                        report.append(f"  - {path} (H={entropy:.3f})")
                
                return ToolResult(True, "\n".join(report), verified=True)
            except Exception as e:
                return ToolResult(False, "", f"Error: {e}")
        
        def _read_annotation(path: str):
            """Read stigmergic annotation from file."""
            try:
                reader = StigmaReader()
                annotation = reader.read_annotation(path)
                if not annotation:
                    return ToolResult(False, "", "No annotation found")
                
                report = [
                    f"Stigmergic Annotation: {path}",
                    f"=" * 40,
                    f"Entropy (H): {annotation.H:.3f}",
                    f"Regime: {get_regime(annotation.H)}",
                    f"Dependencies: {len(annotation.dependencies)}",
                    f"Last modified: {annotation.last_modified}",
                    f"Error count: {annotation.error_count}",
                ]
                
                if annotation.dependencies:
                    report.append(f"\nDependencies:")
                    for dep in annotation.dependencies[:5]:
                        report.append(f"  - {dep}")
                
                return ToolResult(True, "\n".join(report), verified=True)
            except Exception as e:
                return ToolResult(False, "", f"Error: {e}")
        
        def _search_by_entropy(regime: str = None, min_entropy: float = None, 
                               max_entropy: float = None, path: str = "."):
            """Search files by entropy regime."""
            try:
                health = calculate_substrate_health(project_root / path)
                # Handle dict return
                if isinstance(health, dict):
                    files = health.get('files', [])
                else:
                    files = health.files
                
                results = []
                
                for f in files:
                    if isinstance(f, dict):
                        entropy = f.get('entropy', 0)
                        fpath = f.get('path', '')
                    else:
                        entropy = f.entropy if hasattr(f, 'entropy') else 0
                        fpath = f.path if hasattr(f, 'path') else str(f)
                    
                    include = True
                    if regime and get_regime(entropy) != regime:
                        include = False
                    if min_entropy is not None and entropy < min_entropy:
                        include = False
                    if max_entropy is not None and entropy > max_entropy:
                        include = False
                    
                    if include:
                        results.append(f"{fpath} (H={entropy:.3f}, {get_regime(entropy)})")
                
                return ToolResult(
                    True,
                    f"Found {len(results)} files:\n" + "\n".join(results[:20]),
                    verified=True
                )
            except Exception as e:
                return ToolResult(False, "", f"Error: {e}")
        
        def _entropy_check(file_path: str, proposed_content: str):
            """Check if proposed change passes entropy gate."""
            try:
                from myco.entropy import check_entropy_budget
                path = Path(file_path)
                current = path.read_text() if path.exists() else ""
                within, curr_h, prop_h, msg = check_entropy_budget(current, proposed_content)
                return ToolResult(
                    True,
                    f"{'✓ PASS' if within else '✗ BLOCK'}: {msg}\n(H: {curr_h:.3f} → {prop_h:.3f})",
                    verified=within
                )
            except Exception as e:
                return ToolResult(False, "", f"Error: {e}")
        
        tools["check_entropy"] = _check_entropy
        tools["substrate_health"] = _substrate_health_report
        tools["read_annotation"] = _read_annotation
        tools["search_by_entropy"] = _search_by_entropy
        tools["entropy_check"] = _entropy_check

    # Enhanced system prompt with MYCO Vision
    system_content = (
        "You are MYCO, an AI coding assistant with MYCO Vision capabilities and FULL tool access.\n\n"
        "You can:\n"
        "- File ops: read, write, edit, delete, copy, append files\n"
        "- Commands: run shell commands, execute Python code\n"
        "- Search: text, grep, files, definitions, TODOs, imports, entropy\n"
        "- Browser: open URLs, click, fill forms, screenshots\n"
        "- Processes: start, stop, monitor background processes\n"
        "- Git: status, diff, add, commit, branch\n"
        "- Testing: run pytest, unittest, jest, npm tests\n"
    )

    if HAS_MYCO:
        system_content += (
            "- MYCO Vision: entropy checks, stigmergic annotations, substrate health\n\n"
            "MYCO Vision Guidelines:\n"
            "1. Before modifying Python files, check entropy regime\n"
            "2. Crystallized files (H<0.3) are rigid — avoid changes\n"
            "3. Diffuse files (H>0.7) need refactoring — suggest improvements\n"
            "4. Use stigmergic annotations to track file state\n"
            "5. Prefer changes that reduce entropy (negentropy)\n"
            "6. Entropy tracking is ONLY for .py files. Other files (md, txt, json, yaml, etc.) "
            "have no entropy budget — write them freely\n\n"
        )
    else:
        system_content += "\n"

    system_content += (
        "IMPORTANT:\n"
        "1. Use tools when tasks require file operations or running code\n"
        "2. After using tools, explain what you did\n"
        "3. Verify work by reading files back or running tests\n"
        "4. Be honest about what was verified vs. assumed\n\n"
        "CODE QUALITY RULES:\n"
        "1. NEVER use bare 'Union' — always provide type params: Union[int, float]\n"
        "2. NEVER use 'return' without a value unless the function returns None\n"
        "3. ALWAYS close triple-quoted strings before using them in function calls\n"
        "4. ALWAYS verify syntax before claiming a file is valid\n"
        "5. When writing Python, use 'python' tool to verify syntax before claiming success\n\n"
        "LARGE FILE RULE:\n"
        "1. For files over ~50 lines, write the first 30-40 lines with write_file\n"
        "2. Then use append_file to add the rest in chunks of 30-40 lines\n"
        "3. NEVER split a single file into multiple files just to avoid size limits\n"
        "4. NEVER create files like 'styles_part1.css', 'styles_part2.css' — write ONE file\n"
        "5. If a file gets truncated, use append_file to add the missing content\n\n"
        "When done with a task, wait for the user's next input."
    )

    messages = [
        {"role": "system", "content": system_content}
    ]

    # Show welcome
    console.print("")
    console.print("=" * 60)
    console.print("  MYCO Interactive Agent [green]with Vision[/green]")
    console.print("=" * 60)
    console.print(f"  Model: {model_name}")
    console.print(f"  Server: {base_url}")
    console.print(f"  Project: {project_root}")
    console.print(f"  Approval: {'disabled' if no_approval else 'enabled'}")
    console.print(f"  MYCO Vision: {'✓ Enabled' if HAS_MYCO else '✗ Disabled'}")
    console.print("=" * 60)
    console.print("")
    console.print(f"  [green]Tools Available: {len(tools)}[/green]")
    console.print("  - File ops: read, write, edit, delete, copy, append")
    console.print("  - Search: text, grep, files, definitions, TODOs, entropy")
    console.print("  - Browser: open, click, fill, screenshot, evaluate")
    console.print("  - Process: start, stop, status, list, logs")
    console.print("  - Git: status, diff, add, commit, branch, log")
    console.print("  - Tests: pytest, unittest, jest, npm")
    if HAS_MYCO:
        console.print("  - MYCO: entropy, substrate health, annotations")
    console.print("")
    console.print("Commands:")
    console.print("  help       - Show this help")
    console.print("  status     - Show conversation status")
    console.print("  memory     - Show session action history")
    console.print("  health     - Show conversation health")
    console.print("  gradient   - Show entropy gradient map")
    console.print("  tension    - Show dependency tension map")
    console.print("  trajectory - Show entropy trajectory prediction")
    console.print("  clear      - Clear conversation history")
    console.print("  exit       - End session")
    console.print("")
    console.print("Give me tasks like:")
    console.print('  "Create a hello.py file"')
    console.print('  "Run the tests"')
    console.print('  "Search for TODO comments"')
    console.print('  "Check git status"')
    console.print("")
    console.print("-" * 60)
    console.print("")

    # Initialize Phase 1: MYCO Vision UI components
    status_display = StatusDisplay()
    verification_panel = VerificationPanel()
    actions_taken = []
    
    # MYCO Vision UI components
    entropy_status_bar = EntropyStatusBar(project_root)
    conversation_memory = ConversationMemory(project_root)
    entropy_visualizer = EntropyVisualizer(project_root)
    tension_map_ui = TensionMap(project_root)
    trajectory_ui = TrajectoryDisplay(project_root)

    # Refresh initial entropy state
    if HAS_MYCO:
        entropy_status_bar.update()
        entropy_visualizer.refresh()
        # Show initial substrate health
        entropy_status_bar.render()

        # Load cross-session stigmergic memory
        from myco.stigma import load_annotations as load_stigmergic
        stigmergic_annotations = load_stigmergic(project_root)
        previously_touched = {
            path: hist.current
            for path, hist in stigmergic_annotations.items()
            if hist.current.age == 0 or hist.history
        }
        if previously_touched:
            console.print("")
            console.print(f"[dim]MYCO remembers {len(previously_touched)} file(s) from previous sessions[/dim]")
            for path, ann in list(previously_touched.items())[:5]:
                regime = "crystallized" if ann.H < 0.3 else ("diffuse" if ann.H > 0.75 else "dissipative")
                console.print(f"  [dim]• {path} — H:{ann.H:.2f} {regime} (drift:{ann.drift:+.2f})[/dim]")
            if len(previously_touched) > 5:
                console.print(f"  [dim]... and {len(previously_touched) - 5} more[/dim]")

    pending_approval = None
    iteration = 0
    # Dynamic max iterations based on task complexity
    # Default 50 for complex multi-file tasks, can go higher for large refactoring
    max_iterations = 50
    # Track if user requested more iterations
    soft_limit = True  # Allow continuing past limit with user consent

    while True:
        # Always show entropy status bar before each prompt
        if HAS_MYCO:
            entropy_status_bar.iteration = iteration
            entropy_status_bar.update()
            console.print(entropy_status_bar.render())

            # Proactive intervention suggestions (Phase 2.3)
            if entropy_visualizer.last_health:
                from myco.stigma import load_annotations as load_stig
                annotations = load_stig(project_root)
                at_risk = []
                for path, hist in annotations.items():
                    h = hist.current.H
                    if 0.6 < h < 0.75:
                        at_risk.append((path, h, hist.current.drift))
                if at_risk:
                    at_risk.sort(key=lambda x: x[1], reverse=True)
                    console.print("")
                    console.print("[bold yellow]⚠ MYCO Intervention Suggestions[/bold yellow]")
                    for path, h, drift in at_risk[:3]:
                        console.print(f"  [yellow]• {path}[/yellow] [dim](H:{h:.2f}, drift:{drift:+.2f}) — approaching diffuse regime[/dim]")
                    console.print("  [dim]Type 'gradient' for full entropy map[/dim]")

        try:
            user_input = click.prompt(click.style("You", bold=True, fg="green"), prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            if HAS_MYCO:
                conversation_memory.save()
                # Increment annotation ages for files not touched this session
                from myco.stigma import load_annotations, save_annotations
                annotations = load_annotations(project_root)
                for path, hist in annotations.items():
                    hist.current.age += 1
                save_annotations(project_root, annotations)
                console.print(f"\n[dim]Session saved. {len(annotations)} file(s) annotated.[/dim]")
            click.echo("\n\nGoodbye!")
            break

        user_input = user_input.strip()

        if user_input.lower() in ("quit", "exit", "q"):
            if HAS_MYCO:
                conversation_memory.save()
                # Increment annotation ages for files not touched this session
                from myco.stigma import load_annotations, save_annotations
                annotations = load_annotations(project_root)
                for path, hist in annotations.items():
                    hist.current.age += 1
                save_annotations(project_root, annotations)
                console.print(f"\n[dim]Session saved. {len(annotations)} file(s) annotated.[/dim]")
            click.echo("Goodbye!")
            break

        if user_input.lower() == "help":
            click.echo("\nCommands:")
            click.echo("  help     - Show this help")
            click.echo("  status   - Show conversation status")
            click.echo("  memory   - Show session action history")
            click.echo("  clear    - Clear conversation history")
            click.echo("  exit     - End session")
            if HAS_MYCO:
                click.echo("\nMYCO Vision:")
                click.echo("  check_entropy <path>  - Check file entropy")
                click.echo("  substrate_health      - Project health report")
                click.echo("  read_annotation <path> - Read stigmergic annotation")
            continue

        if user_input.lower() == "memory":
            if HAS_MYCO:
                console.print("")
                console.print(conversation_memory.render_panel(limit=15))
            else:
                console.print("[dim]MYCO Vision not available[/dim]")
            continue

        if user_input.lower() == "gradient":
            if HAS_MYCO:
                entropy_visualizer.refresh()
                console.print("")
                console.print(entropy_visualizer.render_gradient_map(limit=15))
            else:
                console.print("[dim]MYCO Vision not available[/dim]")
            continue

        if user_input.lower() == "health":
            if HAS_MYCO:
                console.print("")
                console.print(conversation_memory.render_health_panel())
            else:
                console.print("[dim]MYCO Vision not available[/dim]")
            continue

        if user_input.lower() == "tension":
            console.print("")
            console.print(tension_map_ui.render(limit=15))
            continue

        if user_input.lower() == "trajectory":
            console.print("")
            console.print(trajectory_ui.render())
            continue

        if user_input.lower() == "status":
            click.echo(f"\nMessages: {len(messages)}")
            click.echo(f"Model: {model_name}")
            if HAS_MYCO and substrate_health_data:
                if isinstance(substrate_health_data, dict):
                    file_count = len(substrate_health_data.get('files', []))
                    avg_h = substrate_health_data.get('avg_entropy', 0)
                else:
                    file_count = len(substrate_health_data.files) if hasattr(substrate_health_data, 'files') else 0
                    avg_h = substrate_health_data.avg_entropy if hasattr(substrate_health_data, 'avg_entropy') else 0
                click.echo(f"Substrate Health: {file_count} files, avg H={avg_h:.3f}")
            if pending_approval:
                click.echo(f"Pending approval: {pending_approval['command'][:50]}...")
            
            # Show conversation memory panel
            if HAS_MYCO:
                console.print("")
                console.print(conversation_memory.render_panel(limit=10))
                entropy_status_bar.iteration = iteration
                console.print(entropy_status_bar.render())
            continue

        if user_input.lower() == "clear":
            messages = [messages[0]]  # Keep system prompt
            click.echo("Conversation cleared.\n")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Phase 3.2: Self-Repair — check for repeated failures before starting
        if HAS_MYCO and len(conversation_memory.actions) >= 3:
            patterns = conversation_memory.detect_patterns()
            if patterns["repeated_errors"]:
                console.print("")
                console.print("[bold yellow]⚠ Autopoietic Recovery[/bold yellow]")
                for err in patterns["repeated_errors"]:
                    tool_name = err["pattern"].split(":")[0].strip()
                    error_msg = err["pattern"].split(":", 1)[1].strip() if ":" in err["pattern"] else ""
                    console.print(f"  [yellow]• Repeated error in {tool_name}[/yellow] [dim]({err['count']}x)[/dim]")
                    if "json" in error_msg.lower() or "parse" in error_msg.lower():
                        console.print(f"    [dim]💡 Suggestion: Check tool argument format — ensure valid JSON[/dim]")
                    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                        console.print(f"    [dim]💡 Suggestion: Check file permissions or path[/dim]")
                    elif "not found" in error_msg.lower() or "no such" in error_msg.lower():
                        console.print(f"    [dim]💡 Suggestion: Verify the file/directory exists first[/dim]")
                    else:
                        console.print(f"    [dim]💡 Suggestion: Try a different approach or tool[/dim]")

        # Loop for tool calls
        tool_call_count = 0
        verified_count = 0
        failed_count = 0

        # Start task display
        status_display.start_task(f"MYCO: {user_input[:40]}...")

        while tool_call_count < max_iterations:
            iteration += 1

            try:
                request_data = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "tools": tool_definitions,
                    "stream": True,  # Stream tokens so user sees progress
                }

                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=request_data,
                    timeout=600,
                    stream=True,
                )
                response.raise_for_status()

            except requests.RequestException as e:
                show_error(f"API error: {e}")
                break

            # Stream tokens in real-time
            content = ""
            streamed_tool_calls = []
            console.print("")
            console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
            try:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data = line_str[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})

                                # Regular text tokens
                                token = delta.get("content", "")
                                if token:
                                    content += token
                                    console.print(token, end="")

                                # Tool call chunks
                                if "tool_calls" in delta:
                                    for tc in delta["tool_calls"]:
                                        streamed_tool_calls.append(tc)
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass  # Stream ended unexpectedly, use what we have
            console.print("")  # Newline after streaming

            # Determine: did the model call tools or just respond?
            if streamed_tool_calls:
                # Merge streamed tool call fragments
                tool_calls_raw = _merge_tool_calls(streamed_tool_calls)
            elif content:
                # Text response, no tool calls — this is the final answer
                # Phase 5.2: Truncation Detector — check for incomplete responses
                if content.endswith(('(', '[', '{', '"', "'", '\\')) or content.count('(') != content.count(')'):
                    console.print("")
                    console.print("[yellow]⚠ Response appears truncated — auto-completing[/yellow]")
                    # Close any unclosed parentheses/brackets
                    open_parens = content.count('(') - content.count(')')
                    open_brackets = content.count('[') - content.count(']')
                    open_braces = content.count('{') - content.count('}')
                    closing = ')' * max(0, open_parens) + ']' * max(0, open_brackets) + '}' * max(0, open_braces)
                    if content.endswith('"') or content.count('"') % 2 == 1:
                        closing = '"' + closing
                    if content.endswith("'") or content.count("'") % 2 == 1:
                        closing = "'" + closing
                    content = content.rstrip() + closing
                    console.print(f"[dim]Added closing chars: {closing}[/dim]")

                messages.append({"role": "assistant", "content": content})
                break
            else:
                # Empty response, stop
                break

            # Process tool calls
            tool_results = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                name = func.get("name")
                arguments = func.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]Warning: Failed to parse tool arguments: {e}[/yellow]")
                        tool_results.append({"tool": name, "result": f"Error: Invalid JSON arguments - {e}"})
                        continue

                if name not in tools:
                    tool_results.append({"tool": name, "result": f"Error: Unknown tool '{name}'"})
                    continue

                # Create step name for UI
                step_name = f"{name}({', '.join(f'{k}={v}' for k, v in arguments.items())})"
                step_index = len(actions_taken)
                status_display.add_step(step_name)

                # Phase 5.1: Syntax Gate — validate Python before writing
                if name in ["write_file", "edit_file", "append_file"]:
                    path = arguments.get("path", "")
                    file_content = arguments.get("content", "")
                    if path.endswith('.py') and file_content:
                        try:
                            import ast
                            ast.parse(file_content)
                        except SyntaxError as e:
                            console.print(f"[red]✗ Syntax Gate blocked: {path}[/red]")
                            console.print(f"[dim]  Line {e.lineno}: {e.msg}[/dim]")
                            status_display.fail_step(step_index, error=f"syntax_error: {e.msg}")
                            tool_results.append({
                                "tool": name,
                                "result": f"BLOCKED by Syntax Gate: {e.msg} (line {e.lineno})"
                            })
                            actions_taken.append({
                                "tool": name, "args": arguments,
                                "success": False, "verified": False
                            })
                            failed_count += 1
                            continue

                # MYCO Vision: Check entropy before file writes
                if HAS_MYCO and name == "write_file":
                    path = arguments.get("path", "")
                    content = arguments.get("content", "")
                    
                    if path.endswith('.py'):
                        # Check entropy budget
                        try:
                            from myco.entropy import check_entropy_budget
                            within_budget, curr_h, prop_h, msg = check_entropy_budget(
                                Path(path), content, threshold=0.15
                            )
                            if not within_budget:
                                console.print(f"[red]MYCO Gate: Entropy budget exceeded![/red]")
                                console.print(f"[dim]{msg}[/dim]")
                                status_display.fail_step(step_index, error="entropy_gate_blocked")
                                tool_results.append({
                                    "tool": name,
                                    "result": f"BLOCKED by MYCO entropy gate: {msg}"
                                })
                                continue
                            else:
                                console.print(f"[green]MYCO: Entropy budget OK (H={curr_h:.3f} → {prop_h:.3f})[/green]")
                        except Exception as e:
                            logger.debug(f"Entropy check skipped: {e}")

                # Handle approval for run_command
                if name == "run_command" and approval_manager:
                    command = arguments.get("command", "")

                    # Check if blocked
                    if approval_manager.is_blocked(command):
                        show_error(f"Command blocked: {command}")
                        status_display.fail_step(step_index, error="blocked")
                        tool_results.append(
                            {"tool": name, "result": f"Command blocked for safety: {command}"}
                        )
                        actions_taken.append(
                            {"tool": name, "args": arguments, "success": False, "verified": False}
                        )
                        failed_count += 1
                        continue

                    requires_approval, rule = approval_manager.check_approval_required(command)

                    if requires_approval:
                        # Get impact info
                        impact_info = get_command_impact(command)

                        prompt = ApprovalPrompt()
                        response = prompt.request_approval(
                            command=command,
                            rule_description=rule.description if rule else "Risky operation",
                            impact_info=impact_info,
                            timeout_seconds=60,
                        )

                        if response not in ("y", "yes", ""):
                            show_error(f"Command denied: {command}")
                            status_display.fail_step(step_index, error="denied")
                            tool_results.append(
                                {"tool": name, "result": f"Command denied by user: {command}"}
                            )
                            actions_taken.append(
                                {
                                    "tool": name,
                                    "args": arguments,
                                    "success": False,
                                    "verified": False,
                                }
                            )
                            failed_count += 1
                            continue

                # Execute tool
                start_time = time.time()
                try:
                    result = tools[name](**arguments)
                    duration = time.time() - start_time
                    result_str = result.output if result.success else f"Error: {result.error}"

                    # Track verification status
                    actions_taken.append(
                        {
                            "tool": name,
                            "args": arguments,
                            "success": result.success,
                            "verified": result.verified,
                        }
                    )
                    
                    # Record in conversation memory (MYCO Vision)
                    conversation_memory.record_action(
                        tool_name=name,
                        arguments=arguments,
                        success=result.success,
                        verified=result.verified,
                        error=result.error if not result.success else None,
                        duration=duration,
                    )

                    # Show entropy annotation for file operations (MYCO Vision)
                    if HAS_MYCO and name in ["write_file", "edit_file", "append_file"] and result.success:
                        file_path = arguments.get("path") or arguments.get("file_path")
                        if file_path and file_path.endswith('.py'):
                            # Calculate entropy for the file
                            try:
                                from myco.entropy import ImportGraphBuilder, EntropyCalculator
                                from myco.stigma import (
                                    StigmergicAnnotation, AnnotationHistory,
                                    load_annotations, save_annotations,
                                )

                                builder = ImportGraphBuilder(project_root)
                                builder.scan()
                                calc = EntropyCalculator(builder)
                                module_name = builder._path_to_module_name(Path(file_path))
                                h_structural = calc.calculate_module_entropy(module_name)

                                regime = "crystallized" if h_structural < 0.3 else ("diffuse" if h_structural > 0.75 else "dissipative")
                                regime_color = "blue" if regime == "crystallized" else ("yellow" if regime == "diffuse" else "green")

                                # Write stigmergic annotation to sidecar file
                                rel_path = str(Path(file_path).relative_to(project_root))
                                annotations = load_annotations(project_root)

                                # Check for previous annotation to calculate drift
                                prev_h = h_structural
                                if rel_path in annotations:
                                    prev_h = annotations[rel_path].current.H

                                drift = h_structural - prev_h

                                # Determine press type based on regime
                                if regime == "diffuse":
                                    press = "compression_collapse"
                                elif regime == "crystallized":
                                    press = "none"
                                else:
                                    press = "decompose"

                                new_annotation = StigmergicAnnotation(
                                    H=h_structural,
                                    press=press,
                                    age=0,
                                    drift=drift,
                                )

                                # Save to sidecar
                                if rel_path not in annotations:
                                    annotations[rel_path] = AnnotationHistory(current=new_annotation, history=[])
                                else:
                                    # Archive old current to history
                                    annotations[rel_path].history.append({
                                        "H": annotations[rel_path].current.H,
                                        "press": annotations[rel_path].current.press,
                                        "drift": annotations[rel_path].current.drift,
                                        "session": iteration,
                                    })
                                    annotations[rel_path].current = new_annotation

                                save_annotations(project_root, annotations)
                                stigma_written = True

                                annotation_lines = [
                                    f"[bold]File:[/bold] {Path(file_path).name}",
                                    f"[bold]Entropy:[/bold] {h_structural:.2f}",
                                    f"[bold]Regime:[/bold] [{regime_color}]{regime}[/{regime_color}]",
                                    f"[bold]Stigmergic Trace:[/bold] {'✓ Written' if stigma_written else '○ Pending'}",
                                    f"[bold]Drift:[/bold] {drift:+.2f}",
                                ]
                                border_color = "red" if regime == "diffuse" else ("yellow" if h_structural > 0.6 else "cyan")
                                annotation_panel = Panel(
                                    "\n".join(annotation_lines),
                                    title="[bold cyan]MYCO Annotation[/bold cyan]",
                                    border_style=border_color,
                                )
                                console.print("")
                                console.print(annotation_panel)
                            except Exception as e:
                                logger.debug(f"Entropy annotation skipped: {e}")

                    if result.verified:
                        verified_count += 1
                        status_display.complete_step(step_index, verified=True, details="verified")
                    elif result.success:
                        status_display.complete_step(step_index, verified=False, details="success")
                    else:
                        failed_count += 1
                        status_display.fail_step(step_index, error=result.error or "failed")

                    tool_results.append({"tool": name, "result": result_str})
                except Exception as e:
                    tool_results.append({"tool": name, "result": f"Error: {e}"})
                    failed_count += 1
                    status_display.fail_step(step_index, error=str(e))

            # Add tool results to conversation
            if tool_results:
                results_text = "\n\n".join(
                    f"[{r['tool']} result]:\n{r['result']}" for r in tool_results
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool execution results:\n\n{results_text}\n\nContinue completing the user's task.",
                    }
                )
                tool_call_count += 1

                # Phase 5.3: Context Pruning — keep only recent tool results
                # Keep: system prompt + user task + last 3 assistant/tool exchanges
                max_exchanges = 3
                if len(messages) > 2 + (max_exchanges * 2):  # system + user + exchanges
                    # Find the cutoff point (keep last N exchanges)
                    cutoff = 2 + (max_exchanges * 2)
                    # Keep system prompt and original user task, prune middle
                    system_msg = messages[0]
                    original_task = messages[1]
                    recent = messages[-cutoff:]
                    messages = [system_msg, original_task] + recent
            else:
                break

        # Show verification summary
        if actions_taken:
            status_display.finish_task(success=(failed_count == 0))
            verification_panel.show_verification(actions_taken)

            # Show conversation memory update (MYCO Vision Phase 1)
            if HAS_MYCO:
                recent_actions = conversation_memory.get_recent_actions(5)
                if recent_actions:
                    verified_count = sum(1 for a in recent_actions if a.verified)
                    if verified_count > 0:
                        console.print("")
                        console.print(f"[green]✓ {verified_count}/{len(recent_actions)} actions verified this iteration[/green]")

                # Phase 3.1: Conversation Health Monitor
                health = conversation_memory.compute_health()
                if health["has_issues"]:
                    console.print("")
                    console.print(conversation_memory.render_health_panel())

                # Update entropy status bar
                entropy_status_bar.iteration = iteration

        if iteration >= max_iterations:
            # Soft limit - ask user if they want to continue
            if soft_limit:
                console.print("")
                console.print(f"[yellow]⚠ Reached {max_iterations} iterations. Task still in progress.[/yellow]")
                console.print(f"[dim]Actions taken: {len(actions_taken)} | Verified: {sum(1 for a in actions_taken if a.get('verified', False))}[/dim]")
                console.print("")
                
                # Ask user
                try:
                    continue_choice = click.prompt(
                        "Continue working? (y/n/increase limit)",
                        default="y",
                        type=click.Choice(['y', 'n', 'i'], case_sensitive=False)
                    )
                    
                    if continue_choice.lower() == 'y':
                        iteration = 0  # Reset counter
                        max_iterations += 50  # Increase limit
                        console.print(f"[green]✓ Continuing for {max_iterations} more iterations[/green]\n")
                        continue
                    elif continue_choice.lower() == 'i':
                        try:
                            extra = click.prompt("How many more iterations?", default=50, type=int)
                            max_iterations += extra
                            iteration = 0
                            console.print(f"[green]✓ Limit increased to {max_iterations}[/green]\n")
                            continue
                        except (ValueError, KeyboardInterrupt):
                            pass
                    
                    # User chose to stop
                    console.print("[yellow]Stopping. Task can be resumed with 'continue'[/yellow]\n")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Interrupted by user[/yellow]\n")
            else:
                console.print("[yellow]Max iterations reached[/yellow]")
            
            # Save session on exit
            if HAS_MYCO:
                try:
                    conversation_memory.save()
                    logger.info(f"Session saved: {conversation_memory.session_id}")
                except Exception as e:
                    logger.debug(f"Failed to save session: {e}")
            break
