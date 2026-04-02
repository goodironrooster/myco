"""Interactive agent mode - chat and give tasks as you go."""

import json
import time

import click
import requests
from rich.console import Console
from rich.panel import Panel

console = Console()

from ..ui import StatusDisplay, VerificationPanel, ApprovalPrompt
from ..ui import show_status, show_error, show_success
from ..agent.approval import ApprovalManager
from ..agent.tools import CommandTools, FileTools, SearchTools, ToolResult
from ..agent.impact_estimator import get_command_impact
from ..ui import ApprovalPrompt, show_status, show_error, show_success
from ..utils.config import Config
from ..utils.logging import LogConfig


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
    """Start interactive agent mode - chat and use tools as you go.

    This mode combines chat with agent capabilities:
    - Chat normally with the AI
    - The AI can use tools (read/write files, run commands) when needed
    - You approve risky commands before execution

    \b
    Commands:
      help          - Show available commands
      status        - Show current task status
      approve       - Approve current pending command
      deny          - Deny current pending command
      clear         - Clear conversation history
      exit/quit     - End session

    \b
    Example:
      gguf agent interactive
    """
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)
    base_url = f"http://{host}:{port}"

    approval_manager = ApprovalManager() if not no_approval else None

    # Check server
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            raise Exception("Server unhealthy")
    except requests.RequestException:
        click.echo(click.style("✗ ", fg="red") + f"Server not running at {base_url}")
        click.echo("Start server with: gguf server start")
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

    # Tool definitions for LLM
    tool_definitions = [
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
                "description": "Write content to file",
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
    ]

    tools = {
        "read_file": FileTools.read_file,
        "write_file": FileTools.write_file,
        "edit_file": FileTools.edit_file,
        "list_files": FileTools.list_files,
        "run_command": CommandTools.run_command,
        "search_text": SearchTools.search_text,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are MYCO, an AI coding assistant with TOOLS to help users.\n\n"
                "You can:\n"
                "- Read, write, and edit files\n"
                "- Run shell commands\n"
                "- Search for text\n"
                "- List directory contents\n\n"
                "IMPORTANT:\n"
                "1. Use tools when tasks require file operations or running code\n"
                "2. After using tools, explain what you did\n"
                "3. Verify work by reading files back or running tests\n"
                "4. Be honest about what was verified vs. assumed\n\n"
                "When done with a task, wait for the user's next input."
            ),
        }
    ]

    # Show welcome - use ASCII-safe output
    console.print("")
    console.print("=" * 60)
    console.print("  MYCO Interactive Agent")
    console.print("=" * 60)
    console.print(f"  Model: {model_name}")
    console.print(f"  Server: {base_url}")
    console.print(f"  Approval: {'disabled' if no_approval else 'enabled'}")
    console.print("=" * 60)
    console.print("")
    console.print("Commands:")
    console.print("  help   - Show help")
    console.print("  status - Show conversation status")
    console.print("  clear  - Clear conversation history")
    console.print("  exit   - End session")
    console.print("")
    console.print("Give me tasks like:")
    console.print('  "Create a hello.py file"')
    console.print('  "Run the tests"')
    console.print('  "What files are in this directory?"')
    console.print("")
    console.print("-" * 60)
    console.print("")

    # Initialize UI components for this session
    status_display = StatusDisplay()
    verification_panel = VerificationPanel()
    actions_taken = []

    pending_approval = None
    iteration = 0
    max_iterations = 20

    while True:
        try:
            user_input = click.prompt(click.style("You", bold=True, fg="green"), prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\n\nGoodbye!")
            break

        user_input = user_input.strip()

        if user_input.lower() in ("quit", "exit", "q"):
            click.echo("Goodbye!")
            break

        if user_input.lower() == "help":
            click.echo("\nCommands:")
            click.echo("  help     - Show this help")
            click.echo("  status   - Show conversation status")
            click.echo("  clear    - Clear conversation history")
            click.echo("  exit     - End session")
            continue

        if user_input.lower() == "status":
            click.echo(f"\nMessages: {len(messages)}")
            click.echo(f"Model: {model_name}")
            if pending_approval:
                click.echo(f"Pending approval: {pending_approval['command'][:50]}...")
            continue

        if user_input.lower() == "clear":
            messages = [messages[0]]  # Keep system prompt
            click.echo("Conversation cleared.\n")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

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
                    "max_tokens": 2048,
                    "tools": tool_definitions,
                }

                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=request_data,
                    timeout=120,
                )
                response.raise_for_status()

            except requests.RequestException as e:
                show_error(f"API error: {e}")
                break

            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {})

            # Check for native tool calls
            tool_calls_raw = message.get("tool_calls", [])

            if not tool_calls_raw:
                # No tool calls - this is the final response
                content = message.get("content", "")
                if content:
                    console.print("")
                    console.print(f"[bold cyan]Assistant:[/bold cyan] {content}")
                    messages.append({"role": "assistant", "content": content})
                break

            # Process tool calls
            tool_results = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                name = func.get("name")
                arguments = func.get("arguments", {})

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                if name not in tools:
                    tool_results.append({"tool": name, "result": f"Error: Unknown tool '{name}'"})
                    continue

                # Create step name for UI
                step_name = f"{name}({', '.join(f'{k}={v}' for k, v in arguments.items())})"
                step_index = len(actions_taken)
                status_display.add_step(step_name)

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
                try:
                    result = tools[name](**arguments)
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
            else:
                break

        # Show verification summary
        if actions_taken:
            status_display.finish_task(success=(failed_count == 0))
            verification_panel.show_verification(actions_taken)

        if iteration >= max_iterations:
            console.print("[yellow]Max iterations reached[/yellow]")
            break
