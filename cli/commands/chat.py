"""Chat and inference commands."""

import json
import sys
from pathlib import Path

import click
import requests

from ..core.server_manager import ServerManager
from ..utils.chat_history import ChatHistoryManager
from ..utils.config import Config
from ..utils.logging import LogConfig
from ..utils.token_counter import TokenCounter, estimate_tokens


@click.group()
@click.pass_context
def chat(ctx):
    """Chat with a model or run inference."""
    pass


@chat.command()
@click.option(
    "--model", "-m",
    "model_name",
    default=None,
    help="Model name (default: from server)",
)
@click.option(
    "--system", "-s",
    default=None,
    help="System prompt",
)
@click.option(
    "--temperature", "-t",
    type=float,
    default=None,
    help="Temperature (default: from config)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Max tokens per response",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream responses",
)
@click.option(
    "--session",
    "session_id",
    default=None,
    help="Resume existing session by ID",
)
@click.option(
    "--save-history/--no-save-history",
    default=True,
    help="Save chat history",
)
@click.pass_context
def interactive(ctx, model_name, system, temperature, max_tokens, stream, session_id, save_history):
    """Start an interactive chat session."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get configuration
    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)
    system = system or config.get("chat", "system_prompt")
    temperature = temperature or config.get("chat", "temperature", default=0.7)
    max_tokens = max_tokens or config.get("chat", "max_tokens", default=2048)

    base_url = f"http://{host}:{port}"

    # Initialize history and token counter
    history_manager = ChatHistoryManager() if save_history else None
    token_counter = TokenCounter()

    # Check server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            raise Exception("Server returned unhealthy status")
    except requests.RequestException as e:
        click.echo(
            click.style("✗ ", fg="red") +
            f"Server is not running or not responding at {base_url}"
        )
        click.echo("Start the server with: gguf server start")
        raise SystemExit(1)

    # Get available models
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        models = response.json().get("data", [])
        if not models:
            click.echo(click.style("⚠ ", fg="yellow") + "No models available from server")
        else:
            model_name = model_name or models[0].get("id")
            click.echo(f"Using model: {model_name}")
    except Exception:
        pass

    # Load or create session
    session = None
    if history_manager:
        if session_id:
            session = history_manager.load_session(session_id)
            if session:
                click.echo(click.style("✓ ", fg="green") + f"Resumed session: {session.title}")
                click.echo(f"  Messages: {len(session.messages)}")
            else:
                click.echo(click.style("⚠ ", fg="yellow") + f"Session not found: {session_id}")
                click.echo("Creating new session...")

        if not session:
            session = history_manager.create_session(
                model=model_name or "unknown",
                system_prompt=system,
            )
            click.echo(click.style("✓ ", fg="green") + f"Started new session: {session.title}")
            click.echo(f"  Session ID: {session.id}")

        token_counter.start_session(session.id)

    click.echo("\nInteractive Chat Mode")
    click.echo("=" * 50)
    click.echo("Type your message and press Enter.")
    click.echo("Type 'quit' or 'exit' to end the session.")
    click.echo("Type 'clear' to clear conversation history.")
    click.echo("Type 'tokens' to show token usage.\n")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    elif session and session.system_prompt:
        messages.append({"role": "system", "content": session.system_prompt})

    # Load history if resuming session
    if session and session.messages:
        messages = session.messages.copy()
        click.echo(f"Loaded {len(messages)} messages from history.\n")

    while True:
        try:
            user_input = click.prompt(click.style("You", bold=True), prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        user_input = user_input.strip()

        if user_input.lower() in ("quit", "exit"):
            click.echo("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = messages[:1] if (system or (session and session.system_prompt)) else []
            if session:
                history_manager.clear_session(session)
                history_manager.save_session(session)
            click.echo("Conversation cleared.\n")
            continue

        if user_input.lower() == "tokens":
            tokens = token_counter.get_current_tokens()
            click.echo(f"\nToken usage: {tokens.human_readable}")
            lifetime = token_counter.get_lifetime_tokens()
            if lifetime.total_tokens > tokens.total_tokens:
                click.echo(f"Lifetime: {lifetime.human_readable}\n")
            else:
                click.echo()
            continue

        if not user_input:
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})
        if session:
            history_manager.add_message("user", user_input, session)

        # Send request
        try:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

            if stream:
                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    stream=True,
                    timeout=120,
                )
                click.echo(click.style("Assistant", bold=True) + "> ", nl=False)

                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    click.echo(content, nl=False)
                            except json.JSONDecodeError:
                                pass

                click.echo()  # Newline after response
            else:
                response = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                result = response.json()
                assistant_message = result.get("choices", [{}])[0].get("message", {})
                full_content = assistant_message.get("content", "")
                click.echo(click.style("Assistant", bold=True) + f"> {full_content}\n")

            # Add assistant message to history
            messages.append({"role": "assistant", "content": full_content})
            if session:
                history_manager.add_message("assistant", full_content, session)

            # Record token usage (estimate if not provided)
            prompt_tokens = estimate_tokens(" ".join(m["content"] for m in messages[:-1]))
            completion_tokens = estimate_tokens(full_content)
            tokens = token_counter.record_usage(prompt_tokens, completion_tokens)

            # Save session periodically
            if session and save_history:
                history_manager.save_session(session)

        except requests.Timeout:
            click.echo(
                click.style("✗ ", fg="red") + "Request timed out. Try reducing max_tokens."
            )
        except requests.RequestException as e:
            click.echo(click.style("✗ ", fg="red") + f"Request failed: {e}")
        except Exception as e:
            click.echo(click.style("✗ ", fg="red") + f"Error: {e}")

    # Final save
    if session and save_history:
        history_manager.save_session(session)
        click.echo(f"\nSession saved: {session.id}")
        tokens = token_counter.get_current_tokens()
        click.echo(f"Total tokens: {tokens.human_readable}")


@chat.command()
@click.argument("prompt")
@click.option(
    "--model", "-m",
    "model_name",
    default=None,
    help="Model name",
)
@click.option(
    "--temperature", "-t",
    type=float,
    default=0.7,
    help="Temperature",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Max tokens",
)
@click.pass_context
def complete(ctx, prompt, model_name, temperature, max_tokens):
    """Complete a single prompt."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)
    base_url = f"http://{host}:{port}"

    # Check server
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            raise Exception("Server unhealthy")
    except requests.RequestException:
        click.echo(
            click.style("✗ ", fg="red") +
            f"Server not running at {base_url}"
        )
        raise SystemExit(1)

    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        click.echo(content)

    except requests.Timeout:
        click.echo(click.style("✗ ", fg="red") + "Request timed out", err=True)
        raise SystemExit(1)
    except requests.RequestException as e:
        click.echo(click.style("✗ ", fg="red") + f"Request failed: {e}", err=True)
        raise SystemExit(1)


@chat.command()
@click.option(
    "--limit", "-n",
    type=int,
    default=10,
    help="Maximum sessions to show",
)
@click.pass_context
def sessions(ctx, limit):
    """List saved chat sessions."""
    history_manager = ChatHistoryManager()
    sessions = history_manager.list_sessions()

    if not sessions:
        click.echo("No saved chat sessions found.")
        raise SystemExit(0)

    click.echo(f"\nSaved Chat Sessions (showing {min(len(sessions), limit)} of {len(sessions)}):")
    click.echo("=" * 70)

    for i, session in enumerate(sessions[:limit], 1):
        created = session.created_at[:10] if session.created_at else "Unknown"
        msg_count = len(session.messages)
        title = session.title or "Untitled"
        model = session.model or "unknown"

        click.echo(f"\n{i}. {click.style(title, bold=True)}")
        click.echo(f"   ID: {session.id[:8]}... | Model: {model}")
        click.echo(f"   Created: {created} | Messages: {msg_count}")

    click.echo()


@chat.command()
@click.argument("session_id")
@click.pass_context
def show_session(ctx, session_id):
    """Show details of a specific chat session."""
    history_manager = ChatHistoryManager()
    session = history_manager.load_session(session_id)

    if not session:
        click.echo(click.style("✗ ", fg="red") + f"Session not found: {session_id}")
        raise SystemExit(1)

    click.echo(f"\n{click.style('Session Details', bold=True)}")
    click.echo("=" * 50)
    click.echo(f"ID: {session.id}")
    click.echo(f"Title: {session.title or 'Untitled'}")
    click.echo(f"Model: {session.model}")
    click.echo(f"Created: {session.created_at}")
    click.echo(f"Messages: {len(session.messages)}")

    if session.system_prompt:
        click.echo(f"\nSystem Prompt: {session.system_prompt[:100]}...")

    if session.messages:
        click.echo(f"\n{click.style('Recent Messages:', bold=True)}")
        for msg in session.messages[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            click.echo(f"  {role}: {content}...")

    click.echo()


@chat.command()
@click.argument("session_id")
@click.confirmation_option(
    prompt="Are you sure you want to delete this session?"
)
@click.pass_context
def delete_session(ctx, session_id):
    """Delete a chat session."""
    history_manager = ChatHistoryManager()

    if not history_manager.delete_session(session_id):
        click.echo(click.style("✗ ", fg="red") + f"Session not found: {session_id}")
        raise SystemExit(1)

    click.echo(click.style("✓ ", fg="green") + f"Session deleted: {session_id}")
