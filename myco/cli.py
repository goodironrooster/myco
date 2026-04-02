# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""CLI entry point for myco.

The entry point. Assembles the context contract. Calls the local model.
Runs the gate. Writes annotations. Updates the world model.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import requests

from .attractor import AttractorDetector
from .entropy import analyze_entropy, get_priority_files, get_regime_intervention, calculate_substrate_health, get_related_files, read_related_content, compute_internal_entropy, classify_dual_regime, compute_gradient_field, get_fault_line_modules
from .energy import EnergyTracker, get_tracker
from .gate import AutopoieticGate, gate_action
from .git_tools import get_git_tools, GitStatus, GitTools
from .model_provider import check_provider_health, detect_provider, get_all_providers, list_available_models, add_custom_provider, remove_custom_provider, load_custom_providers, PROVIDERS
from .session_log import SessionLogger
from .stigma import StigmaReader, StigmergicAnnotation, migrate_source_annotations
from .tensegrity import TensegrityClassifier
from .trajectory import compute_entropy_trajectory, interpret_trajectory
from .validate import validate_interventions, format_validation_table
from .world import WorldModel

import difflib


@click.group()
@click.version_option(version="0.9.8")
def cli():
    """myco - A CLI that writes itself, reads the soil, and never forgets where it grew."""
    pass


@cli.command()
@click.argument("project_name")
@click.option("--template", "-t", type=click.Choice(["fastapi", "flask", "cli", "empty"]), default="fastapi", help="Project template to use")
@click.option("--myco-path", type=click.Path(exists=True), default=None, help="Path to MYCO installation (default: auto-detect)")
def new(project_name: str, template: str, myco_path: Optional[str]):
    """Create a new MYCO project.
    
    Creates a new project folder with MYCO infrastructure, templates, and convenience scripts.
    
    Example:
        myco new MY_webapp --template fastapi
        myco new MY_cli --template cli
    """
    from pathlib import Path
    import shutil
    
    project_path = Path.cwd() / project_name
    
    # Check if project already exists
    if project_path.exists():
        click.echo(click.style(f"✗ Project '{project_name}' already exists", fg="red"))
        return
    
    click.echo(click.style(f"🍄 Creating new MYCO project: {project_name}", fg="green"))
    click.echo(f"  Template: {template}")
    click.echo(f"  Location: {project_path}")
    click.echo()
    
    # Get MYCO path
    if myco_path is None:
        myco_path = Path(__file__).parent.parent
    else:
        myco_path = Path(myco_path)
    
    # Create project structure
    project_path.mkdir(parents=True)
    (project_path / "src").mkdir()
    (project_path / ".myco").mkdir()
    
    # Copy template files
    template_dir = Path(__file__).parent.parent / "templates" / template
    if not template_dir.exists():
        click.echo(click.style(f"✗ Template '{template}' not found", fg="red"))
        return
    
    # Copy template files with variable substitution
    variables = {
        "project_name": project_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "myco_version": "0.9.8",
    }
    
    files_copied = 0
    for src_file in template_dir.rglob("*"):
        if src_file.is_file():
            rel_path = src_file.relative_to(template_dir)
            dst_file = project_path / rel_path
            
            # Create parent directories
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file with variable substitution for text files
            if dst_file.suffix in [".md", ".yaml", ".bat", ".sh", ".py", ".txt"]:
                content = src_file.read_text(encoding="utf-8")
                for key, value in variables.items():
                    content = content.replace(f"{{{{{key}}}}}", value)
                dst_file.write_text(content, encoding="utf-8")
            else:
                shutil.copy2(src_file, dst_file)
            
            files_copied += 1
    
    # Create symlink to MYCO
    myco_link = project_path / "myco"
    try:
        if sys.platform == "win32":
            # Windows: Use directory junction or symlink
            import subprocess
            try:
                # Try to create symlink (requires admin or developer mode)
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(myco_link), str(myco_path / "myco")],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError:
                # Fallback: copy MYCO folder
                click.echo(click.style("  Note: Symlink creation failed, copying MYCO folder...", fg="yellow"))
                shutil.copytree(myco_path / "myco", myco_link)
        else:
            # Linux/Mac: Use symlink
            myco_link.symlink_to(myco_path / "myco")
        
        click.echo(click.style("  ✓ MYCO symlink created", fg="green"))
    except Exception as e:
        click.echo(click.style(f"  Note: Could not create symlink: {e}", fg="yellow"))
        click.echo("  You can manually create symlink or copy MYCO folder")
    
    # Initialize world model
    world = WorldModel.load(project_path)
    world.save()
    click.echo(click.style("  ✓ World model initialized", fg="green"))
    
    click.echo()
    click.echo(click.style(f"✓ Project '{project_name}' created successfully!", fg="green"))
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    if sys.platform == "win32":
        click.echo("  myco_run.bat run \"Add your first feature\"")
    else:
        click.echo("  ./myco_run.sh run \"Add your first feature\"")
    click.echo("  myco_run.bat entropy  # Check substrate health")


@cli.command()
@click.option("--myco-path", type=click.Path(exists=True), default=None, help="Path to MYCO installation")
def init(myco_path: Optional[str]):
    """Initialize MYCO in an existing project.
    
    Adds MYCO infrastructure to the current directory.
    
    Example:
        cd my_existing_project
        myco init
    """
    from pathlib import Path
    import shutil
    
    project_path = Path.cwd()
    
    click.echo(click.style("🍄 Initializing MYCO in existing project", fg="green"))
    click.echo(f"  Location: {project_path}")
    click.echo()
    
    # Get MYCO path
    if myco_path is None:
        myco_path = Path(__file__).parent.parent
    else:
        myco_path = Path(myco_path)
    
    # Create .myco directory
    myco_dir = project_path / ".myco"
    myco_dir.mkdir(exist_ok=True)
    
    # Initialize world model
    world = WorldModel.load(project_path)
    world.save()
    click.echo(click.style("  ✓ World model initialized", fg="green"))
    
    # Analyze existing codebase
    click.echo("  Analyzing existing codebase...")
    entropy_report = analyze_entropy(project_path)
    world.entropy_baseline = entropy_report.global_entropy
    world.crystallized_modules = entropy_report.crystallized[:10]
    world.diffuse_modules = entropy_report.diffuse[:10]
    world.save()
    
    click.echo(f"  ✓ Baseline entropy: {entropy_report.global_entropy:.3f}")
    click.echo(f"    Crystallized: {len(entropy_report.crystallized)} modules")
    click.echo(f"    Dissipative: {len(entropy_report.dissipative)} modules")
    click.echo(f"    Diffuse: {len(entropy_report.diffuse)} modules")
    
    # Create symlink to MYCO
    myco_link = project_path / "myco"
    if not myco_link.exists():
        try:
            if sys.platform == "win32":
                import subprocess
                try:
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(myco_link), str(myco_path / "myco")],
                        check=True, capture_output=True
                    )
                    click.echo(click.style("  ✓ MYCO symlink created", fg="green"))
                except subprocess.CalledProcessError:
                    click.echo(click.style("  Note: Could not create symlink, copy MYCO folder manually", fg="yellow"))
            else:
                myco_link.symlink_to(myco_path / "myco")
                click.echo(click.style("  ✓ MYCO symlink created", fg="green"))
        except Exception as e:
            click.echo(click.style(f"  Note: Could not create symlink: {e}", fg="yellow"))
    
    # Create convenience script
    if sys.platform == "win32":
        bat_file = project_path / "myco_run.bat"
        if not bat_file.exists():
            bat_file.write_text(
                "@echo off\n"
                "setlocal enabledelayedexpansion\n"
                "set \"SCRIPT_DIR=%~dp0\"\n"
                "cd /d \"%SCRIPT_DIR%\"\n"
                "python -m myco.cli %*\n"
                "endlocal\n",
                encoding="utf-8"
            )
            click.echo(click.style("  ✓ myco_run.bat created", fg="green"))
    else:
        sh_file = project_path / "myco_run.sh"
        if not sh_file.exists():
            sh_file.write_text(
                "#!/bin/bash\n"
                "SCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\n"
                "cd \"$SCRIPT_DIR\" || exit 1\n"
                "python3 -m myco.cli \"$@\"\n",
                encoding="utf-8"
            )
            sh_file.chmod(0o755)
            click.echo(click.style("  ✓ myco_run.sh created", fg="green"))
    
    click.echo()
    click.echo(click.style("✓ MYCO initialized successfully!", fg="green"))
    click.echo()
    click.echo("Next steps:")
    click.echo("  myco_run.bat entropy  # Check substrate health")
    click.echo("  myco_run.bat report   # Generate health report")
    click.echo("  myco_run.bat run \"Your first task\"")


@cli.command()
@click.argument("task")
@click.option("--verbose", "-v", is_flag=True, help="Show progress and details")
@click.option("--max-iterations", "-i", type=int, default=10, help="Maximum iterations")
@click.option("--model", "-m", type=str, default=None, help="Model to use")
@click.option("--base-url", "-u", type=str, default="http://127.0.0.1:1234", help="Server base URL")
@click.option("--confirm", "-c", is_flag=True, help="Confirm before file modifications")
@click.option("--auto-commit", is_flag=True, help="Auto-commit changes with descriptive messages")
@click.option("--preview", is_flag=True, help="Show diff preview without applying changes")
def run(task: str, verbose: bool, max_iterations: int, model: Optional[str], base_url: str, confirm: bool, auto_commit: bool, preview: bool):
    """Run myco to complete a task.

    myco reads the thermodynamic state of your codebase, applies the minimum
    structural intervention needed, and leaves a trace for the next session.

    Example:

        myco run "Extract the payment retry logic into a separate module"
        myco run "Refactor the auth module" --confirm  # Confirm before changes
        myco run "Add type hints" --auto-commit  # Auto-commit changes
    """
    # Initialize components
    project_root = Path.cwd()
    world = WorldModel.load(project_root)
    world.start_session()

    tracker = get_tracker()
    tracker.start_session()

    # Initialize session logger
    session_logger = SessionLogger(project_root)
    session_logger.log_session_start(task)
    
    # Migrate source annotations to sidecar (one-time operation)
    migrated = migrate_source_annotations(project_root)
    if migrated > 0 and verbose:
        click.echo(click.style(f"✓ Migrated {migrated} annotations to sidecar file", fg="bright_black"))

    # Log confirmation mode status
    if confirm:
        session_logger.log(
            "confirmation_mode",
            "Confirmation mode enabled - user approval required for file changes",
            level="INFO"
        )
    
    # Log auto-commit status
    if auto_commit:
        session_logger.log(
            "auto_commit",
            "Auto-commit enabled - changes will be committed automatically",
            level="INFO"
        )

    # Check server connectivity before starting
    click.echo(click.style("🍄 myco", fg="green", bold=True))
    click.echo(f"Task: {task}")
    click.echo()

    click.echo(click.style("Checking model server...", fg="cyan"))
    
    # Detect provider and check health
    provider_name = detect_provider(base_url)
    is_healthy, status_msg = check_provider_health(base_url, timeout=5)
    
    if is_healthy:
        if provider_name:
            click.echo(click.style(f"✓ {provider_name.title()} server is ready", fg="green"))
        else:
            click.echo(click.style(f"✓ Model server is ready", fg="green"))
        click.echo(click.style(f"  {status_msg}", fg="bright_black"))
    else:
        click.echo(click.style(
            f"✗ {status_msg}",
            fg="red",
            bold=True
        ))
        
        # Provide helpful suggestions based on common providers
        click.echo(click.style("\nTry these providers:", fg="yellow"))
        click.echo("  LM Studio:  python -m myco.cli run TASK -u http://localhost:1234")
        click.echo("  Ollama:     python -m myco.cli run TASK -u http://localhost:11434")
        click.echo("  LocalAI:    python -m myco.cli run TASK -u http://localhost:8080")
        
        session_logger.log(
            "server_unreachable",
            f"Could not connect to model server at {base_url}",
            level="ERROR"
        )
        session_logger.log_session_end(iterations=0, tokens=0, joules=0, entropy_delta=0)
        return
    
    # Check git repository status
    git_tools = get_git_tools(project_root)
    git_status = git_tools.get_status()
    if git_status.is_repo:
        if verbose:
            click.echo(click.style(f"✓ Git repository: {git_status.branch}", fg="green"))
            if git_status.dirty:
                click.echo(click.style(
                    f"  {len(git_status.modified_files)} modified, "
                    f"{len(git_status.untracked_files)} untracked",
                    fg="yellow"
                ))
    else:
        if verbose:
            click.echo(click.style("Git: Not a repository", fg="bright_black"))
    
    # Log git status
    session_logger.log(
        "git_status",
        f"Git repository: {git_status.is_repo}",
        level="INFO",
        git_status=git_status.to_dict()
    )
    
    # Assemble context contract
    context = assemble_context(project_root, world, task)

    if verbose:
        click.echo(click.style("Context Contract", fg="cyan", bold=True))
        click.echo(click.style("=" * 50, fg="cyan"))
        click.echo(format_context(context))
        click.echo()

    # Initialize attractor detector
    attractor = AttractorDetector()

    # Initialize gate
    gate = AutopoieticGate(project_root, world)

    # Run the agent loop

    iteration = 0
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(context)
        },
        {
            "role": "user",
            "content": task
        }
    ]

    final_response = ""
    
    while iteration < max_iterations:
        iteration += 1

        if verbose:
            click.echo(click.style(f"\n{'='*50}", fg="bright_cyan"))
            click.echo(click.style(f"Iteration {iteration}/{max_iterations}", fg="bright_cyan", bold=True))
            click.echo(click.style(f"{'='*50}", fg="bright_cyan"))

        # Call the model with retry logic
        max_retries = 2
        retry_count = 0
        response = None
        content = ""
        session_streaming_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while retry_count <= max_retries:
            try:
                if verbose:
                    # Use streaming for verbose mode
                    click.echo(click.style("\n🤖 Model response: ", fg="magenta"), nl=False)
                    response = requests.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": model or "default",
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 2048,
                            "stream": True,
                        },
                        timeout=300,  # 5 minutes for CPU mode
                        stream=True
                    )
                    response.raise_for_status()

                    # Process streaming response
                    streaming_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data = line_str[6:]
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    delta = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                    if delta:
                                        content += delta
                                        click.echo(delta, nl=False)

                                    # Capture usage data if present (sent in final chunk)
                                    usage = chunk.get('usage')
                                    if usage:
                                        streaming_tokens['prompt_tokens'] = usage.get('prompt_tokens', 0)
                                        streaming_tokens['completion_tokens'] = usage.get('completion_tokens', 0)
                                        streaming_tokens['total_tokens'] = usage.get('total_tokens', 0)
                                except json.JSONDecodeError:
                                    continue
                    click.echo()  # Newline after streaming

                    # Store streaming tokens for later use
                    session_streaming_tokens = streaming_tokens
                    break
                else:
                    # Non-streaming for non-verbose mode
                    response = requests.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": model or "default",
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 2048,
                        },
                        timeout=300,  # 5 minutes for CPU mode
                    )
                    response.raise_for_status()
                    break  # Success, exit retry loop
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                if retry_count <= max_retries:
                    if verbose:
                        click.echo(click.style(
                            f"Connection failed, retrying ({retry_count}/{max_retries})...",
                            fg="yellow"
                        ))
                    session_logger.log(
                        "model_retry",
                        f"Connection failed, retrying ({retry_count}/{max_retries})",
                        level="WARNING",
                        error=str(e)
                    )
                else:
                    click.echo(click.style(
                        f"Error: Could not connect to model server at {base_url}",
                        fg="red",
                        bold=True
                    ))
                    click.echo(click.style(
                        f"  Make sure the server is running: gguf server start",
                        fg="yellow"
                    ))
                    world.add_intention(f"Failed to complete task (connection error): {task[:50]}...")
                    world.save()
                    session_logger.log_session_end(
                        iterations=iteration,
                        tokens=0,
                        joules=0,
                        entropy_delta=0
                    )
                    return
            except requests.exceptions.Timeout as e:
                retry_count += 1
                if retry_count <= max_retries:
                    if verbose:
                        click.echo(click.style(
                            f"Request timed out, retrying ({retry_count}/{max_retries})...",
                            fg="yellow"
                        ))
                    session_logger.log(
                        "model_retry",
                        f"Request timed out, retrying ({retry_count}/{max_retries})",
                        level="WARNING",
                        error=str(e)
                    )
                else:
                    click.echo(click.style(
                        f"Error: Request timed out after 120 seconds",
                        fg="red",
                        bold=True
                    ))
                    world.add_intention(f"Failed to complete task (timeout): {task[:50]}...")
                    world.save()
                    session_logger.log_session_end(
                        iterations=iteration,
                        tokens=0,
                        joules=0,
                        entropy_delta=0
                    )
                    return
            except requests.RequestException as e:
                click.echo(click.style(f"Error: API request failed - {e}", fg="red"))
                world.add_intention(f"Failed to complete task: {task[:50]}...")
                world.save()
                session_logger.log_session_end(
                    iterations=iteration,
                    tokens=0,
                    joules=0,
                    entropy_delta=0
                )
                return

        # Parse response (only needed for non-streaming mode)
        if not verbose:
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                click.echo(click.style(f"Error: Invalid JSON response from server: {e}", fg="red"))
                world.add_intention(f"Failed to complete task (invalid response): {task[:50]}...")
                world.save()
                session_logger.log_session_end(
                    iterations=iteration,
                    tokens=0,
                    joules=0,
                    entropy_delta=0
                )
                return

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")

        # Check for empty response
        if not content:
            click.echo(click.style(
                "Warning: Model returned empty response, continuing...",
                fg="yellow"
            ))
            session_logger.log(
                "model_empty_response",
                "Model returned empty response",
                level="WARNING"
            )
            # Add nudge to continue
            messages.append({
                "role": "user",
                "content": "Please provide a response. If you need to use tools, output the tool call JSON."
            })
            continue

        # Track energy with accurate token counts
        if verbose:
            # Streaming mode - use captured usage data if available
            tokens = session_streaming_tokens.get('total_tokens', 0)
            if tokens == 0:
                # Fallback to estimation if no usage data
                tokens = len(content) // 4
        else:
            # Non-streaming mode - use actual usage data
            usage = result.get("usage", {})
            tokens = usage.get("total_tokens", len(content) // 4)
        tracker.record_inference(tokens)
        
        # Check for attractor lock-in
        attractor.add_output(content)
        if attractor.is_locked():
            if verbose:
                click.echo(click.style(
                    f"\n⚠️  Attractor detected: {attractor.get_attractor_name()}",
                    fg="yellow"
                ))
            
            perturbation = attractor.select_perturbation()
            guidance = attractor.apply_perturbation(perturbation)
            
            if verbose:
                click.echo(click.style(guidance, fg="yellow"))
            
            world.add_attractor(attractor.get_attractor_name())
            
            # Add perturbation guidance to messages
            messages.append({
                "role": "system",
                "content": f"{guidance}\n\nApply this perspective to the current task."
            })
            continue

        # In verbose mode, we already showed the streaming response
        # For non-verbose, show the full response
        if not verbose and len(content) > 0:
            click.echo(click.style(f"\n🤖 Model ({len(content)} chars):", fg="magenta"))
            click.echo(content[:800] if len(content) > 800 else content)

        # Parse tool calls and execute
        tool_results = parse_and_execute_tools(content, project_root, gate, session_logger, verbose, confirm, git_tools, auto_commit, task, preview)

        if tool_results:
            # Build results text
            results_text = "\n\n".join(
                f"[{r['tool']} result]:\n{r['result']}"
                for r in tool_results
            )
            
            # Check for blocked actions and add feedback
            blocked_feedback = ""
            for r in tool_results:
                if r['result'].startswith("BLOCKED:"):
                    blocked_feedback += f"\n\n⚠️  ACTION BLOCKED: {r['tool']}\n"
                    blocked_feedback += f"Reason: {r['result'].replace('BLOCKED: ', '')}\n"
                    blocked_feedback += "The autopoietic gate blocked this action because it would degrade the codebase.\n"
                    blocked_feedback += "Propose an alternative approach that preserves structural integrity.\n"
            
            # Add feedback to conversation
            messages.append({
                "role": "user",
                "content": f"Tool execution completed:\n\n{results_text}{blocked_feedback}\n\nContinue with the task. If complete, summarize what was done."
            })
        else:
            # No tool calls - assume final response
            final_response = content
            break
    
    # End session
    session_energy = tracker.end_session()

    # Update world model
    entropy_report = analyze_entropy(project_root)
    world.end_session(
        entropy_baseline=entropy_report.global_entropy,
        crystallized=entropy_report.crystallized,
        diffuse=entropy_report.diffuse,
        tensegrity_violations=0  # Would need to calculate
    )

    # Step 3: Update self-entropy (myco's own substrate)
    myco_dir = project_root / "myco"
    if myco_dir.exists():
        myco_report = analyze_entropy(myco_dir)
        world.update_self_entropy(
            baseline=myco_report.global_entropy,
            crystallized_count=len(myco_report.crystallized)
        )

    # Log session end
    session_logger.log_session_end(
        iterations=iteration,
        tokens=session_energy.total_tokens,
        joules=session_energy.total_joules,
        entropy_delta=world.entropy_trend
    )

    # Print session summary
    click.echo()
    click.echo(click.style("Session Complete", fg="green", bold=True))
    click.echo(click.style("=" * 50, fg="green"))
    click.echo(f"Final response: {final_response[:200] if final_response else 'Task completed'}...")
    click.echo()
    click.echo(f"Tokens: {session_energy.total_tokens}")
    click.echo(f"Joules: {session_energy.total_joules:.2f}")
    click.echo(f"J/tok: {session_energy.total_joules / max(session_energy.total_tokens, 1):.6f}")
    click.echo(f"Entropy delta: {world.entropy_trend:+.3f}")
    click.echo(f"Files touched: {iteration}")

    if world.open_intentions:
        click.echo()
        click.echo(click.style("Open intentions:", fg="yellow"))
        for intention in world.open_intentions[-5:]:
            click.echo(f"  - {intention}")


def assemble_context(project_root: Path, world: WorldModel, task: str = "") -> dict:
    """Assemble the context contract.

    Args:
        project_root: Root directory of the project
        world: World model instance
        task: Current task (used for regime analysis of mentioned files)

    Returns:
        Context dictionary
    """
    # Get entropy gradient
    entropy_report = analyze_entropy(project_root)

    # Calculate substrate health score
    substrate_health = calculate_substrate_health(project_root)

    # Get priority files based on entropy regime
    priority_files = get_priority_files(project_root, top_n=5)

    # Get regime analysis for priority files with ACTUAL entropy values
    regime_analysis = []
    related_files_map = {}  # Map file -> related files

    # Build import graph once for efficiency
    from .entropy import ImportGraphBuilder, EntropyCalculator
    builder = ImportGraphBuilder(project_root)
    builder.scan()
    calculator = EntropyCalculator(builder)

    for pf in priority_files:
        file_path = project_root / pf["file"]
        if file_path.exists():
            # Calculate actual H for this module
            module_name = builder._path_to_module_name(file_path)
            H_structural = calculator.calculate_module_entropy(module_name)

            # Calculate internal entropy
            internal_metrics = compute_internal_entropy(file_path)
            H_internal = internal_metrics["H_internal"]

            # Get dual regime classification (Proposal 1)
            dual_regime = classify_dual_regime(H_structural, H_internal)

            # Build regime analysis with dual regime info
            regime = get_regime_intervention(file_path, H_structural)
            regime["H_structural"] = H_structural
            regime["H_internal"] = H_internal
            regime["internal_regime"] = dual_regime["internal_regime"]
            regime["combined_regime"] = dual_regime["combined_regime"]
            regime["dual_guidance"] = dual_regime["guidance"]
            regime_analysis.append(regime)

            # Get related files for multi-file context
            related = get_related_files(project_root, file_path, max_files=3)
            if related:
                related_files_map[str(file_path)] = related

    # Read content of related files for multi-file context
    related_content_map = {}
    for file_path, relations in related_files_map.items():
        content_list = read_related_content(project_root, relations, max_content_length=500)
        if content_list:
            related_content_map[file_path] = content_list

    # Get stigmergic surface (files touched in last 3 sessions)
    stigmergic_surface = []
    for py_file in project_root.rglob("*.py"):
        if any(part.startswith('.') for part in py_file.parts):
            continue
        try:
            reader = StigmaReader(py_file, project_root)
            annotation = reader.read_annotation()
            if annotation and annotation.age <= 3:
                stigmergic_surface.append({
                    "file": str(py_file.relative_to(project_root)),
                    "annotation": annotation,
                })
        except (SyntaxError, FileNotFoundError):
            continue

    # Get tensegrity status
    tensegrity = TensegrityClassifier(project_root)
    tensegrity.scan()
    violations = tensegrity.get_violations()

    # Get gradient field (Proposal 3)
    gradient_field = compute_gradient_field(project_root, threshold=0.3)

    # Step 1: Self-entropy projection (myco analyzes its own substrate)
    # This enables self-monitoring during agent runs
    myco_dir = project_root / "myco"
    myco_entropy = None
    myco_crystallized = []
    if myco_dir.exists():
        myco_report = analyze_entropy(myco_dir)
        myco_entropy = {
            "global_entropy": myco_report.global_entropy,
            "crystallized_count": len(myco_report.crystallized),
            "dissipative_count": len(myco_report.dissipative),
            "diffuse_count": len(myco_report.diffuse),
            "total_modules": myco_report.module_count,
            "status": "crystallized" if myco_report.global_entropy < 0.3 else "dissipative" if myco_report.global_entropy <= 0.75 else "diffuse",
        }
        myco_crystallized = myco_report.crystallized[:5]  # Top 5 crystallized

    return {
        "world_model": world.to_context_dict(),
        "entropy_gradient": {
            "global_entropy": entropy_report.global_entropy,
            "top_deviations": [
                {"module": name, "entropy": h, "drift": drift}
                for name, h, drift in entropy_report.top_deviations[:5]
            ],
            "priority_files": priority_files,
            "regime_analysis": regime_analysis,
            "substrate_health": substrate_health,
            "related_files": related_files_map,
            "related_content": related_content_map,
            "gradient_field": {
                "fault_lines": [fl.to_dict() for fl in gradient_field.fault_lines[:10]],
                "module_stress": [ms.to_dict() for ms in sorted(gradient_field.module_stress, key=lambda m: m.mean_gradient, reverse=True)[:5]],
                "total_edges": gradient_field.total_edges,
                "fault_line_count": gradient_field.fault_line_count,
                "mean_gradient": gradient_field.mean_gradient,
            }
        },
        "self_entropy": myco_entropy,  # Step 1: myco's own substrate health
        "stigmergic_surface": stigmergic_surface,
        "gate_status": {
            "entropy_budget": 0.5 - entropy_report.global_entropy,  # Remaining delta
            "tensegrity_violations": len(violations),
            "attractor_status": "clear",
            "gradient_field": {
                "fault_line_count": gradient_field.fault_line_count,
                "mean_gradient": gradient_field.mean_gradient,
            }
        },
        "open_intentions": world.open_intentions
    }


def format_context(context: dict) -> str:
    """Format context for display.
    
    Args:
        context: Context dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    
    # World model
    wm = context["world_model"]
    lines.append(f"[WORLD MODEL]")
    lines.append(f"  Sessions: {wm['session_count']}")
    lines.append(f"  Entropy baseline: {wm['entropy_baseline']:.3f}")
    lines.append(f"  Entropy trend: {wm['entropy_trend']:+.3f}")
    lines.append(f"  Crystallized: {len(wm['crystallized_modules'])} modules")
    lines.append(f"  Open intentions: {len(wm['open_intentions'])}")
    lines.append("")

    # Substrate health
    health = context["entropy_gradient"].get("substrate_health", {})
    if health:
        lines.append(f"[SUBSTRATE HEALTH]")
        lines.append(f"  Score: {health.get('health_score', 0):.2f}/1.00 ({health.get('status', 'unknown')})")
        lines.append(f"  {health.get('status_message', '')}")
        lines.append(f"  Modules: {health.get('metrics', {}).get('total_modules', 0)} total, "
                    f"{health.get('metrics', {}).get('dissipative_count', 0)} dissipative")
        lines.append("")

    # Entropy gradient with dual regime (Proposal 1)
    lines.append(f"[ENTROPY GRADIENT]")
    for dev in context["entropy_gradient"]["top_deviations"]:
        lines.append(f"  {dev['module']}: H={dev['entropy']:.3f}, drift={dev['drift']:+.3f}")
    lines.append("")

    # Regime analysis with internal entropy
    regime_data = context["entropy_gradient"].get("regime_analysis", [])
    if regime_data:
        lines.append(f"[REGIME ANALYSIS - DUAL CLASSIFICATION]")
        for reg in regime_data[:5]:  # Show top 5
            lines.append(f"  {reg['file']}:")
            lines.append(f"    H_structural: {reg.get('H_structural', reg.get('H', 0)):.3f} ({reg.get('structural_regime', reg.get('regime', 'unknown'))})")
            lines.append(f"    H_internal:   {reg.get('H_internal', 0):.3f} ({reg.get('internal_regime', 'unknown')})")
            lines.append(f"    Combined:     {reg.get('combined_regime', 'mixed')}")
            if reg.get('dual_guidance'):
                lines.append(f"    Guidance: {reg['dual_guidance'][:80]}...")
        lines.append("")

    # Gate status
    gs = context["gate_status"]
    lines.append(f"[GATE STATUS]")
    lines.append(f"  Entropy budget: {gs['entropy_budget']:+.3f}")
    lines.append(f"  Tensegrity violations: {gs['tensegrity_violations']}")
    lines.append(f"  Attractor status: {gs['attractor_status']}")

    # Gradient field (Proposal 3)
    gf = context["entropy_gradient"].get("gradient_field", {})
    if gf and gf.get("fault_line_count", 0) > 0:
        lines.append("")
        lines.append(f"[GRADIENT FIELD - STRUCTURAL STRESS]")
        lines.append(f"  Fault lines: {gf['fault_line_count']} (edges with gradient > 0.3)")
        lines.append(f"  Mean gradient: {gf['mean_gradient']:.3f}")

        # Show top fault lines
        fault_lines = gf.get("fault_lines", [])[:5]
        if fault_lines:
            lines.append("  Top fault lines:")
            for fl in fault_lines:
                lines.append(
                    f"    {fl['importer']} (H={fl['H_importer']:.2f}) → "
                    f"{fl['imported']} (H={fl['H_imported']:.2f}) | "
                    f"Δ={fl['gradient']:.2f}"
                )

        # Show highest stress modules
        stressed = gf.get("module_stress", [])[:3]
        if stressed:
            lines.append("  Highest stress modules:")
            for ms in stressed:
                if ms["fault_line_count"] > 0:
                    lines.append(
                        f"    {ms['module']}: stress={ms['mean_gradient']:.2f} "
                        f"({ms['fault_line_count']} fault lines)"
                    )

    # Step 1: Self-entropy projection (myco's own substrate health)
    self_ent = context.get("self_entropy")
    if self_ent:
        lines.append("")
        lines.append(f"[MYCO SUBSTRATE (self-entropy)]")
        lines.append(f"  Status: {self_ent['status']} (H={self_ent['global_entropy']:.3f})")
        lines.append(f"  Modules: {self_ent['total_modules']} total, "
                    f"{self_ent['crystallized_count']} crystallized, "
                    f"{self_ent['dissipative_count']} dissipative, "
                    f"{self_ent['diffuse_count']} diffuse")
        if self_ent['status'] == 'crystallized':
            lines.append(f"  ⚠️  WARNING: myco itself is crystallized. Restructure before adding features.")

    return '\n'.join(lines)


def build_system_prompt(context: dict) -> str:
    """Build the system prompt for the model.

    Args:
        context: Context dictionary

    Returns:
        System prompt string
    """
    # Build substrate health section
    health_section = ""
    health = context['entropy_gradient'].get('substrate_health', {})
    if health:
        health_section = f"""
SUBSTRATE HEALTH: {health.get('health_score', 0):.2f}/1.00 ({health.get('status', 'unknown')})
  {health.get('status_message', '')}
  """

    # Build regime guidance section with dual regime (Proposal 1)
    regime_section = ""
    regime_analysis = context['entropy_gradient'].get('regime_analysis', [])
    if regime_analysis:
        regime_section = "\n\nENTROPY REGIME GUIDANCE (DUAL CLASSIFICATION):\n"
        for ra in regime_analysis[:3]:
            H_struct = ra.get('H_structural', ra.get('H', 0))
            H_int = ra.get('H_internal', 0)
            combined = ra.get('combined_regime', ra.get('regime', 'unknown'))
            structural_regime = ra.get('structural_regime', ra.get('regime', 'unknown'))
            internal_regime = ra.get('internal_regime', 'unknown')
            
            regime_section += f"- {ra['file']}: {combined}\n"
            regime_section += f"  Structural: {structural_regime} (H={H_struct:.2f}), Internal: {internal_regime} (H={H_int:.2f})\n"
            # Use dual_guidance if available, otherwise fall back to original guidance
            guidance = ra.get('dual_guidance', ra.get('guidance', ''))
            regime_section += f"  {guidance}\n"

    # Build open intentions section
    intentions_section = ""
    open_intentions = context.get('open_intentions', [])
    if open_intentions:
        intentions_section = "\n\nOPEN INTENTIONS (structural pressures from previous sessions):\n"
        for intention in open_intentions[:5]:
            intentions_section += f"- {intention}\n"
        intentions_section += "\nConsider these intentions alongside the current task. "
        intentions_section += "They represent structural pressures that may need attention.\n"

    # Build priority files section
    priority_section = ""
    priority_files = context['entropy_gradient'].get('priority_files', [])
    if priority_files:
        priority_section = "\n\nPRIORITY FILES (focus on these):\n"
        for pf in priority_files[:3]:
            priority_section += f"- {pf['file']} ({pf['reason']}): {pf['action_hint']}\n"

    # Build related files section for multi-file context
    related_section = ""
    related_files = context['entropy_gradient'].get('related_files', {})
    related_content = context['entropy_gradient'].get('related_content', {})
    if related_files:
        related_section = "\n\nFILE RELATIONSHIPS (import graph):\n"
        for file_path, relations in list(related_files.items())[:3]:
            related_section += f"- {file_path}:\n"
            for rel in relations:
                related_section += f"  • {rel['relationship']}: {rel['file']}\n"
            # Include content preview for related files
            if file_path in related_content:
                for content_item in related_content[file_path][:2]:
                    related_section += f"\n  [{content_item['relationship']} {content_item['file']}]:\n"
                    related_section += f"  ```python\n  {content_item['content'][:300]}\n  ```\n"

    # Build gradient field section (Proposal 3)
    gradient_section = ""
    gf = context['entropy_gradient'].get('gradient_field', {})
    if gf and gf.get('fault_line_count', 0) > 0:
        gradient_section = "\n\nGRADIENT FIELD (structural stress concentrations):\n"
        gradient_section += f"  Fault lines: {gf['fault_line_count']} | Mean gradient: {gf['mean_gradient']:.2f}\n"
        fault_lines = gf.get('fault_lines', [])[:5]
        for fl in fault_lines:
            gradient_section += (
                f"  - {fl['importer']} (H={fl['H_importer']:.2f}) → "
                f"{fl['imported']} (H={fl['H_imported']:.2f}) | Δ={fl['gradient']:.2f}\n"
            )
        gradient_section += "  These are structural stress points. Consider refactoring to reduce gradient.\n"

    # Step 1: Self-entropy section (myco's own substrate)
    self_entropy_section = ""
    self_ent = context.get('self_entropy')
    if self_ent:
        self_entropy_section = f"\n\nMYCO SUBSTRATE (self-analysis):\n"
        self_entropy_section += f"  myco itself is {self_ent['status']} (H={self_ent['global_entropy']:.3f})\n"
        self_entropy_section += f"  {self_ent['crystallized_count']}/{self_ent['total_modules']} modules crystallized\n"
        if self_ent['status'] == 'crystallized':
            self_entropy_section += f"  ⚠️  CRITICAL: myco is crystallized. Apply decompose to itself before adding features.\n"

    return f"""You are myco, a coding agent that operates by reading thermodynamic traces embedded in source code.

Your purpose is NOT to produce code. Your purpose is to TEND THE SUBSTRATE.

You complete tasks as a SIDE EFFECT of tending the substrate. Not the other way around.{health_section}

Current state:
- Entropy baseline: {context['entropy_gradient']['global_entropy']:.3f}
- Tensegrity violations: {context['gate_status']['tensegrity_violations']}
- Fault lines: {context['gate_status'].get('gradient_field', {}).get('fault_line_count', 0)}
- myco status: {self_ent['status'] if self_ent else 'unknown'}{priority_section}{related_section}{regime_section}{gradient_section}{self_entropy_section}{intentions_section}

ENTROPY REGIME RULES (INVARIANTS):
1. Crystallized modules (H < 0.3): Do NOT add features. Apply decompose or interface_inversion FIRST.
2. Dissipative modules (0.3 ≤ H ≤ 0.75): Safe to make changes. Keep them minimal.
3. Diffuse modules (H > 0.75): Apply compression_collapse or tension_extraction before adding features.

GRADIENT FIELD RULES:
- Fault lines (gradient > 0.3) indicate structural stress between modules
- High-stress modules are prone to bugs and frequent changes
- When modifying a module on a fault line, consider reducing the gradient
- Prefer changes that balance entropy across import edges

You have access to these tools:
1. write_file(path, content) - Create or overwrite a file
2. read_file(path) - Read file contents
3. edit_file(path, old_text, new_text) - Replace text in a file
4. list_files(path) - List files in directory

TO USE A TOOL, OUTPUT EXACTLY THIS JSON FORMAT (no other text, no markdown):
{{"name": "write_file", "arguments": {{"path": "filename.py", "content": "file content here"}}}}

IMPORTANT:
- Always include stigmergic annotation as first line: # ⊕ H:x.xx | press:type | age:0 | drift:+/-x.xx
- Valid press types: decompose, interface_inversion, tension_extraction, compression_collapse, entropy_drain, attractor_escape, none
- Apply MINIMUM structural intervention needed
- One tool at a time
- When task is complete, provide a brief summary

Example tool call:
{{"name": "write_file", "arguments": {{"path": "example.py", "content": "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\\ndef hello():\\n    pass"}}}}
"""


def parse_and_execute_tools(
    content: str,
    project_root: Path,
    gate: AutopoieticGate,
    session_logger: SessionLogger,
    verbose: bool,
    confirm: bool = False,
    git_tools: Optional[GitTools] = None,
    auto_commit: bool = False,
    task_description: str = "",
    preview: bool = False
) -> list[dict]:
    """Parse tool calls from content and execute them.

    Handles multiple formats:
    1. JSON: {"name": "write_file", "arguments": {"path": "x.py", "content": "..."}}
    2. Markdown code blocks: ```python\n# code here\n```
    3. Function calls: write_file(path="x.py", content="...")

    Args:
        content: Model output content
        project_root: Project root directory
        gate: Autopoietic gate instance
        session_logger: Session logger instance
        verbose: Verbose flag
        confirm: Require confirmation before file modifications

    Returns:
        List of tool results
    """
    import re

    tool_results = []
    blocked_actions = []  # Track blocked actions for feedback

    # Pattern 1: JSON tool calls (preferred format)
    json_pattern = r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'

    for match in re.finditer(json_pattern, content, re.DOTALL):
        try:
            tool_name = match.group(1)
            args = json.loads(match.group(2))
            result = execute_tool(tool_name, args, project_root, gate, confirm, verbose, git_tools, auto_commit, task_description, preview, session_logger)
            tool_results.append({"tool": tool_name, "result": result})
            
            # Track blocked actions
            if result.startswith("BLOCKED:"):
                blocked_actions.append({
                    "tool": tool_name,
                    "args": args,
                    "reason": result.replace("BLOCKED: ", "")
                })
            
            _log_and_show_tool(session_logger, tool_name, args, result, verbose)
        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                click.echo(click.style(f"Failed to parse JSON tool call: {e}", fg="red"))

    # Pattern 2: Markdown code blocks with file creation hint
    if not tool_results:
        markdown_pattern = r'```(\w+)?\s*\n(?:#.*?file[:\s]+([^\n]+)\n)?(.*?)```'
        for match in re.finditer(markdown_pattern, content, re.DOTALL | re.IGNORECASE):
            lang = match.group(1) or ''
            filename = match.group(2) or ''
            code_content = match.group(3).strip()

            if not filename:
                file_match = re.search(r'#\s*(?:file|File|FILENAME)[:\s]*([^\n]+)', code_content)
                if file_match:
                    filename = file_match.group(1).strip()

            if filename and (filename.endswith('.py') or lang.lower() in ('python', 'py')):
                filename = filename.strip('`*"\'')
                args = {"path": filename, "content": code_content}
                result = execute_tool("write_file", args, project_root, gate, confirm, verbose, git_tools, auto_commit, task_description, preview, session_logger)
                tool_results.append({"tool": "write_file", "result": result})
                
                if result.startswith("BLOCKED:"):
                    blocked_actions.append({
                        "tool": "write_file",
                        "args": args,
                        "reason": result.replace("BLOCKED: ", "")
                    })
                
                _log_and_show_tool(session_logger, "write_file", args, result, verbose)
                break

    # Pattern 3: Direct function calls
    if not tool_results:
        func_pattern = r'(write_file|read_file|edit_file|list_files)\s*\(\s*(.*?)\s*\)'
        for match in re.finditer(func_pattern, content, re.DOTALL):
            tool_name = match.group(1)
            args_str = match.group(2)

            args = {}
            arg_pattern = r'(\w+)\s*=\s*["\'](.+?)["\']'
            for arg_match in re.finditer(arg_pattern, args_str, re.DOTALL):
                args[arg_match.group(1)] = arg_match.group(2)

            if args:
                result = execute_tool(tool_name, args, project_root, gate, confirm, verbose, git_tools, auto_commit, task_description, preview, session_logger)
                tool_results.append({"tool": tool_name, "result": result})
                
                if result.startswith("BLOCKED:"):
                    blocked_actions.append({
                        "tool": tool_name,
                        "args": args,
                        "reason": result.replace("BLOCKED: ", "")
                    })
                
                _log_and_show_tool(session_logger, tool_name, args, result, verbose)
                break

    # Store blocked actions for feedback to model
    if blocked_actions:
        session_logger.log(
            "gate_blocked",
            f"{len(blocked_actions)} action(s) blocked by autopoietic gate",
            level="WARNING",
            blocked_actions=blocked_actions
        )

    return tool_results


def _log_and_show_tool(
    session_logger: SessionLogger,
    tool_name: str,
    args: dict,
    result: str,
    verbose: bool
) -> None:
    """Log tool call and show result if verbose."""
    # Log tool call
    session_logger.log_tool_call(
        tool_name=tool_name,
        arguments=args,
        result=result[:500],
        success="Success" in result and "BLOCKED" not in result
    )

    if verbose:
        status = "✅" if "Success" in result and "BLOCKED" not in result else "❌"
        click.echo(click.style(f"{status} {tool_name}: {result[:100]}",
                               fg="green" if "Success" in result and "BLOCKED" not in result else "red"))


def execute_tool(
    tool_name: str,
    args: dict,
    project_root: Path,
    gate: AutopoieticGate,
    confirm: bool = False,
    verbose: bool = False,
    git_tools: Optional[GitTools] = None,
    auto_commit: bool = False,
    task_description: str = "",
    preview: bool = False,
    session_logger: Optional[SessionLogger] = None
) -> str:
    """Execute a single tool call.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        project_root: Project root directory
        gate: Autopoietic gate instance
        confirm: Require confirmation before file modifications
        verbose: Show detailed output
        git_tools: Git tools instance
        auto_commit: Auto-commit changes
        task_description: Current task description
        preview: Show diff preview only, don't apply changes

    Returns:
        Result string
    """
    try:
        if tool_name == "write_file":
            path = project_root / args.get("path", "")
            content = args.get("content", "")

            # Gate check
            result = gate.check_action(path, "write", proposed_content=content)
            if not result.permitted:
                return f"BLOCKED: {result.reason}"

            # Preview mode - show diff without applying
            if preview:
                if path.exists():
                    existing = path.read_text(encoding="utf-8")
                    diff = difflib.unified_diff(
                        existing.splitlines(keepends=True),
                        content.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}"
                    )
                    diff_text = ''.join(diff)
                    return f"PREVIEW - Would modify {path}:\n\n{diff_text}"
                else:
                    return f"PREVIEW - Would create {path}:\n\n{content[:500]}..."

            # Confirmation mode - show diff and ask for approval
            if confirm:
                # Show what will be written
                click.echo()
                click.echo(click.style("⚠️  Confirmation Required", fg="yellow", bold=True))
                click.echo(f"File: {path}")

                # Show if file exists (edit) or new (create)
                if path.exists():
                    click.echo(click.style("Action: MODIFY existing file", fg="yellow"))
                    
                    # Show git diff if available
                    git_diff = git_tools.get_diff(path)
                    if git_diff:
                        click.echo(click.style("\nGit diff:", fg="cyan"))
                        click.echo(git_diff[:500] + ("..." if len(git_diff) > 500 else ""))
                    
                    # Show existing content
                    existing = path.read_text(encoding="utf-8")
                    click.echo(click.style("\nCurrent content:", fg="cyan"))
                    click.echo(existing[:500] + ("..." if len(existing) > 500 else ""))
                else:
                    click.echo(click.style("Action: CREATE new file", fg="green"))

                click.echo(click.style("\nNew content:", fg="cyan"))
                click.echo(content[:500] + ("..." if len(content) > 500 else ""))
                click.echo()

                # Ask for confirmation
                if not click.confirm("Proceed with this change?", default=False):
                    return "CANCELLED: User declined the change"

            # Write file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            # Write annotation if not present
            if not content.startswith("# ⊕"):
                reader = StigmaReader(path, project_root)
                session_id = getattr(session_logger, 'session_id', '') if session_logger else ''
                reader.update_annotation(press="write", age=0, session_id=session_id)

            result = f"Successfully wrote {len(content)} bytes to {path}"
            
            # Auto-commit if enabled and git is available
            if auto_commit and git_tools and git_tools.is_repository():
                try:
                    git_tools.stage_file(path)
                    # Create commit message with task and entropy info
                    commit_msg = f"myco: {task_description[:50]}\n\nFile: {path}\nPress: write"
                    git_tools._run_git("commit", "-m", commit_msg)
                    result += f"\n[Auto-committed: {commit_msg.split(chr(10))[0]}]"
                except Exception:
                    pass  # Silently fail if commit fails
            
            return result
            
        elif tool_name == "read_file":
            path = project_root / args.get("path", "")
            
            if not path.exists():
                return f"Error: File not found: {path}"
            
            content = path.read_text(encoding="utf-8")
            lines = args.get("lines")
            if lines:
                content_lines = content.splitlines()
                content = "\n".join(content_lines[:lines])
            
            return f"File: {path}\n\n{content}"
            
        elif tool_name == "edit_file":
            path = project_root / args.get("path", "")
            old_text = args.get("old_text", "")
            new_text = args.get("new_text", "")

            if not path.exists():
                return f"Error: File not found: {path}"

            content = path.read_text(encoding="utf-8")

            if old_text not in content:
                return "Error: Text to replace not found in file"

            # Gate check
            new_content = content.replace(old_text, new_text, 1)
            result = gate.check_action(path, "edit", proposed_content=new_content)
            
            # Step 4: Verbose gate output for visibility
            if verbose:
                if result.affects_self:
                    click.echo(click.style(f"  [Gate] {path} (SELF): H={result.entropy_before:.3f} → {result.entropy_after:.3f}", fg="cyan"))
                if not result.permitted:
                    click.echo(click.style(f"  [Gate] BLOCKED: {result.reason}", fg="yellow"))
            
            if not result.permitted:
                return f"BLOCKED: {result.reason}"

            # Preview mode - show diff without applying
            if preview:
                diff = difflib.unified_diff(
                    content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}"
                )
                diff_text = ''.join(diff)
                return f"PREVIEW - Would edit {path}:\n\n{diff_text}"

            # Confirmation mode - show diff and ask for approval
            if confirm:
                click.echo()
                click.echo(click.style("⚠️  Confirmation Required", fg="yellow", bold=True))
                click.echo(f"File: {path}")
                click.echo(click.style("Action: EDIT existing file", fg="yellow"))
                
                # Show what's changing
                click.echo(click.style("\nRemoving:", fg="red"))
                click.echo(old_text[:300] + ("..." if len(old_text) > 300 else ""))
                click.echo(click.style("\nAdding:", fg="green"))
                click.echo(new_text[:300] + ("..." if len(new_text) > 300 else ""))
                click.echo()
                
                if not click.confirm("Proceed with this edit?", default=False):
                    return "CANCELLED: User declined the change"

            # Write updated content
            path.write_text(new_content, encoding="utf-8")

            # Update annotation
            reader = StigmaReader(path, project_root)
            session_id = getattr(session_logger, 'session_id', '') if session_logger else ''
            reader.update_annotation(press="edit", age=0, session_id=session_id)

            result = f"Successfully edited {path}"
            
            # Auto-commit if enabled and git is available
            if auto_commit and git_tools and git_tools.is_repository():
                try:
                    git_tools.stage_file(path)
                    commit_msg = f"myco: {task_description[:50]}\n\nFile: {path}\nPress: edit"
                    git_tools._run_git("commit", "-m", commit_msg)
                    result += f"\n[Auto-committed: {commit_msg.split(chr(10))[0]}]"
                except Exception:
                    pass  # Silently fail if commit fails
            
            return result
            
        elif tool_name == "list_files":
            path = project_root / args.get("path", ".")
            pattern = args.get("pattern", "*")
            
            if not path.exists():
                return f"Error: Directory not found: {path}"
            
            files = list(path.glob(pattern))
            file_list = "\n".join(str(f.relative_to(path)) for f in sorted(files))
            
            return f"Directory: {path}\n\n{file_list or '(empty)'}"
            
        else:
            return f"Error: Unknown tool: {tool_name}"
            
    except Exception as e:
        return f"Error: {e}"


@cli.command()
@click.option("--path", "-p", type=str, default=".", help="Path to analyze")
def entropy(path: str):
    """Analyze the entropy of the codebase."""
    project_root = Path(path).resolve()
    report = analyze_entropy(project_root)

    click.echo(click.style("Entropy Analysis", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo(report.summary())


@cli.command()
@click.option("--path", "-p", type=str, default=".", help="Path to analyze")
@click.option("--threshold", "-t", type=float, default=0.3, help="Gradient threshold for fault lines")
def gradient(path: str, threshold: float):
    """Analyze gradient field and structural stress in the codebase.

    Identifies fault lines where modules with different entropy regimes connect.
    High gradient edges indicate structural stress concentrations.
    """
    from .entropy import compute_gradient_field

    project_root = Path(path).resolve()

    click.echo(click.style("Gradient Field Analysis", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))

    report = compute_gradient_field(project_root, threshold)

    click.echo(report.summary())

    if not report.fault_lines:
        click.echo()
        click.echo(click.style(
            f"No fault lines detected (threshold={threshold}). "
            "Modules have similar entropy levels across import edges.",
            fg="green"
        ))


@cli.command()
@click.option("--path", "-p", type=str, default=".", help="Path to analyze")
def tensegrity(path: str):
    """Analyze the tensegrity of the import graph."""
    project_root = Path(path).resolve()
    classifier = TensegrityClassifier(project_root)
    classifier.scan()

    click.echo(click.style("Tensegrity Analysis", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo(classifier.to_report())


@cli.command()
def world():
    """Show the current world model state."""
    project_root = Path.cwd()
    world = WorldModel.load(project_root)

    click.echo(click.style("World Model", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo(str(world))


@cli.group()
def models():
    """Manage model providers and list available models."""
    pass


@models.command("list")
@click.option("--url", "-u", type=str, default=None, help="Provider URL to query")
def models_list(url: Optional[str]):
    """List available models from provider."""
    base_url = url or "http://localhost:1234"
    
    click.echo(click.style(f"Querying models at {base_url}...", fg="cyan"))
    
    models = list_available_models(base_url)
    
    if models:
        click.echo()
        click.echo(click.style(f"Available models ({len(models)}):", fg="green", bold=True))
        for model in models:
            click.echo(f"  • {model}")
    else:
        click.echo(click.style("No models found or provider unavailable", fg="yellow"))


@models.command("add")
@click.argument("provider")
@click.option("--url", "-u", type=str, required=True, help="Provider base URL")
@click.option("--api-key", "-k", type=str, default=None, help="API key (optional)")
def models_add(provider: str, url: str, api_key: Optional[str]):
    """Add a custom model provider."""
    config = add_custom_provider(provider, url, api_key)
    
    click.echo(click.style(f"✓ Added provider: {provider}", fg="green", bold=True))
    click.echo(f"  URL: {config.base_url}")
    if api_key:
        click.echo("  API Key: ***configured***")


@models.command("remove")
@click.argument("provider")
def models_remove(provider: str):
    """Remove a custom model provider."""
    if remove_custom_provider(provider):
        click.echo(click.style(f"✓ Removed provider: {provider}", fg="green"))
    else:
        click.echo(click.style(f"Provider '{provider}' not found", fg="yellow"))


@cli.command()
def providers():
    """List configured model providers."""
    click.echo(click.style("Model Providers", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    
    all_providers = get_all_providers()
    custom_providers = load_custom_providers()
    
    click.echo(click.style("\nBuilt-in Providers:", fg="green", bold=True))
    for name, config in PROVIDERS.items():
        click.echo(f"  • {name}: {config.base_url}")
    
    if custom_providers:
        click.echo(click.style("\nCustom Providers:", fg="yellow", bold=True))
        for name, config in custom_providers.items():
            click.echo(f"  • {name}: {config.base_url}")
    else:
        click.echo("\n  No custom providers configured")
        click.echo("\n  Add one with: myco models add NAME -u URL")


@cli.command()
@click.option("--limit", "-n", type=int, default=10, help="Number of sessions to show")
def history(limit: int):
    """Show recent myco session history."""
    from .session_log import SessionLogger
    
    project_root = Path.cwd()
    logger = SessionLogger(project_root)
    
    entries = logger.read_log_file()
    
    if not entries:
        click.echo(click.style("No session history found.", fg="yellow"))
        return
    
    # Group by session
    sessions = {}
    for entry in entries:
        session_id = entry.data.get("session_id", "unknown")
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(entry)
    
    # Show recent sessions
    click.echo(click.style("Session History", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    
    session_list = list(sessions.items())[-limit:]
    
    for session_id, session_entries in session_list:
        # Find session start and end
        start_entry = next((e for e in session_entries if e.event_type == "session_start"), None)
        end_entry = next((e for e in session_entries if e.event_type == "session_end"), None)
        
        if start_entry:
            task = start_entry.data.get("task", "Unknown task")[:50]
            timestamp = start_entry.timestamp[:16].replace("T", " ")
            
            click.echo()
            click.echo(click.style(f"📅 {timestamp}", fg="bright_black"))
            click.echo(click.style(f"   Session: {session_id}", fg="bright_black"))
            click.echo(click.style(f"   Task: {task}", fg="white"))
            
            if end_entry:
                iterations = end_entry.data.get("iterations", 0)
                tokens = end_entry.data.get("tokens", 0)
                click.echo(click.style(f"   Iterations: {iterations}, Tokens: {tokens}", fg="bright_black"))
            
            # Count tool calls
            tool_calls = [e for e in session_entries if e.event_type == "tool_call"]
            if tool_calls:
                click.echo(click.style(f"   Tools used: {len(tool_calls)}", fg="bright_black"))
            
            # Check for blocked actions
            blocked = [e for e in session_entries if e.event_type == "gate_blocked"]
            if blocked:
                click.echo(click.style(f"   ⚠️  Blocked: {len(blocked)} action(s)", fg="yellow"))


@cli.command()
@click.option("--limit", "-n", type=int, default=None, help="Number of sessions to analyze")
@click.option("--compare-last", "-c", type=int, default=None, help="Compare last N sessions (trend analysis)")
@click.option("--gate-readiness", "-g", is_flag=True, help="Check readiness for gate threshold relaxation (Step 4)")
def validate(limit: Optional[int], compare_last: Optional[int], gate_readiness: bool):
    """Validate past myco interventions against entropy changes.

    Analyzes session history to determine whether past interventions
    correlated with entropy improvement. This is read-only analysis.

    With --compare-last, shows trend analysis across sessions:
    - Entropy delta trend
    - Blocked rate trend
    - Efficiency ratio (entropy improvement per joule)

    With --gate-readiness, assesses readiness for Step 4 (gate threshold
    relaxation). Requires 10+ sessions of data.

    Output shows:
    - Each intervention with H_before, H_after, and delta
    - Summary by intervention type
    - Overall improvement rate
    - Session trends (with --compare-last)
    - Gate readiness assessment (with --gate-readiness)
    """
    from .validate import compare_last_n_sessions, format_session_comparison, assess_gate_readiness, format_gate_readiness

    project_root = Path.cwd()

    # If --gate-readiness specified, show readiness assessment
    if gate_readiness:
        click.echo(click.style("Assessing gate threshold relaxation readiness...", fg="cyan"))

        metrics = compare_last_n_sessions(project_root, n=20)  # Get up to 20 sessions
        assessment = assess_gate_readiness(metrics, min_sessions=10)
        readiness_text = format_gate_readiness(assessment)
        click.echo(readiness_text)
        return

    # If --compare-last specified, show trend analysis
    if compare_last:
        click.echo(click.style(f"Analyzing last {compare_last} sessions...", fg="cyan"))

        metrics = compare_last_n_sessions(project_root, n=compare_last)
        comparison = format_session_comparison(metrics)
        click.echo(comparison)
        return

    # Default: show intervention validation
    click.echo(click.style("Validating interventions...", fg="cyan"))

    records = validate_interventions(project_root, limit=limit)

    table = format_validation_table(records)
    click.echo(table)


@cli.command()
def report():
    """Generate a substrate health report in Markdown format.

    This is a read-only analysis command that aggregates all entropy signals
    into a developer-friendly report. Use this to understand myco's analysis
    before running interventions.

    Example:
        myco report > substrate-health.md
        myco report --output substrate-health.md
    """
    from datetime import datetime
    from .entropy import ImportGraphBuilder, EntropyCalculator

    project_root = Path.cwd()

    # Gather all signals
    entropy_report = analyze_entropy(project_root)
    substrate_health = calculate_substrate_health(project_root)
    gradient_field = compute_gradient_field(project_root)
    world = WorldModel.load(project_root)

    # Get priority files with dual regime
    priority_files = get_priority_files(project_root, top_n=10)

    # Build regime analysis for priority files
    builder = ImportGraphBuilder(project_root)
    builder.scan()
    calculator = EntropyCalculator(builder)

    regime_details = []
    for pf in priority_files:
        file_path = project_root / pf["file"]
        if file_path.exists():
            module_name = builder._path_to_module_name(file_path)
            H_structural = calculator.calculate_module_entropy(module_name)
            internal = compute_internal_entropy(file_path)
            H_internal = internal["H_internal"]
            dual = classify_dual_regime(H_structural, H_internal)

            # Get trajectory if available
            trajectory_data = None
            try:
                trajectory_data = compute_entropy_trajectory(file_path, n_commits=10)
            except Exception:
                pass  # Not a git repo or no history

            regime_details.append({
                "priority_file": pf,
                "H_structural": H_structural,
                "H_internal": H_internal,
                "dual_regime": dual,
                "trajectory": trajectory_data
            })

    # Get intervention validation if available
    intervention_records = validate_interventions(project_root, limit=20)

    # Generate markdown report
    report_lines = generate_substrate_report(
        project_root=project_root,
        entropy_report=entropy_report,
        substrate_health=substrate_health,
        gradient_field=gradient_field,
        world=world,
        regime_details=regime_details,
        intervention_records=intervention_records
    )

    report_text = "\n".join(report_lines)

    # Output
    click.echo(report_text)


def generate_substrate_report(
    project_root: Path,
    entropy_report,
    substrate_health: dict,
    gradient_field,
    world,
    regime_details: list[dict],
    intervention_records
) -> list[str]:
    """Generate markdown substrate health report.

    Args:
        project_root: Project root
        entropy_report: EntropyReport from analyze_entropy
        substrate_health: Dict from calculate_substrate_health
        gradient_field: GradientFieldReport from compute_gradient_field
        world: WorldModel instance
        regime_details: List of regime analysis dicts
        intervention_records: List of InterventionRecord from validate

    Returns:
        List of markdown lines
    """
    lines = []

    # Header
    lines.append("# Substrate Health Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Project:** {project_root.name}")
    lines.append(f"**Sessions analyzed:** {world.session_count}")
    lines.append("")

    # Overall Health
    lines.append("---")
    lines.append("")
    lines.append("## Overall Health")
    lines.append("")
    health_score = substrate_health.get("health_score", 0)
    health_status = substrate_health.get("status", "unknown")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Health Score | {health_score:.2f}/1.00 ({health_status}) |")
    lines.append(f"| Entropy Baseline | {entropy_report.global_entropy:.3f} |")
    lines.append(f"| Entropy Trend | {world.entropy_trend:+.3f}/session |")
    lines.append(f"| Modules Analyzed | {entropy_report.module_count} |")
    lines.append(f"| Crystallized | {len(entropy_report.crystallized)} |")
    lines.append(f"| Dissipative | {len(entropy_report.dissipative)} |")
    lines.append(f"| Diffuse | {len(entropy_report.diffuse)} |")
    lines.append(f"| Tensegrity Violations | {world.tensegrity_violations} |")
    lines.append(f"| Fault Lines | {gradient_field.fault_line_count} |")
    lines.append(f"| Mean Gradient | {gradient_field.mean_gradient:.3f} |")
    lines.append("")

    # Health status message
    lines.append(f"**Status:** {substrate_health.get('status_message', 'Unknown')}")
    lines.append("")

    # Modules Requiring Attention
    lines.append("---")
    lines.append("")
    lines.append("## Modules Requiring Attention")
    lines.append("")

    if regime_details:
        for i, rd in enumerate(regime_details[:5], 1):
            pf = rd["priority_file"]
            dual = rd["dual_regime"]
            trajectory = rd["trajectory"]

            file_name = pf["file"]
            combined_regime = dual["combined_regime"]
            H_struct = rd["H_structural"]
            H_int = rd["H_internal"]
            guidance = dual["guidance"]

            lines.append(f"### {i}. {file_name}")
            lines.append("")
            lines.append(f"- **Regime:** {combined_regime}")
            lines.append(f"- **H_structural:** {H_struct:.3f} ({dual['structural_regime']})")
            lines.append(f"- **H_internal:** {H_int:.3f} ({dual['internal_regime']})")
            lines.append(f"- **Priority:** {dual['priority']}")

            # Add trajectory if available
            if trajectory and trajectory.points:
                vel = trajectory.velocity_structural
                trend = "crystallizing" if vel < -0.02 else "diffusing" if vel > 0.02 else "stable"
                lines.append(f"- **Trajectory:** {trend} (velocity: {vel:+.4f}/session)")

            lines.append(f"- **Recommendation:** {guidance}")
            lines.append("")
    else:
        lines.append("*No modules requiring immediate attention.*")
        lines.append("")

    # Gradient Field Summary
    lines.append("---")
    lines.append("")
    lines.append("## Structural Stress (Gradient Field)")
    lines.append("")

    if gradient_field.fault_lines:
        lines.append(f"**Fault lines detected:** {gradient_field.fault_line_count}")
        lines.append("")
        lines.append("| Edge | Gradient | Status |")
        lines.append("|------|----------|--------|")
        for fl in gradient_field.fault_lines[:10]:
            status = "⚠️ High" if fl.gradient > 0.5 else "⚡ Moderate"
            lines.append(
                f"| {fl.importer} → {fl.imported} | "
                f"{fl.gradient:.3f} | {status} |"
            )
        lines.append("")

        # Highest stress modules
        stressed = sorted(gradient_field.module_stress,
                         key=lambda m: m.mean_gradient, reverse=True)[:5]
        if any(m.fault_line_count > 0 for m in stressed):
            lines.append("**Highest stress modules:**")
            lines.append("")
            for m in stressed:
                if m.fault_line_count > 0:
                    lines.append(
                        f"- {m.module}: stress={m.mean_gradient:.3f} "
                        f"({m.fault_line_count} fault lines)"
                    )
            lines.append("")
    else:
        lines.append("*No fault lines detected. Module entropy is balanced across import edges.*")
        lines.append("")

    # Intervention History
    lines.append("---")
    lines.append("")
    lines.append("## Intervention History")
    lines.append("")

    if intervention_records:
        # Group by type
        by_type = {}
        for rec in intervention_records:
            t = rec.intervention_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(rec)

        lines.append(f"**Total interventions analyzed:** {len(intervention_records)}")
        lines.append("")

        # Summary table
        lines.append("| Intervention Type | Count | Avg ΔH | Improvements |")
        lines.append("|-------------------|-------|--------|--------------|")
        for intervention_type, records in sorted(by_type.items()):
            count = len(records)
            avg_delta = sum(r.H_after - r.H_before for r in records) / count
            improvements = sum(1 for r in records if r.H_after < r.H_before)
            lines.append(
                f"| {intervention_type} | {count} | {avg_delta:+.3f} | "
                f"{improvements}/{count} |"
            )
        lines.append("")

        # Most effective
        best_type = min(by_type.keys(),
                       key=lambda t: sum(r.H_after - r.H_before for r in by_type[t]) / len(by_type[t]))
        best_records = by_type[best_type]
        best_delta = sum(r.H_after - r.H_before for r in best_records) / len(best_records)
        lines.append(f"**Most effective:** {best_type} (avg ΔH = {best_delta:+.3f})")
        lines.append("")
    else:
        lines.append("*No intervention history available. Run myco interventions to build validation data.*")
        lines.append("")

    # Trajectory Warnings
    lines.append("---")
    lines.append("")
    lines.append("## Trajectory Warnings")
    lines.append("")

    trajectory_warnings = []
    for rd in regime_details:
        trajectory = rd["trajectory"]
        if trajectory and trajectory.points:
            vel = trajectory.velocity_structural
            if vel < -0.02:  # Crystallizing
                trajectory_warnings.append(
                    f"- **{rd['priority_file']['file']}**: Crystallizing "
                    f"(velocity: {vel:+.4f}/session)"
                )
            elif vel > 0.02:  # Diffusing
                trajectory_warnings.append(
                    f"- **{rd['priority_file']['file']}**: Diffusing "
                    f"(velocity: {vel:+.4f}/session)"
                )

    if trajectory_warnings:
        for warning in trajectory_warnings[:5]:
            lines.append(warning)
        lines.append("")
    else:
        lines.append("*No trajectory warnings. Modules are stable or improving.*")
        lines.append("")

    # Open Intentions
    lines.append("---")
    lines.append("")
    lines.append("## Open Intentions")
    lines.append("")

    if world.open_intentions:
        for intention in world.open_intentions[-10:]:
            lines.append(f"- {intention}")
        lines.append("")
    else:
        lines.append("*No open intentions. Substrate is in equilibrium.*")
        lines.append("")

    # Recommendations
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    # Generate recommendations based on analysis
    recommendations = []

    if len(entropy_report.crystallized) > 0:
        recommendations.append(
            f"1. **Decompose crystallized modules** — "
            f"{len(entropy_report.crystallized)} modules are rigid and over-coupled. "
            f"Apply `decompose` or `interface_inversion`."
        )

    if gradient_field.fault_line_count > 0:
        recommendations.append(
            f"2. **Address fault lines** — {gradient_field.fault_line_count} structural stress points detected. "
            f"Consider refactoring to balance entropy across import edges."
        )

    if world.entropy_trend > 0.02:
        recommendations.append(
            f"3. **Reverse entropy trend** — Entropy increasing by {world.entropy_trend:.3f}/session. "
            f"Focus on restructuring over feature addition."
        )

    if not recommendations:
        recommendations.append(
            "*Substrate is healthy. Continue monitoring and make minimal changes.*"
        )

    for rec in recommendations:
        lines.append(rec)
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by myco — A CLI that writes itself, reads the soil, and never forgets where it grew.*")

    return lines


@cli.command()
def migrate():
    """Migrate source file annotations to sidecar file.

    Scans all Python files for source annotations and migrates them
    to .myco/annotations.json. Source annotations are kept for
    backward compatibility.
    """
    project_root = Path.cwd()

    click.echo(click.style("Migrating annotations...", fg="cyan"))

    migrated = migrate_source_annotations(project_root)

    if migrated > 0:
        click.echo(click.style(f"✓ Migrated {migrated} annotations to .myco/annotations.json", fg="green"))
    else:
        click.echo(click.style("No annotations to migrate", fg="yellow"))


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
def internal(filepath: str):
    """Show internal entropy metrics for a file.
    
    Analyzes function size distribution, nesting depth, and name cohesion.
    Shows combined H_internal score and dual regime classification.
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    
    click.echo(click.style(f"Internal Entropy: {filepath}", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    
    # Compute internal entropy
    metrics = compute_internal_entropy(filepath)
    
    click.echo(f"\nFunction size entropy:  {metrics['H_function_size']:.3f}")
    click.echo(f"Nesting depth entropy:  {metrics['H_nesting']:.3f}")
    click.echo(f"Name cohesion:          {metrics['cohesion']:.3f}")
    click.echo(f"\nH_internal:             {metrics['H_internal']:.3f}")
    
    # Get structural entropy
    from .entropy import ImportGraphBuilder, EntropyCalculator
    builder = ImportGraphBuilder(filepath.parent)
    builder.scan()
    calculator = EntropyCalculator(builder)
    module_name = builder._path_to_module_name(filepath)
    H_structural = calculator.calculate_module_entropy(module_name)
    
    click.echo(f"H_structural:           {H_structural:.3f}")
    
    # Dual regime classification
    regime = classify_dual_regime(H_structural, metrics["H_internal"])
    
    click.echo()
    click.echo(click.style("Dual Regime Classification:", fg="yellow", bold=True))
    click.echo(f"  Combined:    {regime['combined_regime']}")
    click.echo(f"  Structural:  {regime['structural_regime']}")
    click.echo(f"  Internal:    {regime['internal_regime']}")
    click.echo(f"  Priority:    {regime['priority']}")
    click.echo(f"\n  Guidance: {regime['guidance']}")


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--commits", "-n", type=int, default=20, help="Number of commits to analyze")
def trajectory(filepath: str, commits: int):
    """Show entropy trajectory for a file from git history.
    
    Analyzes entropy changes over the last N commits.
    Shows velocity (rate of change) and acceleration.
    """
    from pathlib import Path
    
    filepath = Path(filepath)
    
    click.echo(click.style(f"Entropy Trajectory: {filepath}", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    
    trajectory_data = compute_entropy_trajectory(filepath, n_commits=commits)
    
    if not trajectory_data:
        click.echo(click.style("No trajectory data available.", fg="yellow"))
        click.echo("  File must be in a git repository with commit history.")
        return
    
    click.echo(f"\nCommits analyzed: {len(trajectory_data.points)}")
    click.echo(f"File: {trajectory_data.file_path}")
    
    # Show recent points
    click.echo(click.style("\nRecent entropy history:", fg="yellow", bold=True))
    for point in trajectory_data.points[-5:]:
        click.echo(f"  {point.commit_hash[:7]} ({point.timestamp[:10]}): "
                  f"H_struct={point.H_structural:.3f}, H_int={point.H_internal:.3f}")
    
    # Show velocity and acceleration
    click.echo()
    click.echo(click.style("Trajectory metrics:", fg="yellow", bold=True))
    click.echo(f"  Velocity (structural):  {trajectory_data.velocity_structural:+.4f}/commit")
    click.echo(f"  Velocity (internal):    {trajectory_data.velocity_internal:+.4f}/commit")
    click.echo(f"  Acceleration (struct):  {trajectory_data.acceleration_structural:+.5f}/commit²")
    click.echo(f"  Acceleration (internal):{trajectory_data.acceleration_internal:+.5f}/commit²")
    
    # Interpret trajectory
    interpretation = interpret_trajectory(trajectory_data)
    
    click.echo()
    click.echo(click.style("Interpretation:", fg="yellow", bold=True))
    click.echo(f"  Status:     {interpretation['status']}")
    click.echo(f"  Acceleration: {interpretation['acceleration']}")
    click.echo(f"  Urgency:    {interpretation['urgency']}")
    click.echo(f"  Priority:   {interpretation['priority']}")
    click.echo(f"\n  Guidance: {interpretation['guidance']}")


if __name__ == "__main__":
    cli()
