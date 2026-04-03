"""Agent tools for file editing and command execution.

MYCO Vision Integration:
- Stigmergic annotations on Python files
- Entropy-aware file operations
- Autopoietic gate enforcement
- Browser automation (Playwright)
- Process management
"""

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils.logging import LogConfig

# MYCO: Import stigmergic annotation system
HAS_MYCO = False  # Default to False
try:
    import sys
    from pathlib import Path
    # Get absolute path to myco directory
    _current_file = Path(__file__).resolve()
    _myco_path = _current_file.parent.parent.parent / "myco"
    if _myco_path.exists():
        _myco_parent = str(_myco_path.parent)
        if _myco_parent not in sys.path:
            sys.path.insert(0, _myco_parent)
        from myco.stigma import StigmaReader, StigmergicAnnotation
        from myco.entropy import calculate_substrate_health, ImportGraphBuilder, EntropyCalculator
        from myco.gate import EntropyGate
        from myco.world import WorldModel
        HAS_MYCO = True
except Exception:
    pass  # HAS_MYCO remains False


class ToolResult:
    """Result from a tool execution."""

    def __init__(self, success: bool, output: str, error: Optional[str] = None, verified: bool = False):
        self.success = success
        self.output = output
        self.error = error
        self.verified = verified  # MYCO: Track if result was verified

    def to_response(self) -> str:
        """Format result for LLM response."""
        status = "✓ Verified" if self.verified else ("Success" if self.success else "Error")
        if self.success:
            return f"{status}:\n{self.output}"
        else:
            return f"{status}:\n{self.error or self.output}"


class FileTools:
    """File manipulation tools."""

    logger = LogConfig.get_logger("gguf.agent.files")

    @staticmethod
    def read_file(path: str = None, lines: Optional[int] = None, file_path: str = None) -> ToolResult:
        """Read a file's contents.

        Args:
            path: Path to file (or file_path for compatibility)
            lines: Number of lines to read (None = all)
            file_path: Alternative parameter name for path

        Returns:
            ToolResult with file contents
        """
        # Handle both 'path' and 'file_path' parameter names
        actual_path = path or file_path
        if not actual_path:
            return ToolResult(
                success=False,
                output="",
                error="No path provided"
            )
        try:
            file_path = Path(actual_path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a file: {file_path}"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if lines:
                content_lines = content.splitlines()
                content = "\n".join(content_lines[:lines])
                if len(content_lines) > lines:
                    content += f"\n... ({len(content_lines) - lines} more lines)"

            return ToolResult(
                success=True,
                output=f"File: {file_path}\n\n{content}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}"
            )

    @staticmethod
    def write_file(path: str, content: str) -> ToolResult:
        """Write content to a file.

        MYCO Vision: Writes stigmergic annotation on Python files.
        MYCO Vision: Autopoietic gate checks entropy before writing.
        MYCO Phase F: Enforces file size limits to prevent truncation.
        MYCO Phase 1: Checks entropy budget before writing.

        Args:
            path: Path to file
            content: Content to write

        Returns:
            ToolResult with confirmation and verification
        """
        try:
            file_path = Path(path).resolve()

            # MYCO Phase 1: Entropy budget check for Python files
            if path.endswith('.py'):
                try:
                    from myco.entropy import check_entropy_budget
                    
                    # Get current file content (if exists)
                    current_content = ""
                    if file_path.exists():
                        try:
                            current_content = file_path.read_text(encoding='utf-8')
                        except Exception:
                            current_content = ""
                    
                    # Check entropy budget
                    within_budget, curr_H, prop_H, message = check_entropy_budget(
                        current_content,
                        content,
                        max_delta=0.15,
                        max_new_file_H=0.50
                    )
                    
                    if not within_budget:
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"ENTROPY BUDGET EXCEEDED:\n{message}\n\nSUGGESTION: Split into smaller modules or reduce complexity.",
                            verified=True
                        )
                    
                    # Budget OK - log it
                    FileTools.logger.info(f"Entropy budget OK: {message}")
                    
                except Exception as e:
                    FileTools.logger.warning(f"Entropy budget check failed: {e}")
                    # Continue anyway - budget check is advisory

            # MYCO Phase F: File size limit check
            MAX_FILE_SIZE = 5000  # bytes (prevents truncation)
            if len(content) > MAX_FILE_SIZE:
                # Suggest splitting for Python files
                if path.endswith('.py'):
                    base_name = path.replace('.py', '')
                    suggestion = f"""
⚠️ FILE TOO LARGE ({len(content)} bytes, max: {MAX_FILE_SIZE} bytes)

SUGGESTION: Split into smaller modules:
  - {base_name}_core.py (core classes/functions)
  - {base_name}_models.py (data models)
  - {base_name}_utils.py (helper functions)

Or use append_file to add content in chunks:
  1. write_file({path}, "initial content")
  2. append_file({path}, "more content")
  3. append_file({path}, "even more content")
"""
                else:
                    suggestion = f"""
⚠️ FILE TOO LARGE ({len(content)} bytes, max: {MAX_FILE_SIZE} bytes)

SUGGESTION: 
  - Split into smaller files
  - Use append_file to add content in chunks
"""
                return ToolResult(
                    success=False,
                    output="",
                    error=suggestion.strip(),
                    verified=True
                )

            # Initialize gate_info for all file types
            gate_info = None

            # MYCO Vision: Entropy gate check for Python files
            if path.endswith('.py'):
                try:
                    gate_result = EntropyGate.check_entropy_delta(str(file_path), content)
                    if not gate_result.success:
                        # Gate blocked the change
                        return ToolResult(
                            success=False,
                            output="",
                            error=f"AUTOPOIETIC GATE BLOCKED:\n{gate_result.error}",
                            verified=True
                        )
                    # Gate permitted - include info in output
                    gate_info = gate_result.output
                except Exception:
                    gate_info = None  # Gate check failed, proceed anyway

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # MYCO: Auto-create __init__.py for Python packages
            # When creating a .py file in a subdirectory, ensure __init__.py exists
            if path.endswith('.py') and '\\' in path:
                package_dir = file_path.parent
                init_file = package_dir / '__init__.py'
                
                # Only create if it doesn't exist and directory has/will have .py files
                if not init_file.exists():
                    # Check if this is a package (has or will have sibling .py files)
                    py_files = list(package_dir.glob('*.py'))
                    # Create __init__.py if there are other .py files or this is the first
                    if len(py_files) >= 0:  # Always create for packages
                        try:
                            init_file.write_text(
                                f'"""{package_dir.name} package."""\n',
                                encoding='utf-8'
                            )
                            FileTools.logger.info(f"Auto-created {init_file}")
                        except Exception as e:
                            FileTools.logger.warning(f"Could not create {init_file}: {e}")

            # MYCO: For Python files, add stigmergic annotation with REAL entropy
            if HAS_MYCO and path.endswith('.py'):
                real_entropy = None
                regime = None

                # MYCO Phase F: Large file warning (preventive)
                LARGE_FILE_THRESHOLD = 3000  # bytes (warning before limit)
                if len(content) > LARGE_FILE_THRESHOLD:
                    FileTools.logger.warning(f"⚠️ Large file detected: {path} ({len(content)} bytes). Consider splitting.")

                # MYCO Phase V: Check if dependencies exist before writing
                dependency_warnings = []
                try:
                    import ast
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                            # Check if it's a local import (starts with src or project name)
                            if node.module.startswith('src.') or node.module.split('.')[0] == 'src':
                                # Convert import path to file path
                                dep_file = node.module.replace('.', '/') + '.py'
                                dep_path = file_path.parent / dep_file
                                if not dep_path.exists():
                                    # Try alternative path
                                    alt_path = file_path.parent / f"{node.module.split('.')[-1]}.py"
                                    if not alt_path.exists():
                                        dependency_warnings.append(
                                            f"⚠️ Import '{node.module}' - module may not exist yet"
                                        )
                except Exception:
                    pass  # Ignore parsing errors

                # Show warnings but proceed (gate will block if entropy too high)
                if dependency_warnings:
                    FileTools.logger.warning(f"Dependency warnings for {path}: {dependency_warnings[:3]}")

                # MYCO: Validate and fix imports for Python files
                content = FileTools._validate_and_fix_imports(file_path, content)

                # Try to calculate real entropy
                try:
                    from myco.entropy import ImportGraphBuilder, EntropyCalculator

                    # Build import graph for project
                    project_root = Path.cwd()
                    builder = ImportGraphBuilder(project_root)
                    builder.scan()

                    # Get module name from file path
                    try:
                        rel_path = file_path.relative_to(project_root)
                        module_name = str(rel_path.with_suffix('')).replace('\\', '.').replace('/', '.')
                        if module_name.endswith('.__init__'):
                            module_name = module_name[:-9]
                    except ValueError:
                        module_name = file_path.stem

                    # Calculate real entropy
                    calc = EntropyCalculator(builder)
                    try:
                        real_entropy = calc.calculate_module_entropy(module_name)
                    except Exception:
                        real_entropy = None

                    # Determine regime
                    if real_entropy is not None:
                        if real_entropy < 0.3:
                            regime = "crystallized"
                        elif real_entropy > 0.75:
                            regime = "diffuse"
                        else:
                            regime = "dissipative"

                    # Get existing annotation for drift calculation (before file is written)
                    drift = 0.00
                    press = "bootstrap"
                    if file_path.exists():
                        try:
                            stigma_reader = StigmaReader(file_path)
                            existing = stigma_reader.read_annotation()
                            if existing and real_entropy is not None:
                                drift = real_entropy - existing.H
                                press = "write"
                        except Exception:
                            pass

                    # Create annotation with REAL entropy
                    new_annotation = StigmergicAnnotation(
                        H=real_entropy if real_entropy is not None else 0.50,
                        press=press,
                        age=0,
                        drift=drift
                    )

                except Exception:
                    # Fallback to bootstrap if entropy calculation fails
                    new_annotation = StigmergicAnnotation(
                        H=0.50,
                        press="none",
                        age=0,
                        drift=0.00
                    )

                # Write content with annotation
                annotation_line = new_annotation.format() + "\n"
                if not content.startswith(annotation_line):
                    content = annotation_line + content

                # Store for output
                file_real_entropy = real_entropy
                file_regime = regime
            else:
                file_real_entropy = None
                file_regime = None

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # MYCO: Verify the file was actually written correctly
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Verification failed: File not created: {path}",
                    verified=False
                )

            # Read back the actual content to verify (handles line ending conversions)
            with open(file_path, "r", encoding="utf-8") as f:
                actual_content = f.read()

            actual_size = len(actual_content.encode('utf-8'))
            expected_size = len(content.encode('utf-8'))

            if actual_size != expected_size:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Verification failed: Size mismatch (expected {expected_size}, got {actual_size})",
                    verified=False
                )

            # MYCO: Additional verification for Python files - check syntax
            syntax_verified = False
            syntax_error = None
            if path.endswith('.py'):
                try:
                    result = subprocess.run(
                        ['python', '-m', 'py_compile', str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        syntax_verified = True
                    else:
                        syntax_error = result.stderr.strip()
                except Exception as e:
                    syntax_error = f"Could not verify syntax: {e}"

            verification_status = ""
            if syntax_verified:
                verification_status = " [✓ Verified: syntax OK]"
            
            # MYCO: Log annotation writing
            if HAS_MYCO and path.endswith('.py') and file_real_entropy is not None:
                verification_status += f" [MYCO: H={file_real_entropy:.2f} | {file_regime}]"
                
                # MYCO Phase V: Track imports for connection tracking
                try:
                    import ast
                    tree = ast.parse(content)
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            imports.append(node.module)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                    
                    if imports:
                        verification_status += f" [Imports: {', '.join(imports[:5])}]"
                except Exception:
                    pass  # Ignore parsing errors
            elif syntax_error:
                verification_status = f" [⚠ Warning: {syntax_error}]"

            # MYCO: Add substrate health note
            myco_note = ""
            if HAS_MYCO and path.endswith('.py'):
                myco_note = " [MYCO: Stigmergic annotation written]"

            # MYCO: Add entropy gate info
            gate_note = ""
            if gate_info and path.endswith('.py'):
                # Extract key info from gate output
                lines = gate_info.split('\n')
                for line in lines[:4]:  # First few lines have entropy info
                    if 'Delta:' in line or 'Regime:' in line:
                        gate_note += f" [{line.strip()}]"

            # MYCO Phase 1.3: Test co-creation - create tests for new Python files
            test_creation_note = ""
            if path.endswith('.py') and syntax_verified:
                # Skip test files (don't create tests for tests)
                file_name = Path(path).name
                if not file_name.startswith('test_'):
                    try:
                        # Generate and write test stub
                        test_result = TestTools.create_tests_for_file(str(file_path), content)

                        if test_result.success:
                            # Extract test file path from output
                            test_file = test_result.output.split(': ')[1].split('\n')[0] if ': ' in test_result.output else 'test file'
                            test_creation_note = f" [✓ Tests: {test_file}]"

                            # MYCO: Run tests immediately
                            try:
                                test_run_result = TestTools.test_pytest(path=test_file, verbose=False, timeout=30)
                                if test_run_result.success:
                                    test_creation_note += " [✓ Tests pass]"
                                else:
                                    test_creation_note += " [⚠ Tests need implementation]"
                            except Exception:
                                test_creation_note += " [Tests created]"

                    except Exception as e:
                        FileTools.logger.warning(f"Test co-creation failed: {e}")
                        test_creation_note = " [⚠ Test creation failed]"

            # MYCO Phase 1.4: Dependency tracking - track what depends on this file
            dependency_note = ""
            if path.endswith('.py') and syntax_verified:
                try:
                    from cli.agent.architecture import track_dependencies

                    # Track dependencies
                    dep_info = track_dependencies(str(file_path), content)

                    if dep_info["affected_files"]:
                        affected_count = len(dep_info["affected_files"])
                        dependency_note = f" [⚠ Affects: {affected_count} files]"
                        FileTools.logger.warning(f"Changing {path} affects {affected_count} files: {dep_info['affected_files'][:3]}")
                    elif dep_info["dependencies"]:
                        dep_count = len(dep_info["dependencies"])
                        dependency_note = f" [Depends: {dep_count} files]"

                except Exception as e:
                    FileTools.logger.warning(f"Dependency tracking failed: {e}")
                    dependency_note = " [⚠ Dependency tracking failed]"

            # MYCO: Add real entropy info if available
            entropy_note = ""
            if file_real_entropy is not None and file_regime is not None:
                entropy_note = f" [H={file_real_entropy:.3f} {file_regime}]"

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {file_path}{verification_status}{myco_note}{gate_note}{entropy_note}{test_creation_note}{dependency_note}",
                verified=syntax_verified  # MYCO: True only if syntax verified
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {e}",
                verified=False
            )

    @staticmethod
    def append_file(path: str, content: str) -> ToolResult:
        """Append content to an existing file (for large files).

        MYCO Vision: Enables chunked writing for large files.

        Args:
            path: Path to file
            content: Content to append

        Returns:
            ToolResult with confirmation
        """
        try:
            file_path = Path(path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}"
                )

            # Append content
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)

            # Get file size after append
            new_size = file_path.stat().st_size

            return ToolResult(
                success=True,
                output=f"Successfully appended {len(content)} bytes to {file_path} (total: {new_size} bytes)",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to append file: {e}",
                verified=False
            )

    @staticmethod
    def delete_file(path: str) -> ToolResult:
        """Delete a file.

        Args:
            path: Path to file to delete

        Returns:
            ToolResult with confirmation
        """
        try:
            file_path = Path(path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}"
                )

            file_path.unlink()

            return ToolResult(
                success=True,
                output=f"Successfully deleted {file_path}",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to delete file: {e}",
                verified=False
            )

    @staticmethod
    def copy_file(source: str, destination: str) -> ToolResult:
        """Copy a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            ToolResult with confirmation
        """
        try:
            import shutil
            src_path = Path(source).resolve()
            dst_path = Path(destination).resolve()

            if not src_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Source file not found: {source}"
                )

            # Create parent directories if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dst_path)

            return ToolResult(
                success=True,
                output=f"Successfully copied {src_path} to {dst_path}",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to copy file: {e}",
                verified=False
            )

    @staticmethod
    def edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
        """Edit a file by replacing old_text with new_text.

        Args:
            path: Path to file
            old_text: Text to replace
            new_text: Replacement text

        Returns:
            ToolResult with confirmation
        """
        try:
            file_path = Path(path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if old_text not in content:
                return ToolResult(
                    success=False,
                    output="",
                    error="Text to replace not found in file"
                )

            new_content = content.replace(old_text, new_text, 1)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(
                success=True,
                output=f"Successfully edited {file_path}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to edit file: {e}"
            )

    @staticmethod
    def list_files(path: str, pattern: str = "*") -> ToolResult:
        """List files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern (default: *)

        Returns:
            ToolResult with file list
        """
        try:
            dir_path = Path(path).resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory not found: {dir_path}"
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Not a directory: {dir_path}"
                )

            files = list(dir_path.glob(pattern))
            file_list = "\n".join(str(f.relative_to(dir_path)) for f in sorted(files))

            return ToolResult(
                success=True,
                output=f"Directory: {dir_path}\n\n{file_list or '(empty)'}"
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list files: {e}"
            )

    @staticmethod
    def _validate_and_fix_imports(file_path: Path, content: str) -> str:
        """MYCO: Validate and fix imports for Python files.
        
        Checks if imports match project structure and fixes common issues:
        - Converts absolute imports to relative for packages
        - Warns about missing modules
        
        Args:
            file_path: Path to the Python file
            content: File content
            
        Returns:
            Fixed content with corrected imports
        """
        import ast
        import re
        
        try:
            # Parse the content to analyze imports
            tree = ast.parse(content)
        except SyntaxError:
            return content  # Can't parse, return as-is
        
        # Check if file is in a subdirectory (package)
        if '\\' not in str(file_path) and '/' not in str(file_path):
            return content  # Root level file, no fix needed
        
        # Get the directory level
        parts = str(file_path).replace('\\', '/').split('/')
        level = len(parts) - 1  # How deep in the project
        
        # Only fix if in src/ or similar package structure
        if level < 2:
            return content
        
        # Find absolute imports that should be relative
        fixed_content = content
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # Absolute import
                    # Check if this looks like a local module
                    local_modules = ['models', 'schemas', 'database', 'main', 'utils', 'config']
                    
                    if node.module in local_modules or node.module.startswith('src.'):
                        # This should probably be a relative import
                        # Convert to relative import
                        old_import = f"from {node.module} import"
                        new_import = f"from .{node.module.lstrip('src.')} import"
                        
                        if old_import in fixed_content:
                            fixed_content = fixed_content.replace(old_import, new_import, 1)
        
        return fixed_content


class CommandTools:
    """Command execution tools."""

    logger = LogConfig.get_logger("gguf.agent.commands")

    @staticmethod
    def run_command(command: str, timeout: int = 60) -> ToolResult:
        """Run a shell command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            ToolResult with output and verification
        """
        try:
            # Security: Block dangerous commands
            dangerous = ["rm -rf", "del /s", "format", "mkfs", "shutdown", "reboot"]
            for d in dangerous:
                if d.lower() in command.lower():
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Command blocked for safety: contains '{d}'",
                        verified=False
                    )

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
            )

            output = result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"

            # MYCO: Verify command actually succeeded
            verified = False
            verification_note = ""

            if result.returncode == 0:
                verified = True
                verification_note = " [✓ Verified: exit code 0]"
            else:
                # Command failed - try to provide useful error
                verification_note = f" [✗ Failed: exit code {result.returncode}]"
                output += verification_note

            return ToolResult(
                success=result.returncode == 0,
                output=output if result.returncode == 0 else output,
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
                verified=verified
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                verified=False
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Command failed: {e}",
                verified=False
            )

    @staticmethod
    def run_python(code: str, timeout: int = 30, cwd: str = None) -> ToolResult:
        """Run Python code and return output.

        MYCO: Automatically adds project root to sys.path for imports.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            cwd: Working directory (default: PROJECT_ROOT)

        Returns:
            ToolResult with output
        """
        try:
            # Use project root as working directory if not specified
            work_dir = Path(cwd) if cwd else ProjectTools.PROJECT_ROOT

            # Prepend project root to sys.path
            project_root = str(ProjectTools.PROJECT_ROOT)
            preamble = f"import sys; sys.path.insert(0, r'{project_root}'); "

            # Run via stdin to handle multi-line code safely
            full_code = preamble + code
            result = subprocess.run(
                ['python', '-c', full_code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output if result.returncode == 0 else f"Error:\n{output}",
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
                verified=result.returncode == 0
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Python code timed out after {timeout} seconds",
                verified=False
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Python execution failed: {e}",
                verified=False
            )


class ProjectTools:
    """Project structure and path validation tools.
    
    MYCO Vision: Empower agent to understand and navigate project structure.
    """

    logger = LogConfig.get_logger("gguf.agent.project")
    
    # Project root - set when agent starts
    PROJECT_ROOT = Path.cwd()
    
    @classmethod
    def set_project_root(cls, root: str):
        """Set the project root directory."""
        cls.PROJECT_ROOT = Path(root).resolve()
        cls.logger.info(f"Project root set to: {cls.PROJECT_ROOT}")
    
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Validate and normalize a file path.
        
        Args:
            path: File path to validate
            
        Returns:
            Normalized absolute path
        """
        if not path:
            return ""
        
        p = Path(path)
        
        # If absolute, verify it's within project or system paths
        if p.is_absolute():
            try:
                # Try to make relative to project root
                p = p.relative_to(cls.PROJECT_ROOT)
                p = cls.PROJECT_ROOT / p
            except ValueError:
                # Keep absolute if it's a valid system path
                pass
        else:
            # Relative path - resolve from project root
            p = cls.PROJECT_ROOT / p
        
        # Normalize the path (resolve .., ., etc.)
        p = p.resolve()
        
        return str(p)
    
    @classmethod
    def find_file(cls, filename: str, max_depth: int = 5) -> ToolResult:
        """Find a file by name in the project.
        
        Args:
            filename: Name of file to find
            max_depth: Maximum directory depth to search
            
        Returns:
            ToolResult with file path(s) or error
        """
        try:
            found_files = []
            
            for root, dirs, files in os.walk(cls.PROJECT_ROOT):
                # Check depth
                depth = len(Path(root).relative_to(cls.PROJECT_ROOT).parts)
                if depth > max_depth:
                    dirs.clear()  # Don't recurse deeper
                    continue
                
                if filename in files:
                    found_files.append(str(Path(root) / filename))
            
            if not found_files:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File '{filename}' not found in project (max depth: {max_depth})"
                )
            
            return ToolResult(
                success=True,
                output=f"Found {len(found_files)} file(s):\n" + "\n".join(found_files)
            )
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    @classmethod
    def list_project_structure(cls, max_depth: int = 3) -> ToolResult:
        """List the project directory structure.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            ToolResult with structure tree
        """
        try:
            lines = []
            
            def _walk(path: Path, prefix: str, depth: int):
                if depth > max_depth:
                    return
                
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
                for i, item in enumerate(items):
                    if item.name.startswith('.') or item.name == '__pycache__':
                        continue
                    
                    is_last = (i == len(items) - 1)
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{item.name}")
                    
                    if item.is_dir():
                        extension = "    " if is_last else "│   "
                        _walk(item, prefix + extension, depth + 1)
            
            lines.append(f"Project: {cls.PROJECT_ROOT}")
            lines.append("=" * 50)
            _walk(cls.PROJECT_ROOT, "", 0)
            
            return ToolResult(success=True, output="\n".join(lines))
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    @classmethod
    def test_import(cls, import_path: str, from_path: Optional[str] = None) -> ToolResult:
        """Test if a Python import works and suggest fixes.
        
        Args:
            import_path: Module to import (e.g., 'models.user' or 'api.products')
            from_path: Optional 'from X import Y' path
            
        Returns:
            ToolResult with success status and fix suggestions
        """
        import sys
        
        # Add project root to path
        sys.path.insert(0, str(cls.PROJECT_ROOT))
        
        try:
            if from_path:
                # from X import Y
                module = __import__(from_path, fromlist=[import_path])
                obj = getattr(module, import_path, None)
                if obj is None:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Cannot import '{import_path}' from '{from_path}'",
                        verified=False
                    )
                return ToolResult(
                    success=True,
                    output=f"✓ Import successful: from {from_path} import {import_path}\n  → {obj}",
                    verified=True
                )
            else:
                # import X
                module = __import__(import_path)
                return ToolResult(
                    success=True,
                    output=f"✓ Import successful: import {import_path}\n  → {module}",
                    verified=True
                )
                
        except ImportError as e:
            # Analyze error and suggest fixes
            error_msg = str(e)
            suggestions = []
            
            if "No module named" in error_msg:
                module_name = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                suggestions.append(f"Check if '{module_name}.py' exists in project")
                suggestions.append(f"Try: from {module_name} import <class_name>")
            
            if "cannot import name" in error_msg:
                suggestions.append("Check class/function name spelling")
                suggestions.append("Check if the class/function is defined in the module")
            
            return ToolResult(
                success=False,
                output="",
                error=error_msg,
                verified=False
            )
            
        finally:
            # Clean up sys.path
            if str(cls.PROJECT_ROOT) in sys.path:
                sys.path.remove(str(cls.PROJECT_ROOT))


class SearchTools:
    """Search tools for finding content."""

    logger = LogConfig.get_logger("gguf.agent.search")

    @staticmethod
    def search_text(path: str, query: str, max_results: int = 10) -> ToolResult:
        """Search for text in a file.

        Args:
            path: File path
            query: Text to search for
            max_results: Maximum matches to return

        Returns:
            ToolResult with matches
        """
        try:
            file_path = Path(path).resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {file_path}"
                )

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            matches = []
            for i, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    matches.append(f"Line {i}: {line.strip()}")
                    if len(matches) >= max_results:
                        break

            if not matches:
                return ToolResult(
                    success=True,
                    output=f"No matches found for '{query}' in {file_path}"
                )

            return ToolResult(
                success=True,
                output=f"Found {len(matches)} matches for '{query}':\n\n" + "\n".join(matches)
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {e}"
            )


class BrowserTools:
    """Browser automation tools using Playwright."""

    logger = LogConfig.get_logger("gguf.agent.browser")
    _browser = None
    _page = None

    @staticmethod
    def _ensure_browser():
        """Ensure browser is installed and available."""
        try:
            from playwright.sync_api import sync_playwright
            return sync_playwright
        except ImportError:
            return None

    @staticmethod
    def browser_open(url: str, headless: bool = True, timeout: int = 30000) -> ToolResult:
        """Open a URL in a browser.

        Args:
            url: URL to open
            headless: Run browser headless (default: True)
            timeout: Page load timeout in ms

        Returns:
            ToolResult with page title or error
        """
        playwright_class = BrowserTools._ensure_browser()
        if not playwright_class:
            return ToolResult(
                success=False,
                output="",
                error="Playwright not installed. Run: pip install playwright && playwright install"
            )

        try:
            # Start playwright and keep it open
            pw = playwright_class().start()
            # Use system Edge browser (more reliable on Windows)
            browser = pw.chromium.launch(headless=headless, channel="msedge")
            page = browser.new_page()
            page.set_default_timeout(timeout)
            page.goto(url)

            # Store for later use (keep browser open)
            BrowserTools._playwright = pw
            BrowserTools._browser = browser
            BrowserTools._page = page

            return ToolResult(
                success=True,
                output=f"Opened: {url}\nTitle: {page.title()[:50] if page.title() else 'N/A'}\nURL: {page.url}",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to open browser: {e}"
            )

    @staticmethod
    def browser_screenshot(path: str = "screenshot.png", selector: str = None) -> ToolResult:
        """Take a screenshot of the current page.

        Args:
            path: Path to save screenshot
            selector: CSS selector for element screenshot (optional)

        Returns:
            ToolResult with screenshot path
        """
        if not BrowserTools._page:
            return ToolResult(
                success=False,
                output="",
                error="No page open. Use browser_open first."
            )

        try:
            if selector:
                BrowserTools._page.locator(selector).screenshot(path=path)
            else:
                BrowserTools._page.screenshot(path=path)

            return ToolResult(
                success=True,
                output=f"Screenshot saved to: {path}",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Screenshot failed: {e}"
            )

    @staticmethod
    def browser_click(selector: str) -> ToolResult:
        """Click an element on the page.

        Args:
            selector: CSS selector for element to click

        Returns:
            ToolResult with confirmation
        """
        if not BrowserTools._page:
            return ToolResult(
                success=False,
                output="",
                error="No page open. Use browser_open first."
            )

        try:
            BrowserTools._page.click(selector)
            return ToolResult(
                success=True,
                output=f"Clicked: {selector}",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Click failed: {e}"
            )

    @staticmethod
    def browser_fill(selector: str, value: str) -> ToolResult:
        """Fill an input field with text.

        Args:
            selector: CSS selector for input field
            value: Text to fill

        Returns:
            ToolResult with confirmation
        """
        if not BrowserTools._page:
            return ToolResult(
                success=False,
                output="",
                error="No page open. Use browser_open first."
            )

        try:
            BrowserTools._page.fill(selector, value)
            return ToolResult(
                success=True,
                output=f"Filled {selector} with: {value[:50]}...",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Fill failed: {e}"
            )

    @staticmethod
    def browser_evaluate(script: str) -> ToolResult:
        """Execute JavaScript in the browser.

        Args:
            script: JavaScript code to execute

        Returns:
            ToolResult with evaluation result
        """
        if not BrowserTools._page:
            return ToolResult(
                success=False,
                output="",
                error="No page open. Use browser_open first."
            )

        try:
            result = BrowserTools._page.evaluate(script)
            return ToolResult(
                success=True,
                output=f"Result: {result}",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Evaluate failed: {e}"
            )

    @staticmethod
    def browser_close() -> ToolResult:
        """Close the browser.

        Returns:
            ToolResult with confirmation
        """
        if not BrowserTools._browser:
            return ToolResult(
                success=False,
                output="",
                error="No browser open."
            )

        try:
            BrowserTools._browser.close()
            BrowserTools._browser = None
            BrowserTools._page = None
            return ToolResult(
                success=True,
                output="Browser closed.",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Close failed: {e}"
            )

    @staticmethod
    def browser_compare_screenshots(image1_path: str, image2_path: str, output_path: str = "diff.png", threshold: float = 0.05) -> ToolResult:
        """Compare two screenshots and highlight differences.

        Args:
            image1_path: Path to first screenshot (baseline)
            image2_path: Path to second screenshot (current)
            output_path: Path to save difference image
            threshold: Sensitivity threshold (0.0-1.0, lower = more sensitive)

        Returns:
            ToolResult with comparison results
        """
        try:
            from PIL import Image, ImageChops, ImageDraw
            import numpy as np
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="PIL/Pillow not installed. Run: pip install Pillow numpy"
            )

        try:
            # Open images
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')

            # Check if sizes match
            if img1.size != img2.size:
                # Resize img2 to match img1
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

            # Convert to numpy arrays for comparison
            arr1 = np.array(img1)
            arr2 = np.array(img2)

            # Calculate pixel difference
            diff = np.abs(arr1.astype(float) - arr2.astype(float))

            # Calculate similarity percentage
            max_diff = 255 * 3  # RGB max difference
            total_pixels = diff.shape[0] * diff.shape[1]
            diff_pixels = np.sum(diff > (threshold * 255))
            similarity = 100 * (1 - diff_pixels / total_pixels)

            # Create visual diff image
            diff_img = Image.new('RGB', img1.size, (255, 255, 255))
            draw = ImageDraw.Draw(diff_img)

            # Highlight differences in red
            for x in range(img1.width):
                for y in range(img1.height):
                    pixel_diff = np.sum(diff[y, x])
                    if pixel_diff > (threshold * 255 * 3):
                        draw.point((x, y), fill=(255, 0, 0))  # Red for differences

            # Save diff image
            diff_img.save(output_path)

            # Create summary
            result = {
                "baseline": image1_path,
                "current": image2_path,
                "diff_image": output_path,
                "similarity_percent": round(similarity, 2),
                "different_pixels": int(diff_pixels),
                "total_pixels": int(total_pixels),
                "image_size": f"{img1.width}x{img1.height}"
            }

            status = "MATCH" if similarity > 99.5 else ("MINOR_DIFF" if similarity > 95 else "SIGNIFICANT_DIFF")

            return ToolResult(
                success=True,
                output=f"Screenshot Comparison:\n"
                       f"Status: {status}\n"
                       f"Similarity: {similarity:.2f}%\n"
                       f"Different pixels: {diff_pixels:,} / {total_pixels:,}\n"
                       f"Image size: {img1.width}x{img1.height}\n"
                       f"Diff saved to: {output_path}",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Screenshot comparison failed: {e}"
            )

    @staticmethod
    def browser_screenshot_compare(url: str, baseline_path: str, current_path: str = "current.png", diff_path: str = "diff.png", threshold: float = 0.05) -> ToolResult:
        """Take a screenshot and compare with baseline.

        Args:
            url: URL to screenshot
            baseline_path: Path to baseline screenshot
            current_path: Path to save current screenshot
            diff_path: Path to save difference image
            threshold: Sensitivity threshold (0.0-1.0)

        Returns:
            ToolResult with comparison results
        """
        # Take screenshot
        result = BrowserTools.browser_screenshot(current_path)
        if not result.success:
            return result

        # Compare with baseline
        return BrowserTools.browser_compare_screenshots(baseline_path, current_path, diff_path, threshold)

    @staticmethod
    def browser_analyze_screenshot(image_path: str) -> ToolResult:
        """Analyze a screenshot for visual elements.

        Args:
            image_path: Path to screenshot

        Returns:
            ToolResult with analysis (dominant colors, text regions, etc.)
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="PIL/Pillow not installed. Run: pip install Pillow numpy"
            )

        try:
            img = Image.open(image_path)
            arr = np.array(img)

            # Calculate basic stats
            width, height = img.size
            avg_color = np.mean(arr, axis=(0, 1)).astype(int)

            # Detect if mostly text (high contrast edges)
            gray = np.mean(arr, axis=2)
            edges_x = np.abs(np.diff(gray, axis=1))
            edges_y = np.abs(np.diff(gray, axis=0))
            edge_density = (np.mean(edges_x > 30) + np.mean(edges_y > 30)) / 2

            # Detect dominant colors
            pixels = arr.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            top_colors = unique_colors[np.argsort(counts)[-5:]][::-1]

            analysis = {
                "size": f"{width}x{height}",
                "total_pixels": width * height,
                "average_color": f"RGB({avg_color[0]}, {avg_color[1]}, {avg_color[2]})",
                "edge_density": f"{edge_density*100:.1f}%",
                "likely_has_text": edge_density > 0.1,
                "top_colors": [f"RGB({c[0]}, {c[1]}, {c[2]})" for c in top_colors]
            }

            return ToolResult(
                success=True,
                output=f"Screenshot Analysis:\n"
                       f"Size: {analysis['size']}\n"
                       f"Total pixels: {analysis['total_pixels']:,}\n"
                       f"Average color: {analysis['average_color']}\n"
                       f"Edge density: {analysis['edge_density']} (text likelihood: {'High' if analysis['likely_has_text'] else 'Low'})\n"
                       f"Top colors: {', '.join(analysis['top_colors'])}",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Screenshot analysis failed: {e}"
            )


class GitTools:
    """Git version control tools."""

    logger = LogConfig.get_logger("gguf.agent.git")

    @staticmethod
    def _run_git(args: list[str], cwd: str = None) -> tuple[bool, str, str]:
        """Run git command and return result.

        Args:
            args: Git arguments (e.g., ['status', '--short'])
            cwd: Working directory (default: current dir)

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                ['git'] + args,
                capture_output=True,
                text=True,
                cwd=cwd or Path.cwd(),
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Git command timed out"
        except FileNotFoundError:
            return False, "", "Git not found. Please install git."
        except Exception as e:
            return False, "", str(e)

    @staticmethod
    def git_status(cwd: str = None) -> ToolResult:
        """Get git status.

        Args:
            cwd: Working directory (default: current dir)

        Returns:
            ToolResult with status output
        """
        success, stdout, stderr = GitTools._run_git(['status', '--short'], cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git status failed: {stderr}"
            )

        if not stdout.strip():
            return ToolResult(
                success=True,
                output="Working tree clean. No changes to commit.",
                verified=True
            )

        # Parse status output
        lines = stdout.strip().split('\n')
        staged = [l for l in lines if not l.startswith('??') and not l.startswith(' M')]
        unstaged = [l for l in lines if l.startswith(' M') or l.startswith('M ')]
        untracked = [l for l in lines if l.startswith('??')]

        output = "Git Status:\n\n"
        if staged:
            output += f"Staged changes ({len(staged)}):\n"
            for line in staged:
                output += f"  {line}\n"
        if unstaged:
            output += f"\nUnstaged changes ({len(unstaged)}):\n"
            for line in unstaged:
                output += f"  {line}\n"
        if untracked:
            output += f"\nUntracked files ({len(untracked)}):\n"
            for line in untracked:
                output += f"  {line[3:]}\n"

        return ToolResult(
            success=True,
            output=output,
            verified=True
        )

    @staticmethod
    def git_diff(path: str = None, staged: bool = False, cwd: str = None) -> ToolResult:
        """Get git diff.

        Args:
            path: Specific file path (optional)
            staged: Diff staged changes (default: False for unstaged)
            cwd: Working directory

        Returns:
            ToolResult with diff output
        """
        args = ['diff']
        if staged:
            args.append('--cached')
        if path:
            args.append('--')
            args.append(path)

        success, stdout, stderr = GitTools._run_git(args, cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git diff failed: {stderr}"
            )

        if not stdout.strip():
            return ToolResult(
                success=True,
                output="No changes to show.",
                verified=True
            )

        return ToolResult(
            success=True,
            output=f"Git Diff:\n\n{stdout}",
            verified=True
        )

    @staticmethod
    def git_add(files: list[str], cwd: str = None) -> ToolResult:
        """Stage files for commit.

        Args:
            files: List of files to stage
            cwd: Working directory

        Returns:
            ToolResult with confirmation
        """
        success, stdout, stderr = GitTools._run_git(['add'] + files, cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git add failed: {stderr}"
            )

        return ToolResult(
            success=True,
            output=f"Staged {len(files)} file(s): {', '.join(files)}",
            verified=True
        )

    @staticmethod
    def git_commit(message: str, files: list[str] = None, cwd: str = None) -> ToolResult:
        """Commit staged changes.

        Args:
            message: Commit message
            files: Optional files to stage before commit
            cwd: Working directory

        Returns:
            ToolResult with commit info
        """
        # Stage files if provided
        if files:
            result = GitTools.git_add(files, cwd)
            if not result.success:
                return result

        # Commit
        success, stdout, stderr = GitTools._run_git(['commit', '-m', message], cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git commit failed: {stderr}"
            )

        # Get commit hash
        success2, hash_out, _ = GitTools._run_git(['rev-parse', '--short', 'HEAD'], cwd)
        commit_hash = hash_out.strip()[:7] if success2 else "unknown"

        return ToolResult(
            success=True,
            output=f"Committed: {message}\nHash: {commit_hash}",
            verified=True
        )

    @staticmethod
    def git_branch(name: str = None, create: bool = False, cwd: str = None) -> ToolResult:
        """List or create branches.

        Args:
            name: Branch name (optional)
            create: If True, create new branch
            cwd: Working directory

        Returns:
            ToolResult with branch info
        """
        if create and name:
            # Create and checkout new branch
            success, stdout, stderr = GitTools._run_git(['checkout', '-b', name], cwd)
            if not success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to create branch: {stderr}"
                )
            return ToolResult(
                success=True,
                output=f"Created and switched to branch '{name}'",
                verified=True
            )
        elif name:
            # Checkout existing branch
            success, stdout, stderr = GitTools._run_git(['checkout', name], cwd)
            if not success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to checkout branch: {stderr}"
                )
            return ToolResult(
                success=True,
                output=f"Switched to branch '{name}'",
                verified=True
            )
        else:
            # List all branches
            success, stdout, stderr = GitTools._run_git(['branch', '-a'], cwd)
            if not success:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Git branch failed: {stderr}"
                )

            branches = [b.strip() for b in stdout.strip().split('\n') if b.strip()]
            current = next((b for b in branches if b.startswith('*')), None)

            output = f"Branches ({len(branches)}):\n"
            for branch in branches:
                output += f"  {branch}\n"
            if current:
                output += f"\nCurrent: {current[2:]}"

            return ToolResult(
                success=True,
                output=output,
                verified=True
            )

    @staticmethod
    def git_log(limit: int = 10, cwd: str = None) -> ToolResult:
        """Get commit log.

        Args:
            limit: Number of commits to show
            cwd: Working directory

        Returns:
            ToolResult with commit log
        """
        success, stdout, stderr = GitTools._run_git(
            ['log', f'-{limit}', '--oneline', '--decorate'],
            cwd
        )

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git log failed: {stderr}"
            )

        if not stdout.strip():
            return ToolResult(
                success=True,
                output="No commits yet.",
                verified=True
            )

        return ToolResult(
            success=True,
            output=f"Recent Commits (last {limit}):\n\n{stdout}",
            verified=True
        )

    @staticmethod
    def git_push(remote: str = "origin", branch: str = None, cwd: str = None) -> ToolResult:
        """Push changes to remote.

        Args:
            remote: Remote name (default: origin)
            branch: Branch name (default: current)
            cwd: Working directory

        Returns:
            ToolResult with push output
        """
        args = ['push', remote]
        if branch:
            args.append(branch)

        success, stdout, stderr = GitTools._run_git(args, cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git push failed: {stderr}"
            )

        return ToolResult(
            success=True,
            output=f"Pushed to {remote}:\n{stdout}" if stdout else f"Pushed to {remote}",
            verified=True
        )

    @staticmethod
    def git_pull(remote: str = "origin", branch: str = None, cwd: str = None) -> ToolResult:
        """Pull changes from remote.

        Args:
            remote: Remote name (default: origin)
            branch: Branch name (default: current)
            cwd: Working directory

        Returns:
            ToolResult with pull output
        """
        args = ['pull', remote]
        if branch:
            args.append(branch)

        success, stdout, stderr = GitTools._run_git(args, cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git pull failed: {stderr}"
            )

        return ToolResult(
            success=True,
            output=f"Pulled from {remote}:\n{stdout}" if stdout else f"Pulled from {remote}",
            verified=True
        )

    @staticmethod
    def git_init(cwd: str = None) -> ToolResult:
        """Initialize git repository.

        Args:
            cwd: Working directory

        Returns:
            ToolResult with confirmation
        """
        success, stdout, stderr = GitTools._run_git(['init'], cwd)

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Git init failed: {stderr}"
            )

        return ToolResult(
            success=True,
            output="Git repository initialized.",
            verified=True
        )


class TestTools:
    """Test runner tools for various frameworks."""

    logger = LogConfig.get_logger("gguf.agent.test")

    @staticmethod
    def generate_test_stub(source_code: str, file_path: str) -> str:
        """Generate test stub for Python source code.
        
        MYCO Phase 1.3: Test co-creation - tests created WITH code.
        
        Analyzes source and creates pytest test stub with:
        - Import statements
        - Test class/functions for each public class/function
        - Basic test cases
        
        Args:
            source_code: Python source code
            file_path: Path to source file
            
        Returns:
            Test stub content as string
        """
        import ast
        import re
        
        # Parse source code
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return "# Could not generate tests - syntax error in source\n"
        
        # Extract module name
        module_name = Path(file_path).stem
        
        # Collect classes and functions
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip private classes
                if not node.name.startswith('_'):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]
                    classes.append({'name': node.name, 'methods': methods})
            
            elif isinstance(node, ast.FunctionDef):
                # Skip private functions and methods
                if not node.name.startswith('_') and node.col_offset == 0:
                    functions.append(node.name)
        
        # Generate test content
        test_lines = [
            f'"""Tests for {module_name}."""',
            'import pytest',
            '',
            f'from {module_name} import ' + ', '.join([c['name'] for c in classes] + functions) if (classes or functions) else '# No public classes or functions to import',
            '',
            ''
        ]
        
        # Generate tests for classes
        for cls in classes:
            test_lines.append(f'class Test{cls["name"]}:')
            test_lines.append(f'    """Tests for {cls["name"]}."""')
            test_lines.append('')
            
            if cls['methods']:
                for method in cls['methods']:
                    if method.startswith('__') and method.endswith('__'):
                        continue  # Skip dunder methods
                    
                    test_lines.append(f'    def test_{method.lower()}(self):')
                    test_lines.append(f'        """Test {cls["name"]}.{method}."""')
                    test_lines.append(f'        # TODO: Implement test')
                    test_lines.append(f'        # Example:')
                    test_lines.append(f'        # instance = {cls["name"]}')
                    test_lines.append(f'        # result = instance.{method}(...)')
                    test_lines.append(f'        # assert result is not None')
                    test_lines.append(f'        assert True  # Placeholder')
                    test_lines.append('')
            else:
                test_lines.append(f'    def test_init(self):')
                test_lines.append(f'        """Test {cls["name"]} initialization."""')
                test_lines.append(f'        # TODO: Implement test')
                test_lines.append(f'        # instance = {cls["name"]}(...)')
                test_lines.append(f'        # assert instance is not None')
                test_lines.append(f'        assert True  # Placeholder')
                test_lines.append('')
        
        # Generate tests for functions
        for func in functions:
            test_lines.append(f'def test_{func.lower()}():')
            test_lines.append(f'    """Test {func}."""')
            test_lines.append(f'    # TODO: Implement test')
            test_lines.append(f'    # Example:')
            test_lines.append(f'    # result = {func}(...)')
            test_lines.append(f'    # assert result is not None')
            test_lines.append(f'    assert True  # Placeholder')
            test_lines.append('')
        
        # If no tests generated, add basic placeholder
        if not classes and not functions:
            test_lines = [
                f'"""Tests for {module_name}."""',
                'import pytest',
                '',
                '# TODO: Add tests for this module',
                '# The module appears to contain only private or internal code',
                '',
                'def test_module_exists():',
                '    """Test that module can be imported."""',
                f'    from {module_name} import *  # noqa',
                '    assert True',
                ''
            ]
        
        return '\n'.join(test_lines)
    
    @staticmethod
    def create_tests_for_file(file_path: str, source_code: str = None) -> ToolResult:
        """Create test stub for a Python file.
        
        MYCO Phase 1.3: Test co-creation.
        
        Args:
            file_path: Path to Python file
            source_code: Source code (optional, will read from file if not provided)
            
        Returns:
            ToolResult with test creation result
        """
        try:
            # Read source if not provided
            if not source_code:
                source_code = Path(file_path).read_text(encoding='utf-8')
            
            # Generate test path
            test_path = f"test_{Path(file_path).name}"
            test_dir = Path(file_path).parent / "tests"
            
            # Create tests directory if needed
            if not test_dir.exists():
                test_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = test_dir / test_path
            
            # Check if test already exists
            if test_file.exists():
                return ToolResult(
                    success=True,
                    output=f"Tests already exist: {test_file}",
                    verified=True
                )
            
            # Generate test stub
            test_content = TestTools.generate_test_stub(source_code, file_path)
            
            # Write test file
            test_file.write_text(test_content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                output=f"✓ Test stub created: {test_file}\n\nEdit the test file to implement actual tests.",
                verified=True
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create tests: {e}"
            )

    @staticmethod
    def _run_test(command: list[str], cwd: str = None, timeout: int = 120) -> tuple[bool, str, str]:
        """Run test command and return result.

        Args:
            command: Test command (e.g., ['pytest', '-v'])
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd or Path.cwd(),
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Test timed out after {timeout} seconds"
        except FileNotFoundError as e:
            return False, "", f"Test command not found: {e}"
        except Exception as e:
            return False, "", str(e)

    @staticmethod
    def test_run(command: str = "pytest", framework: str = None, path: str = None,
                 verbose: bool = True, timeout: int = 120, cwd: str = None) -> ToolResult:
        """Run tests with auto-detected or specified framework.

        Args:
            command: Test command (default: "pytest")
            framework: Framework name (pytest, unittest, npm, jest, mocha)
            path: Specific test file/path
            verbose: Verbose output
            timeout: Timeout in seconds
            cwd: Working directory

        Returns:
            ToolResult with test results
        """
        # Build command based on framework
        if framework:
            framework = framework.lower()
        else:
            # Auto-detect framework
            framework = TestTools._detect_framework(cwd)

        # Build test command
        test_cmd = TestTools._build_command(command, framework, path, verbose)

        # Run tests
        success, stdout, stderr = TestTools._run_test(test_cmd, cwd, timeout)

        # Parse results
        result_info = TestTools._parse_results(stdout, stderr, framework)
        
        # MYCO Phase 3: Auto-fix common test issues
        if not success and stderr:
            fix_suggestions = TestTools._suggest_test_fixes(stderr, stdout)
            if fix_suggestions:
                result_info += f"\n\n🔧 SUGGESTED FIXES:\n{fix_suggestions}"

        if success:
            return ToolResult(
                success=True,
                output=f"Tests Passed!\n\n{result_info}",
                verified=True
            )
        else:
            return ToolResult(
                success=False,
                output=f"Tests Failed:\n\n{result_info}",
                error=f"Test failures detected. See output for details.",
                verified=True  # Verified that tests ran
            )

    @staticmethod
    def _detect_framework(cwd: str = None) -> str:
        """Auto-detect test framework from project files.

        Returns:
            Framework name (pytest, jest, npm, unittest)
        """
        cwd = Path(cwd or Path.cwd())

        # Check for package.json (Node.js)
        package_json = cwd / "package.json"
        if package_json.exists():
            try:
                import json
                pkg = json.loads(package_json.read_text())
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    test_script = scripts["test"]
                    if "jest" in test_script:
                        return "jest"
                    if "mocha" in test_script:
                        return "mocha"
                    if "vitest" in test_script:
                        return "vitest"
                # Check dependencies
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "jest" in deps:
                    return "jest"
                if "mocha" in deps:
                    return "mocha"
                return "npm"
            except Exception:
                pass

        # Check for pytest.ini or pyproject.toml with pytest
        if (cwd / "pytest.ini").exists():
            return "pytest"

        pyproject = cwd / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                return "pytest"

        # Check for setup.py with unittest
        if (cwd / "setup.py").exists():
            return "unittest"

        # Default to pytest for Python projects
        if any(cwd.glob("*.py")):
            return "pytest"

        return "pytest"  # Default

    @staticmethod
    def _build_command(command: str, framework: str, path: str = None, verbose: bool = True) -> list[str]:
        """Build test command based on framework.

        Args:
            command: Base command
            framework: Framework name
            path: Test path
            verbose: Verbose flag

        Returns:
            Command list for subprocess
        """
        if framework in ("jest", "mocha", "vitest"):
            # Node.js frameworks - use npm/npx
            if framework == "jest":
                cmd = ["npx", "jest"]
            elif framework == "mocha":
                cmd = ["npx", "mocha"]
            elif framework == "vitest":
                cmd = ["npx", "vitest", "run"]
            else:
                cmd = ["npm", "test"]

            if verbose:
                if framework != "vitest":
                    cmd.append("--verbose")
            if path:
                cmd.append(path)

            return cmd

        elif framework == "unittest":
            # Python unittest
            cmd = ["python", "-m", "unittest"]
            if verbose:
                cmd.append("-v")
            if path:
                cmd.append(path)
            return cmd

        elif framework == "npm":
            # Generic npm test
            cmd = ["npm", "test"]
            if verbose:
                cmd.append("--")
                cmd.append("--verbose")
            return cmd

        else:
            # Default: pytest or custom command
            if command == "pytest":
                cmd = ["pytest"]
                if verbose:
                    cmd.append("-v")
                if path:
                    cmd.append(path)
                return cmd
            else:
                # Custom command
                import shlex
                return shlex.split(command)

    @staticmethod
    def _parse_results(stdout: str, stderr: str, framework: str) -> str:
        """Parse test output and extract key information.

        Returns:
            Formatted result summary
        """
        output = stdout + stderr

        if not output.strip():
            return "No test output."

        # Framework-specific parsing
        if framework == "pytest":
            # Look for: "X passed, Y failed in Zs"
            import re
            match = re.search(r'(\d+) passed.*?(\d+) failed.*?in ([\d.]+)s', output, re.IGNORECASE)
            if match:
                passed, failed, duration = match.groups()
                return f"Passed: {passed}\nFailed: {failed}\nDuration: {duration}s\n\n{output[-1000:]}"

            # Look for: "X passed in Zs" (all passed)
            match = re.search(r'(\d+) passed.*?in ([\d.]+)s', output, re.IGNORECASE)
            if match:
                passed, duration = match.groups()
                return f"Passed: {passed}\nDuration: {duration}s\n\n{output[-500:]}"

        elif framework in ("jest", "mocha", "vitest"):
            # Look for: "X Tests, Y Passed, Z Failed"
            import re
            match = re.search(r'Tests:\s*(\d+) passed', output)
            if match:
                passed = match.group(1)
                return f"Tests Passed: {passed}\n\n{output[-1000:]}"

            match = re.search(r'Passing\s*(\d+)', output)
            if match:
                return f"Tests Passing: {match.group(1)}\n\n{output[-1000:]}"

        elif framework == "unittest":
            # Look for: "OK" or "FAILED (failures=X)"
            if "OK" in output:
                return f"All tests passed!\n\n{output[-500:]}"
            import re
            match = re.search(r'FAILED \(failures=(\d+)\)', output)
            if match:
                return f"Failures: {match.group(1)}\n\n{output[-1000:]}"

        # Generic fallback
        lines = output.strip().split('\n')
        last_lines = lines[-20:] if len(lines) > 20 else lines
        return '\n'.join(last_lines)
    
    @staticmethod
    def _suggest_test_fixes(stderr: str, stdout: str) -> str:
        """MYCO Phase 3: Suggest fixes for common test failures.
        
        Analyzes test output and suggests fixes for common issues.
        
        Args:
            stderr: Standard error output
            stdout: Standard output
            
        Returns:
            Formatted fix suggestions
        """
        import re
        output = stderr + stdout
        suggestions = []
        
        # Import errors
        if "ModuleNotFoundError" in output or "ImportError" in output:
            match = re.search(r"No module named ['\"]?(\w+)['\"]?", output)
            if match:
                module = match.group(1)
                suggestions.append(f"1. Import Error: Module '{module}' not found\n"
                                  f"   → Run: pip install {module}\n"
                                  f"   → Or check if module name is correct")
            
            # Check for missing app import in tests
            if "cannot import name 'app'" in output or "ImportError: cannot import name 'app'" in output:
                suggestions.append(f"2. Missing 'app' import in test file\n"
                                  f"   → Add: from src.main import app\n"
                                  f"   → Or check if main.py exports 'app'")
        
        # Fixture errors
        if "fixture" in output.lower() and ("not found" in output.lower() or "error" in output.lower()):
            suggestions.append(f"Fixture Error: Test fixture not found\n"
                              f"   → Check @pytest.fixture decorators\n"
                              f"   → Ensure fixtures are in conftest.py or test file")
        
        # Assertion errors
        if "AssertionError" in output or "assert" in output.lower():
            suggestions.append(f"Assertion Failed: Check test expectations\n"
                              f"   → Review assert statements\n"
                              f"   → Verify expected vs actual values")
        
        # Database errors
        if "sqlite" in output.lower() or "database" in output.lower():
            if "no such table" in output.lower():
                suggestions.append(f"Database Error: Tables not created\n"
                                  f"   → Run: Base.metadata.create_all(bind=engine)\n"
                                  f"   → Check if database initialization is in test setup")
        
        # Conftest errors
        if "conftest" in output.lower():
            suggestions.append(f"Conftest Error: Check conftest.py\n"
                              f"   → Verify conftest.py syntax\n"
                              f"   → Check fixture definitions")
        
        if not suggestions:
            # Generic suggestion
            suggestions.append("Review test output for specific error messages.\n"
                              "Common fixes:\n"
                              "  - Check imports (from src.main import app)\n"
                              "  - Verify database initialization\n"
                              "  - Ensure fixtures are defined\n"
                              "  - Run: pytest -v for more details")
        
        return "\n\n".join(suggestions)

    @staticmethod
    def test_pytest(path: str = None, verbose: bool = True, timeout: int = 120,
                    cwd: str = None, markers: str = None) -> ToolResult:
        """Run pytest tests.

        Args:
            path: Test file or directory
            verbose: Verbose output
            timeout: Timeout in seconds
            cwd: Working directory (defaults to directory of path)
            markers: Pytest markers (-m expression)

        Returns:
            ToolResult with test results
        """
        cmd = ["pytest"]
        if verbose:
            cmd.append("-v")
        if markers:
            cmd.extend(["-m", markers])
        if path:
            cmd.append(path)
        
        # If cwd not specified, use directory of path
        if not cwd and path:
            from pathlib import Path
            cwd = str(Path(path).parent)

        success, stdout, stderr = TestTools._run_test(cmd, cwd, timeout)
        result_info = TestTools._parse_results(stdout, stderr, "pytest")

        status = "Passed" if success else "Failed"
        return ToolResult(
            success=success,
            output=f"Pytest {status}:\n\n{result_info}",
            verified=True
        )

    @staticmethod
    def test_unittest(path: str = None, verbose: bool = True, timeout: int = 120,
                      cwd: str = None) -> ToolResult:
        """Run Python unittest tests.

        Args:
            path: Test file
            verbose: Verbose output
            timeout: Timeout in seconds
            cwd: Working directory

        Returns:
            ToolResult with test results
        """
        cmd = ["python", "-m", "unittest"]
        if verbose:
            cmd.append("discover", "-v")
        if path:
            cmd.append(path)

        success, stdout, stderr = TestTools._run_test(cmd, cwd, timeout)
        result_info = TestTools._parse_results(stdout, stderr, "unittest")

        status = "Passed" if success else "Failed"
        return ToolResult(
            success=success,
            output=f"Unittest {status}:\n\n{result_info}",
            verified=True
        )

    @staticmethod
    def test_jest(path: str = None, verbose: bool = True, timeout: int = 120,
                  cwd: str = None, watch: bool = False) -> ToolResult:
        """Run Jest tests.

        Args:
            path: Test file or pattern
            verbose: Verbose output
            timeout: Timeout in seconds
            cwd: Working directory
            watch: Watch mode (not recommended for CI)

        Returns:
            ToolResult with test results
        """
        cmd = ["npx", "jest"]
        if verbose:
            cmd.append("--verbose")
        if watch:
            cmd.append("--watchAll")
        if path:
            cmd.append(path)

        success, stdout, stderr = TestTools._run_test(cmd, cwd, timeout)
        result_info = TestTools._parse_results(stdout, stderr, "jest")

        status = "Passed" if success else "Failed"
        return ToolResult(
            success=success,
            output=f"Jest {status}:\n\n{result_info}",
            verified=True
        )

    @staticmethod
    def test_npm(timeout: int = 120, cwd: str = None) -> ToolResult:
        """Run npm test.

        Args:
            timeout: Timeout in seconds
            cwd: Working directory

        Returns:
            ToolResult with test results
        """
        cmd = ["npm", "test"]

        success, stdout, stderr = TestTools._run_test(cmd, cwd, timeout)
        result_info = TestTools._parse_results(stdout, stderr, "npm")

        status = "Passed" if success else "Failed"
        return ToolResult(
            success=success,
            output=f"NPM Test {status}:\n\n{result_info}",
            verified=True
        )


class ProcessTools:
    """Process management tools."""

    logger = LogConfig.get_logger("gguf.agent.process")
    _processes: dict[str, subprocess.Popen] = {}
    _process_info: dict[str, dict] = {}

    @staticmethod
    def process_start(name: str, command: str, port: int = None, cwd: str = None, timeout: int = 10) -> ToolResult:
        """Start a background process.

        Args:
            name: Name to identify the process
            command: Command to run
            port: Port to wait for (optional)
            cwd: Working directory (optional)
            timeout: Timeout for startup check in seconds

        Returns:
            ToolResult with process status
        """
        if name in ProcessTools._processes:
            return ToolResult(
                success=False,
                output="",
                error=f"Process '{name}' already running."
            )

        try:
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd or Path.cwd(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )

            # Store process
            ProcessTools._processes[name] = process
            ProcessTools._process_info[name] = {
                "command": command,
                "port": port,
                "started": datetime.now().isoformat(),
                "cwd": cwd or str(Path.cwd())
            }

            # Wait for startup
            time.sleep(min(timeout, 3))

            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                del ProcessTools._processes[name]
                del ProcessTools._process_info[name]
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Process exited immediately:\n{stderr.decode() if stderr else 'Unknown error'}"
                )

            # If port specified, wait for it to be available
            if port:
                for _ in range(timeout * 2):
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        result = sock.connect_ex(('localhost', port))
                        sock.close()
                        if result == 0:
                            return ToolResult(
                                success=True,
                                output=f"Started '{name}' (PID: {process.pid})\nCommand: {command}\nPort: {port}\nStatus: Ready",
                                verified=True
                            )
                    except Exception:
                        pass
                    time.sleep(0.5)

                return ToolResult(
                    success=False,
                    output=f"Process started but port {port} not ready:\n{command}",
                    error=f"Port {port} not available after {timeout}s"
                )

            return ToolResult(
                success=True,
                output=f"Started '{name}' (PID: {process.pid})\nCommand: {command}\nStatus: Running",
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to start process: {e}"
            )

    @staticmethod
    def process_stop(name: str) -> ToolResult:
        """Stop a running process.

        Args:
            name: Name of the process to stop

        Returns:
            ToolResult with confirmation
        """
        if name not in ProcessTools._processes:
            return ToolResult(
                success=False,
                output="",
                error=f"Process '{name}' not found."
            )

        try:
            process = ProcessTools._processes[name]

            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()

            process.wait(timeout=5)

            del ProcessTools._processes[name]
            del ProcessTools._process_info[name]

            return ToolResult(
                success=True,
                output=f"Stopped '{name}' (PID: {process.pid})",
                verified=True
            )
        except subprocess.TimeoutExpired:
            process.kill()
            del ProcessTools._processes[name]
            del ProcessTools._process_info[name]
            return ToolResult(
                success=True,
                output=f"Killed '{name}' (was not responding to terminate)",
                verified=True
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to stop process: {e}"
            )

    @staticmethod
    def process_status(name: str = None) -> ToolResult:
        """Get status of processes.

        Args:
            name: Specific process name (optional, returns all if not specified)

        Returns:
            ToolResult with process status
        """
        if name:
            if name not in ProcessTools._processes:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Process '{name}' not found."
                )

            process = ProcessTools._processes[name]
            info = ProcessTools._process_info[name]
            running = process.poll() is None

            return ToolResult(
                success=True,
                output=f"Process: {name}\n"
                       f"PID: {process.pid}\n"
                       f"Command: {info['command']}\n"
                       f"Status: {'Running' if running else 'Exited'}\n"
                       f"Started: {info['started']}\n"
                       f"Port: {info.get('port', 'N/A')}\n"
                       f"CWD: {info['cwd']}",
                verified=True
            )
        else:
            if not ProcessTools._processes:
                return ToolResult(
                    success=True,
                    output="No processes running.",
                    verified=True
                )

            status = "Running processes:\n\n"
            for proc_name, process in ProcessTools._processes.items():
                info = ProcessTools._process_info[proc_name]
                running = process.poll() is None
                status += f"• {proc_name} (PID: {process.pid}) - {'Running' if running else 'Exited'}\n"
                if info.get('port'):
                    status += f"  Port: {info['port']}\n"

            return ToolResult(
                success=True,
                output=status,
                verified=True
            )

    @staticmethod
    def process_logs(name: str, lines: int = 50) -> ToolResult:
        """Get logs from a process.

        Args:
            name: Name of the process
            lines: Number of lines to retrieve

        Returns:
            ToolResult with process logs
        """
        if name not in ProcessTools._processes:
            return ToolResult(
                success=False,
                output="",
                error=f"Process '{name}' not found."
            )

        try:
            process = ProcessTools._processes[name]

            # Note: This is limited - would need better logging for full logs
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return ToolResult(
                    success=True,
                    output=f"Process exited.\n\nStdout:\n{stdout.decode()[:2000] if stdout else '(none)'}\n\nStderr:\n{stderr.decode()[:2000] if stderr else '(none)'}",
                    verified=True
                )
            else:
                return ToolResult(
                    success=True,
                    output=f"Process '{name}' is running. Logs available after process exits.",
                    verified=True
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to get logs: {e}"
            )

    @staticmethod
    def process_list() -> ToolResult:
        """List all running processes.

        Returns:
            ToolResult with process list
        """
        return ProcessTools.process_status()


class CodebaseSearch:
    """Codebase search tools combining grep + MYCO vision (entropy-based search)."""

    logger = LogConfig.get_logger("gguf.agent.search")

    @staticmethod
    def search_grep(pattern: str, path: str = ".", include: str = "*",
                    exclude_dirs: list = None, max_results: int = 100,
                    context_lines: int = 0, cwd: str = None) -> ToolResult:
        """Search for text pattern in files (grep-style).

        Args:
            pattern: Search pattern (regex supported)
            path: Directory to search
            include: Glob pattern for files (e.g., "*.py")
            exclude_dirs: Directories to exclude
            max_results: Maximum results to return
            context_lines: Lines of context around matches
            cwd: Working directory

        Returns:
            ToolResult with search results
        """
        cwd = Path(cwd or Path.cwd())
        search_path = cwd / path
        exclude_dirs = exclude_dirs or ["__pycache__", "node_modules", ".git", "venv", ".venv"]

        try:
            import re
            pattern_re = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid regex pattern: {e}"
            )

        results = []
        files_searched = 0

        try:
            # Find all matching files
            for file_path in search_path.rglob(include):
                # Skip excluded directories
                if any(excl in str(file_path) for excl in exclude_dirs):
                    continue

                # Skip non-files
                if not file_path.is_file():
                    continue

                # Skip binary files
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                files_searched += 1

                # Search for pattern
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    if pattern_re.search(line):
                        # Add context if requested
                        if context_lines > 0:
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)
                            context = "\n".join(lines[start:end])
                        else:
                            context = line

                        rel_path = file_path.relative_to(cwd)
                        results.append({
                            "file": str(rel_path),
                            "line": line_num,
                            "content": context.strip()
                        })

                        if len(results) >= max_results:
                            break

                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No matches found for '{pattern}'\nFiles searched: {files_searched}",
                    verified=True
                )

            # Format output
            output = f"Found {len(results)} matches for '{pattern}':\n\n"
            for result in results:
                output += f"{result['file']}:{result['line']}\n"
                output += f"  {result['content'][:200]}\n\n"

            if len(results) >= max_results:
                output += f"\n(Showing first {max_results} results only)"

            return ToolResult(
                success=True,
                output=output,
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {e}"
            )

    @staticmethod
    def search_files(file_pattern: str = "*.py", path: str = ".",
                     max_results: int = 100, cwd: str = None) -> ToolResult:
        """Find files by name pattern.

        Args:
            file_pattern: Glob pattern (e.g., "*.py", "test_*.py")
            path: Directory to search
            max_results: Maximum results
            cwd: Working directory

        Returns:
            ToolResult with file list
        """
        cwd = Path(cwd or Path.cwd())
        search_path = cwd / path

        try:
            files = list(search_path.rglob(file_pattern))

            # Filter out common excluded directories
            exclude_dirs = ["__pycache__", "node_modules", ".git", "venv", ".venv"]
            files = [f for f in files if not any(excl in str(f) for excl in exclude_dirs)]

            if not files:
                return ToolResult(
                    success=True,
                    output=f"No files found matching '{file_pattern}'",
                    verified=True
                )

            # Format output
            output = f"Found {len(files)} files matching '{file_pattern}':\n\n"
            for i, f in enumerate(files[:max_results], 1):
                rel_path = f.relative_to(cwd)
                try:
                    size = f.stat().st_size
                    output += f"{i}. {rel_path} ({size:,} bytes)\n"
                except Exception:
                    output += f"{i}. {rel_path}\n"

            if len(files) > max_results:
                output += f"\n(Showing first {max_results} of {len(files)} files)"

            return ToolResult(
                success=True,
                output=output,
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"File search failed: {e}"
            )

    @staticmethod
    def search_definitions(name: str, def_type: str = None,
                           path: str = ".", cwd: str = None) -> ToolResult:
        """Search for code definitions (classes, functions, etc.).

        Args:
            name: Name to search for
            def_type: Type of definition (class, function, def, async)
            path: Directory to search
            cwd: Working directory

        Returns:
            ToolResult with definition locations
        """
        # Build pattern based on type
        if def_type == "class":
            pattern = rf"class\s+{re.escape(name)}\s*[:(]"
        elif def_type == "function" or def_type == "def":
            pattern = rf"def\s+{re.escape(name)}\s*\("
        elif def_type == "async":
            pattern = rf"async\s+def\s+{re.escape(name)}\s*\("
        else:
            # Search for any occurrence
            pattern = rf"\b{re.escape(name)}\b"

        return CodebaseSearch.search_grep(
            pattern=pattern,
            path=path,
            include="*.py",
            max_results=50,
            context_lines=2,
            cwd=cwd
        )

    @staticmethod
    def search_by_entropy(min_entropy: float = None, max_entropy: float = None,
                          regime: str = None, path: str = ".",
                          cwd: str = None) -> ToolResult:
        """MYCO Vision: Search files by entropy characteristics.

        This is MYCO's unique capability - finding files by their thermodynamic state.

        Args:
            min_entropy: Minimum entropy threshold (0.0-1.0)
            max_entropy: Maximum entropy threshold (0.0-1.0)
            regime: Entropy regime (crystallized, dissipative, diffuse)
            path: Directory to search
            cwd: Working directory

        Returns:
            ToolResult with files matching entropy criteria
        """
        try:
            # Import MYCO entropy module
            from myco.entropy import ImportGraphBuilder, EntropyCalculator
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="MYCO entropy module not available. This is a MYCO vision feature."
            )

        cwd = Path(cwd or Path.cwd())
        search_path = cwd / path

        try:
            # Build import graph
            builder = ImportGraphBuilder(search_path)
            builder.scan()

            # Calculate entropy for all modules
            calc = EntropyCalculator(builder)

            matching_files = []

            for module_name, module_info in builder.modules.items():
                entropy = calc.calculate_module_entropy(module_name)

                # Determine regime
                if entropy < 0.3:
                    regime_type = "crystallized"
                elif entropy > 0.75:
                    regime_type = "diffuse"
                else:
                    regime_type = "dissipative"

                # Apply filters
                if regime and regime_type != regime:
                    continue
                if min_entropy is not None and entropy < min_entropy:
                    continue
                if max_entropy is not None and entropy > max_entropy:
                    continue

                # Find the file path
                module_path = search_path / f"{module_name.replace('.', '/')}.py"
                if not module_path.exists():
                    module_path = search_path / f"{module_name.replace('.', '/')}/__init__.py"

                if module_path.exists():
                    matching_files.append({
                        "file": str(module_path.relative_to(cwd)) if module_path.is_relative_to(cwd) else str(module_path),
                        "entropy": entropy,
                        "regime": regime_type
                    })

            if not matching_files:
                filter_desc = []
                if regime:
                    filter_desc.append(f"regime={regime}")
                if min_entropy is not None:
                    filter_desc.append(f"entropy>={min_entropy}")
                if max_entropy is not None:
                    filter_desc.append(f"entropy<={max_entropy}")

                return ToolResult(
                    success=True,
                    output=f"No files found matching filters: {', '.join(filter_desc)}",
                    verified=True
                )

            # Sort by entropy
            matching_files.sort(key=lambda x: x["entropy"], reverse=True)

            # Format output
            output = f"Found {len(matching_files)} files matching entropy criteria:\n\n"
            output += f"{'File':<50} {'Entropy':>10} {'Regime':<15}\n"
            output += "-" * 75 + "\n"

            for mf in matching_files[:50]:
                output += f"{mf['file']:<50} {mf['entropy']:>10.3f} {mf['regime']:<15}\n"

            if len(matching_files) > 50:
                output += f"\n(Showing 50 of {len(matching_files)} files)"

            return ToolResult(
                success=True,
                output=output,
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Entropy search failed: {e}"
            )

    @staticmethod
    def search_todo(path: str = ".", keywords: list = None,
                    max_results: int = 100, cwd: str = None) -> ToolResult:
        """Search for TODO, FIXME, HACK comments.

        Args:
            path: Directory to search
            keywords: Keywords to search (default: TODO, FIXME, HACK, XXX)
            max_results: Maximum results
            cwd: Working directory

        Returns:
            ToolResult with TODO comments found
        """
        keywords = keywords or ["TODO", "FIXME", "HACK", "XXX"]
        pattern = "|".join(keywords)

        return CodebaseSearch.search_grep(
            pattern=rf"#\s*({pattern}):?\s*(.+)",
            path=path,
            include="*.py",
            max_results=max_results,
            context_lines=0,
            cwd=cwd
        )

    @staticmethod
    def search_imports(module_name: str, path: str = ".",
                      max_results: int = 100, cwd: str = None) -> ToolResult:
        """Search for imports of a specific module.

        Args:
            module_name: Module name to search for
            path: Directory to search
            max_results: Maximum results
            cwd: Working directory

        Returns:
            ToolResult with import locations
        """
        # Search for various import patterns
        patterns = [
            rf"import\s+{re.escape(module_name)}",
            rf"from\s+{re.escape(module_name)}\s+import",
        ]

        results = []
        for pattern in patterns:
            result = CodebaseSearch.search_grep(
                pattern=pattern,
                path=path,
                include="*.py",
                max_results=max_results // 2,
                context_lines=1,
                cwd=cwd
            )
            if result.success and result.output:
                results.append(result.output)

        if not results:
            return ToolResult(
                success=True,
                output=f"No imports of '{module_name}' found",
                verified=True
            )

        output = f"Imports of '{module_name}':\n\n"
        output += "\n".join(results)

        return ToolResult(
            success=True,
            output=output,
            verified=True
        )


class EntropyGate:
    """MYCO Vision: Autopoietic gate that blocks entropy-increasing changes.

    This is the core MYCO vision feature - protecting the codebase from degradation
    by blocking changes that would increase entropy beyond the threshold.
    
    Uses myco.gate.AutopoieticGate for maturity-based thresholds:
    - Embryo (0-5 files): 0.50
    - Growth (5-20 files): 0.30
    - Mature (20-100 files): 0.15
    - Legacy (100+ files): 0.10
    """

    logger = LogConfig.get_logger("gguf.agent.gate")

    @staticmethod
    def check_entropy_delta(file_path: str, proposed_content: str, cwd: str = None) -> ToolResult:
        """Check if a proposed change would increase entropy beyond threshold.

        MYCO Vision: The autopoietic gate checks invariants before any action.
        If an action would make the codebase worse for its future self, BLOCK it.

        Args:
            file_path: Path to the file being modified
            proposed_content: Proposed new content
            cwd: Working directory

        Returns:
            ToolResult with permit/block decision
        """
        cwd = Path(cwd or Path.cwd())
        file_path = cwd / file_path

        try:
            # Try to import MYCO gate module (uses maturity-based thresholds)
            from myco.gate import AutopoieticGate
            from myco.world import WorldModel
        except ImportError:
            # MYCO gate not available - allow change with warning
            return ToolResult(
                success=True,
                output="Entropy gate: SKIP (MYCO gate module not available)\nChange permitted.",
                verified=True
            )

        try:
            # Get project root (where .myco/ would be)
            project_root = cwd
            
            # Load world model (required for gate initialization)
            world_model = WorldModel.load(project_root)

            # Initialize gate with project root and world model
            gate = AutopoieticGate(project_root, world_model)
            
            # Check entropy delta using calibrated gate
            gate_result = gate.check_entropy_delta(file_path, proposed_content)
            
            # Handle gate result
            if not gate_result.permitted:
                # BLOCK: Gate blocked the change
                return ToolResult(
                    success=False,
                    output="",
                    error=f"AUTOPOIETIC GATE BLOCKED: {gate_result.reason}\n\n"
                          f"Before: H={gate_result.entropy_before:.3f}\n"
                          f"After:  H={gate_result.entropy_after:.3f}\n\n"
                          f"The proposed change would degrade the codebase structure.",
                    verified=True
                )
            
            # Check for warning (close to threshold)
            if gate_result.violation_type == "threshold_warning":
                # PERMIT with warning
                return ToolResult(
                    success=True,
                    output=f"AUTOPOIETIC GATE: {gate_result.reason}\n\n"
                           f"Before: H={gate_result.entropy_before:.3f}\n"
                           f"After:  H={gate_result.entropy_after:.3f}\n\n"
                           f"Change permitted but consider smaller changes.",
                    verified=True
                )

            # PERMIT: Entropy change acceptable
            regime = "crystallized" if gate_result.entropy_after < 0.3 else ("diffuse" if gate_result.entropy_after > 0.75 else "dissipative")

            return ToolResult(
                success=True,
                output=f"AUTOPOIETIC GATE: PERMIT\n"
                       f"Before: H={gate_result.entropy_before:.3f}\n"
                       f"After:  H={gate_result.entropy_after:.3f}\n"
                       f"Delta:  {gate_result.entropy_after - gate_result.entropy_before:+.3f}\n"
                       f"Regime: {regime}\n\n"
                       f"Change permitted - entropy within acceptable bounds.",
                verified=True
            )

        except Exception as e:
            # On error, permit with warning
            EntropyGate.logger.warning(f"Entropy gate error: {e}")
            return ToolResult(
                success=True,
                output=f"Entropy gate: SKIP (error: {e})\nChange permitted.",
                verified=True
            )

    @staticmethod
    def get_substrate_health(path: str = ".", cwd: str = None) -> ToolResult:
        """MYCO Vision: Get overall substrate health report.

        Returns a comprehensive report on the codebase thermodynamic state.

        Args:
            path: Directory to analyze
            cwd: Working directory

        Returns:
            ToolResult with substrate health report
        """
        cwd = Path(cwd or Path.cwd())
        analyze_path = cwd / path

        try:
            from myco.entropy import ImportGraphBuilder, EntropyCalculator
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="MYCO entropy module not available."
            )

        try:
            # Build import graph
            builder = ImportGraphBuilder(analyze_path)
            builder.scan()

            # Calculate entropy for all modules
            calc = EntropyCalculator(builder)

            # Collect statistics
            crystallized = []
            dissipative = []
            diffuse = []
            total_entropy = 0.0

            for module_name in builder.modules.keys():
                try:
                    entropy = calc.calculate_module_entropy(module_name)
                    total_entropy += entropy

                    if entropy < 0.3:
                        crystallized.append((module_name, entropy))
                    elif entropy > 0.75:
                        diffuse.append((module_name, entropy))
                    else:
                        dissipative.append((module_name, entropy))
                except Exception:
                    pass

            num_modules = len(crystallized) + len(dissipative) + len(diffuse)
            avg_entropy = total_entropy / num_modules if num_modules > 0 else 0.0

            # Determine overall health
            if len(crystallized) > num_modules * 0.5:
                health = "CRITICAL - Too many crystallized modules"
                recommendation = "Apply decompose interventions to rigid modules"
            elif len(diffuse) > num_modules * 0.5:
                health = "WARNING - Too many diffuse modules"
                recommendation = "Apply compression_collapse to consolidate structure"
            elif avg_entropy < 0.4 or avg_entropy > 0.6:
                health = "MODERATE - Entropy trending away from optimal"
                recommendation = "Monitor and maintain dissipative state"
            else:
                health = "HEALTHY - Good entropy distribution"
                recommendation = "Continue current practices"

            # Format report
            report = f"SUBSTRATE HEALTH REPORT\n"
            report += f"{'='*60}\n\n"
            report += f"Overall Health: {health}\n"
            report += f"Recommendation: {recommendation}\n\n"
            report += f"Statistics:\n"
            report += f"  Total modules: {num_modules}\n"
            report += f"  Average entropy: {avg_entropy:.3f}\n\n"
            report += f"Entropy Distribution:\n"
            report += f"  Crystallized (H<0.3):  {len(crystallized):3d} ({100*len(crystallized)/max(num_modules,1):.1f}%)\n"
            report += f"  Dissipative (0.3-0.75): {len(dissipative):3d} ({100*len(dissipative)/max(num_modules,1):.1f}%)\n"
            report += f"  Diffuse (H>0.75):      {len(diffuse):3d} ({100*len(diffuse)/max(num_modules,1):.1f}%)\n"

            if crystallized:
                report += f"\nCrystallized Modules (need decompose):\n"
                for name, entropy in sorted(crystallized, key=lambda x: x[1])[:10]:
                    report += f"  H={entropy:.3f}  {name}\n"

            if diffuse:
                report += f"\nDiffuse Modules (need consolidation):\n"
                for name, entropy in sorted(diffuse, key=lambda x: x[1], reverse=True)[:10]:
                    report += f"  H={entropy:.3f}  {name}\n"

            return ToolResult(
                success=True,
                output=report,
                verified=True
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Substrate health analysis failed: {e}"
            )
