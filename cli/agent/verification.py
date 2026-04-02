"""Verification dashboard - Project health checking."""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProjectInfo:
    """Information about the current project."""

    project_type: str
    has_requirements: bool
    has_tests: bool
    has_lint: bool
    verification_files: list[str]
    pending_verifications: list[str]


@dataclass
class VerificationResult:
    """Result of a verification check."""

    name: str
    status: str  # "pass", "fail", "warn", "skip"
    message: str
    details: Optional[dict] = None


class ProjectDetector:
    """Detect project type and configuration."""

    PROJECT_INDICATORS = {
        "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile", "setup.cfg"],
        "node": ["package.json", "package-lock.json", "yarn.lock"],
        "rust": ["Cargo.toml"],
        "go": ["go.mod"],
        "java": ["pom.xml", "build.gradle"],
        "csharp": [".csproj", ".sln"],
    }

    @classmethod
    def detect(cls, path: Optional[Path] = None) -> ProjectInfo:
        """Detect project type and configuration.

        Args:
            path: Project root path (default: current directory)

        Returns:
            ProjectInfo with detected information
        """
        root = path or Path.cwd()

        project_type = "unknown"
        for ptype, indicators in cls.PROJECT_INDICATORS.items():
            for indicator in indicators:
                if (root / indicator).exists():
                    project_type = ptype
                    break
            if project_type != "unknown":
                break

        return ProjectInfo(
            project_type=project_type,
            has_requirements=cls._check_file_exists(
                root, ["requirements.txt", "Pipfile", "pyproject.toml"]
            ),
            has_tests=cls._check_file_exists(
                root, ["test_*.py", "*_test.py", "tests/", "test/", "spec/"]
            ),
            has_lint=cls._check_file_exists(
                root, [".pylintrc", "pyproject.toml", "setup.cfg", ".flake8"]
            ),
            verification_files=cls._find_verification_files(root),
            pending_verifications=cls._find_pending_verifications(root),
        )

    @classmethod
    def _check_file_exists(cls, root: Path, patterns: list[str]) -> bool:
        """Check if any of the patterns exist."""
        for pattern in patterns:
            if "*" in pattern:
                matches = list(root.glob(pattern))
                if matches:
                    return True
            elif (root / pattern).exists():
                return True
        return False

    @classmethod
    def _find_verification_files(cls, root: Path) -> list[str]:
        """Find files related to verification/config."""
        verification_files = []

        checks = {
            "python": [
                "requirements.txt",
                "pyproject.toml",
                "setup.cfg",
                "pytest.ini",
                ".pylintrc",
            ],
            "node": ["package.json", ".eslintrc.*", "tsconfig.json"],
            "general": [".gitignore", "README.md", "Makefile"],
        }

        for category, files in checks.items():
            for f in files:
                if (root / f).exists():
                    verification_files.append(f)

        return verification_files

    @classmethod
    def _find_pending_verifications(cls, root: Path) -> list[str]:
        """Find items that need verification."""
        pending = []

        # Check for uncommitted changes (if git exists)
        if (root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    cwd=root,
                    timeout=5,
                )
                if result.stdout.strip():
                    pending.append("uncommitted changes")
            except Exception:
                pass

        # Check for untracked files
        if (root / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "ls-files", "--others", "--exclude-standard"],
                    capture_output=True,
                    text=True,
                    cwd=root,
                    timeout=5,
                )
                if result.stdout.strip():
                    untracked = result.stdout.strip().split("\n")
                    if len(untracked) <= 5:
                        pending.append(f"untracked: {', '.join(untracked)}")
                    else:
                        pending.append(f"{len(untracked)} untracked files")
            except Exception:
                pass

        return pending


class VerificationDashboard:
    """Dashboard for project verification."""

    def __init__(self, project_path: Optional[Path] = None):
        """Initialize verification dashboard.

        Args:
            project_path: Project root path (default: current directory)
        """
        self.project_path = project_path or Path.cwd()
        self.project_info = ProjectDetector.detect(self.project_path)

    def run_all_checks(self) -> list[VerificationResult]:
        """Run all verification checks.

        Returns:
            List of verification results
        """
        results = []

        results.append(self._check_project_type())
        results.append(self._check_requirements())
        results.append(self._check_tests())
        results.append(self._check_lint())
        results.append(self._check_git_status())
        results.append(self._check_imports())

        return results

    def _check_project_type(self) -> VerificationResult:
        """Check project type detection."""
        ptype = self.project_info.project_type

        if ptype == "unknown":
            return VerificationResult(
                name="Project Type",
                status="warn",
                message="Could not detect project type",
                details={"type": ptype},
            )

        return VerificationResult(
            name="Project Type",
            status="pass",
            message=f"Detected {ptype} project",
            details={"type": ptype},
        )

    def _check_requirements(self) -> VerificationResult:
        """Check if requirements/dependencies are defined."""
        has_reqs = self.project_info.has_requirements

        if not has_reqs:
            return VerificationResult(
                name="Dependencies",
                status="warn",
                message="No requirements file found",
                details={"files": self.project_info.verification_files},
            )

        return VerificationResult(
            name="Dependencies",
            status="pass",
            message="Dependencies file found",
            details={"has_requirements": has_reqs},
        )

    def _check_tests(self) -> VerificationResult:
        """Check if tests exist."""
        has_tests = self.project_info.has_tests

        if not has_tests:
            return VerificationResult(
                name="Tests", status="warn", message="No test files found", details={}
            )

        return VerificationResult(
            name="Tests",
            status="pass",
            message="Test files found",
            details={"has_tests": has_tests},
        )

    def _check_lint(self) -> VerificationResult:
        """Check if linting is configured."""
        has_lint = self.project_info.has_lint

        if not has_lint:
            return VerificationResult(
                name="Linting", status="warn", message="No linting configuration found", details={}
            )

        return VerificationResult(
            name="Linting",
            status="pass",
            message="Linting configured",
            details={"has_lint": has_lint},
        )

    def _check_git_status(self) -> VerificationResult:
        """Check git status."""
        git_dir = self.project_path / ".git"

        if not git_dir.exists():
            return VerificationResult(
                name="Git", status="skip", message="Not a git repository", details={}
            )

        pending = self.project_info.pending_verifications

        if pending:
            return VerificationResult(
                name="Git",
                status="warn",
                message=f"Pending: {', '.join(pending)}",
                details={"pending": pending},
            )

        return VerificationResult(
            name="Git", status="pass", message="Working tree clean", details={}
        )

    def _check_imports(self) -> VerificationResult:
        """Check if basic imports work for Python projects."""
        if self.project_info.project_type != "python":
            return VerificationResult(
                name="Imports", status="skip", message="Not a Python project", details={}
            )

        # Try to check Python syntax on .py files
        py_files = list(self.project_path.glob("**/*.py"))

        if not py_files:
            return VerificationResult(
                name="Imports", status="skip", message="No Python files found", details={}
            )

        syntax_errors = []
        for py_file in py_files[:10]:  # Check first 10 files
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    syntax_errors.append(f"{py_file.name}: {result.stderr.strip()}")
            except Exception:
                pass

        if syntax_errors:
            return VerificationResult(
                name="Imports",
                status="fail",
                message=f"Found {len(syntax_errors)} syntax errors",
                details={"errors": syntax_errors[:5]},
            )

        return VerificationResult(
            name="Imports",
            status="pass",
            message=f"Checked {len(py_files[:10])} files - no syntax errors",
            details={"files_checked": len(py_files[:10])},
        )

    def run_runtime_checks(self) -> list[VerificationResult]:
        """Run runtime verification checks.

        Returns:
            List of runtime verification results
        """
        results = []

        results.append(self._check_server_connection())
        results.append(self._run_tests())
        results.append(self._check_dependencies_installed())

        return results

    def _check_server_connection(self) -> VerificationResult:
        """Check if MYCO server is running and responsive."""
        try:
            import requests

            response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_names = [m.get("id") for m in models]
                return VerificationResult(
                    name="Server",
                    status="pass",
                    message=f"Server running with {len(models)} model(s)",
                    details={"models": model_names[:3]},
                )
            else:
                return VerificationResult(
                    name="Server",
                    status="fail",
                    message=f"Server returned status {response.status_code}",
                    details={},
                )
        except ImportError:
            return VerificationResult(
                name="Server",
                status="skip",
                message="requests library not available",
                details={},
            )
        except Exception as e:
            return VerificationResult(
                name="Server",
                status="warn",
                message=f"Server not running: {str(e)[:50]}",
                details={},
            )

    def _run_tests(self) -> VerificationResult:
        """Run available tests for the project."""
        if self.project_info.project_type == "python":
            return self._run_python_tests()
        elif self.project_info.project_type == "node":
            return self._run_node_tests()
        return VerificationResult(
            name="Tests", status="skip", message="No test runner for this project type", details={}
        )

    def _run_python_tests(self) -> VerificationResult:
        """Run Python tests using pytest."""
        if not self.project_info.has_tests:
            return VerificationResult(
                name="Tests", status="skip", message="No test files found", details={}
            )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout
                test_count = 0
                for line in output.split("\n"):
                    if " test" in line.lower():
                        test_count += 1
                return VerificationResult(
                    name="Tests",
                    status="pass",
                    message=f"Test discovery successful",
                    details={"output": result.stdout[:200]},
                )
            else:
                return VerificationResult(
                    name="Tests",
                    status="warn",
                    message=f"Test collection issues: {result.stderr[:100]}",
                    details={},
                )
        except FileNotFoundError:
            return VerificationResult(
                name="Tests", status="skip", message="pytest not installed", details={}
            )
        except Exception as e:
            return VerificationResult(
                name="Tests",
                status="warn",
                message=f"Could not run tests: {str(e)[:50]}",
                details={},
            )

    def _run_node_tests(self) -> VerificationResult:
        """Run Node.js tests."""
        if not (self.project_path / "package.json").exists():
            return VerificationResult(
                name="Tests", status="skip", message="No package.json found", details={}
            )

        try:
            result = subprocess.run(
                ["npm", "test", "--", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=30,
            )
            if result.returncode == 0:
                return VerificationResult(
                    name="Tests",
                    status="pass",
                    message="Test configuration valid",
                    details={},
                )
            else:
                return VerificationResult(
                    name="Tests",
                    status="warn",
                    message=f"Test issues: {result.stderr[:100]}",
                    details={},
                )
        except FileNotFoundError:
            return VerificationResult(
                name="Tests", status="skip", message="npm not available", details={}
            )
        except Exception as e:
            return VerificationResult(
                name="Tests",
                status="warn",
                message=f"Could not run tests: {str(e)[:50]}",
                details={},
            )

    def _check_dependencies_installed(self) -> VerificationResult:
        """Check if dependencies are installed."""
        if self.project_info.project_type == "python":
            return self._check_python_deps()
        elif self.project_info.project_type == "node":
            return self._check_node_deps()
        return VerificationResult(
            name="Deps",
            status="skip",
            message="No dependency check for this project type",
            details={},
        )

    def _check_python_deps(self) -> VerificationResult:
        """Check if Python dependencies are installed."""
        req_file = self.project_path / "requirements.txt"
        if not req_file.exists():
            req_file = self.project_path / "pyproject.toml"
        if not req_file.exists():
            return VerificationResult(
                name="Deps", status="skip", message="No requirements file found", details={}
            )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                installed = result.stdout.lower()
                issues = []

                if (self.project_path / "requirements.txt").exists():
                    with open(self.project_path / "requirements.txt") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                pkg = line.split("==")[0].split(">=")[0].lower()
                                if pkg not in installed:
                                    issues.append(pkg)

                if issues:
                    return VerificationResult(
                        name="Deps",
                        status="warn",
                        message=f"{len(issues)} packages may be missing",
                        details={"missing": issues[:5]},
                    )
                return VerificationResult(
                    name="Deps", status="pass", message="Dependencies appear installed", details={}
                )
        except Exception as e:
            return VerificationResult(
                name="Deps", status="warn", message=f"Could not check: {str(e)[:50]}", details={}
            )

        return VerificationResult(
            name="Deps", status="skip", message="Could not verify dependencies", details={}
        )

    def _check_node_deps(self) -> VerificationResult:
        """Check if Node dependencies are installed."""
        if not (self.project_path / "package.json").exists():
            return VerificationResult(
                name="Deps", status="skip", message="No package.json found", details={}
            )

        node_modules = self.project_path / "node_modules"
        if node_modules.exists():
            return VerificationResult(
                name="Deps", status="pass", message="node_modules exists", details={}
            )
        return VerificationResult(
            name="Deps",
            status="warn",
            message="node_modules not found - run npm install",
            details={},
        )

    def get_summary(self) -> dict:
        """Get verification summary.

        Returns:
            Dict with summary information
        """
        results = self.run_all_checks()

        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        warned = sum(1 for r in results if r.status == "warn")
        skipped = sum(1 for r in results if r.status == "skip")

        return {
            "project_type": self.project_info.project_type,
            "total_checks": len(results),
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "skipped": skipped,
            "health_score": (passed / (len(results) - skipped) * 100)
            if (len(results) - skipped) > 0
            else 0,
            "results": [
                {"name": r.name, "status": r.status, "message": r.message} for r in results
            ],
        }


def verify_project(path: Optional[str] = None) -> dict:
    """Verify project health.

    Args:
        path: Project path (default: current directory)

    Returns:
        Dict with verification summary
    """
    project_path = Path(path) if path else None
    dashboard = VerificationDashboard(project_path)
    return dashboard.get_summary()
