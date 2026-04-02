"""Enhanced approval context - estimate impact of commands."""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


class CommandImpactEstimator:
    """Estimate the impact of a command before execution."""

    def __init__(self):
        self.cwd = Path.cwd()

    def estimate_pip_install(self, command: str) -> dict:
        """Estimate impact of pip install command.

        Args:
            command: Full pip install command

        Returns:
            Dict with package_count, estimated_size_mb, packages list
        """
        result = {"package_count": 0, "estimated_size_mb": 0.0, "packages": [], "note": ""}

        # Try to extract requirements file
        req_match = re.search(r"-r\s+(\S+)", command)
        pkg_match = re.findall(r"install\s+([^\s\-]+)", command)

        try:
            # Check for requirements file
            if req_match:
                req_file = req_match.group(1)
                if Path(req_file).exists():
                    with open(req_file, "r") as f:
                        packages = [
                            line.strip().split("==")[0].split(">=")[0].split("<")[0].split("[")[0]
                            for line in f
                            if line.strip() and not line.startswith("#")
                        ]
                        result["package_count"] = len(packages)
                        result["packages"] = packages[:10]
                        result["note"] = f"from {req_file}"
                        result["estimated_size_mb"] = len(packages) * 3.5
                else:
                    result["note"] = f"requirements file not found: {req_file}"
                    # Still estimate based on package names in command
                    if pkg_match:
                        result["packages"] = pkg_match[:5]
                        result["package_count"] = len(pkg_match)
                        result["estimated_size_mb"] = len(pkg_match) * 3.5
            # Check for individual packages
            elif pkg_match:
                packages = [p for p in pkg_match if not p.startswith("-")]
                result["packages"] = packages[:10]
                result["package_count"] = len(packages)
                result["estimated_size_mb"] = len(packages) * 3.5
                result["note"] = "individual packages"
            else:
                result["note"] = "no packages specified or requirements.txt found"

        except Exception as e:
            result["note"] = f"Could not parse requirements: {e}"

        return result

    def estimate_npm_install(self, command: str) -> dict:
        """Estimate impact of npm install command.

        Args:
            command: Full npm install command

        Returns:
            Dict with package_count, estimated_size_mb
        """
        result = {"package_count": 0, "estimated_size_mb": 0.0, "packages": [], "note": ""}

        # Check for package.json in current directory
        pkg_json_path = Path("package.json")

        try:
            if pkg_json_path.exists():
                import json

                with open(pkg_json_path, "r") as f:
                    data = json.load(f)

                deps = data.get("dependencies", {})
                dev_deps = data.get("devDependencies", {})
                total = len(deps) + len(dev_deps)

                result["package_count"] = total
                result["packages"] = list(deps.keys())[:5] + list(dev_deps.keys())[:5]
                result["note"] = "from package.json"
                result["estimated_size_mb"] = total * 7.5
            else:
                # Check for individual packages in command
                pkg_match = re.findall(r"install\s+(?:@)?([^\s\-]+)", command)
                if pkg_match:
                    packages = [p for p in pkg_match if not p.startswith("-")]
                    result["packages"] = packages[:10]
                    result["package_count"] = len(packages)
                    result["estimated_size_mb"] = len(packages) * 7.5
                    result["note"] = "individual packages"
                else:
                    result["note"] = "no package.json found, installing all dependencies"
                    result["estimated_size_mb"] = 100.0  # Conservative estimate

        except Exception as e:
            result["note"] = f"Could not parse package.json: {e}"

        return result

    def estimate_disk_change(self, command: str) -> dict:
        """Estimate disk space change from command.

        Args:
            command: Command to analyze

        Returns:
            Dict with estimated_change_mb, files_affected
        """
        result = {"estimated_change_mb": 0.0, "files_affected": [], "directories_affected": []}

        # Detect file operations
        file_patterns = [
            (r"cp\s+(\S+)\s+(\S+)", "copy"),
            (r"mv\s+(\S+)\s+(\S+)", "move"),
            (r"rm\s+(.+)", "delete"),
            (r"del\s+(.+)", "delete"),
            (r"mkdir\s+(.+)", "create_dir"),
            (r"touch\s+(.+)", "create_file"),
        ]

        for pattern, op_type in file_patterns:
            matches = re.findall(pattern, command)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        # cp, mv have 2 args
                        files = list(match)
                    else:
                        # rm, mkdir, touch have 1 arg
                        files = [match]

                    for file in files:
                        file = file.strip('"').strip("'")
                        if op_type in ["delete", "move"]:
                            result["files_affected"].append(f"🗑️ {file}")
                        elif op_type in ["copy", "create_file"]:
                            result["files_affected"].append(f"📝 {file}")
                        elif op_type == "create_dir":
                            result["directories_affected"].append(f"📁 {file}")

        return result

    def get_environment_impact(self, command: str) -> dict:
        """Get environment impact (venv, system-wide, etc.).

        Args:
            command: Command to analyze

        Returns:
            Dict with scope, modifies
        """
        result = {"scope": "unknown", "modifies": []}

        # Check if in virtual environment
        venv = os.environ.get("VIRTUAL_ENV")
        in_venv = venv is not None

        if "pip install" in command.lower():
            if in_venv:
                result["scope"] = "virtualenv"
                result["modifies"] = [
                    f"{venv}/lib/",
                    f"{venv}/Scripts/" if os.name == "nt" else f"{venv}/bin/",
                ]
            else:
                result["scope"] = "system-wide"
                result["modifies"] = ["site-packages/", "Scripts/" if os.name == "nt" else "bin/"]
                result["warning"] = "⚠️ Installing system-wide (consider using virtualenv)"

        elif "npm install" in command.lower():
            result["scope"] = "project"
            result["modifies"] = ["node_modules/", "package-lock.json"]

        elif "rm" in command.lower() or "del" in command.lower():
            result["scope"] = "filesystem"
            result["warning"] = "⚠️ File deletion is permanent"

        return result

    def analyze_command(self, command: str) -> dict:
        """Analyze command and return full impact assessment.

        Args:
            command: Command to analyze

        Returns:
            Dict with all impact information
        """
        impact = {
            "command": command,
            "package_info": None,
            "disk_change": None,
            "environment": None,
            "risk_level": "low",
            "summary": [],
        }

        # Package installation detection
        if "pip install" in command.lower():
            impact["package_info"] = self.estimate_pip_install(command)
            impact["risk_level"] = (
                "medium" if impact["package_info"]["estimated_size_mb"] > 50 else "low"
            )

        elif "npm install" in command.lower():
            impact["package_info"] = self.estimate_npm_install(command)
            impact["risk_level"] = (
                "medium" if impact["package_info"]["estimated_size_mb"] > 100 else "low"
            )

        # Disk change detection
        impact["disk_change"] = self.estimate_disk_change(command)

        # Environment impact
        impact["environment"] = self.get_environment_impact(command)

        # Build summary
        if impact["package_info"]:
            pkg = impact["package_info"]
            if pkg["package_count"] > 0:
                impact["summary"].append(f"📦 {pkg['package_count']} packages")
            if pkg["estimated_size_mb"] > 0:
                impact["summary"].append(f"💾 ~{pkg['estimated_size_mb']:.1f} MB")
            if pkg["packages"]:
                impact["summary"].append(f"First: {', '.join(pkg['packages'][:3])}")
            if pkg.get("note"):
                impact["summary"].append(f"Note: {pkg['note']}")

        if impact["environment"].get("warning"):
            impact["summary"].append(impact["environment"]["warning"])

        if impact["disk_change"]["files_affected"]:
            impact["summary"].extend(impact["disk_change"]["files_affected"][:3])

        return impact


# Convenience function
def get_command_impact(command: str) -> dict:
    """Get impact assessment for a command.

    Args:
        command: Command to analyze

    Returns:
        Dict with impact information
    """
    estimator = CommandImpactEstimator()
    return estimator.analyze_command(command)
