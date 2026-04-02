"""Approval system for risky operations."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ApprovalRule:
    """Rule for when approval is required."""

    pattern: str
    always_require: bool = False
    blocked: bool = False
    auto_approve_under_mb: Optional[int] = None
    description: str = ""


# Default approval rules
DEFAULT_APPROVAL_RULES = {
    "pip install": ApprovalRule(
        pattern="pip install", auto_approve_under_mb=10, description="Install Python packages"
    ),
    "pip uninstall": ApprovalRule(
        pattern="pip uninstall", always_require=True, description="Uninstall Python packages"
    ),
    "npm install": ApprovalRule(
        pattern="npm install", auto_approve_under_mb=50, description="Install Node.js packages"
    ),
    "npm uninstall": ApprovalRule(
        pattern="npm uninstall", always_require=True, description="Uninstall Node.js packages"
    ),
    "rm": ApprovalRule(pattern="rm", always_require=True, description="Remove files/directories"),
    "del": ApprovalRule(pattern="del", always_require=True, description="Delete files (Windows)"),
    "rmdir": ApprovalRule(pattern="rmdir", always_require=True, description="Remove directory"),
    "format": ApprovalRule(pattern="format", blocked=True, description="Format disk (BLOCKED)"),
    "mkfs": ApprovalRule(pattern="mkfs", blocked=True, description="Make filesystem (BLOCKED)"),
    "shutdown": ApprovalRule(
        pattern="shutdown", blocked=True, description="Shutdown system (BLOCKED)"
    ),
    "reboot": ApprovalRule(pattern="reboot", blocked=True, description="Reboot system (BLOCKED)"),
    "sudo": ApprovalRule(
        pattern="sudo", always_require=True, description="Run as administrator/root"
    ),
    "runas": ApprovalRule(
        pattern="runas", always_require=True, description="Run as different user (Windows)"
    ),
}


class ApprovalManager:
    """Manages approval requirements and user choices."""

    DEFAULT_TIMEOUT_SECONDS = 60

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize approval manager.

        Args:
            config_dir: Directory for approval config (default: ~/.myco)
        """
        self.config_dir = config_dir or Path.home() / ".myco"
        self.config_file = self.config_dir / "approval_config.json"
        self.log_file = self.config_dir / "approvals.log"

        # Load user preferences
        self.user_rules = self._load_user_rules()
        self.remembered_choices = self._load_remembered_choices()
        self.trusted_patterns = self._load_trusted_patterns()

        # Session-level batch approvals
        self.session_approved_patterns: set = set()

        # Timeout setting
        self.timeout_seconds = self._load_timeout()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_user_rules(self) -> dict:
        """Load user-customized rules."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("rules", {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _load_remembered_choices(self) -> dict:
        """Load remembered user choices."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("remembered", {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _load_trusted_patterns(self) -> dict:
        """Load trusted command patterns (auto-approved without prompt)."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("trusted_patterns", {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _load_timeout(self) -> int:
        """Load approval timeout setting."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    return config.get("timeout_seconds", self.DEFAULT_TIMEOUT_SECONDS)
            except (json.JSONDecodeError, IOError):
                pass
        return self.DEFAULT_TIMEOUT_SECONDS

    def check_approval_required(self, command: str) -> tuple[bool, Optional[ApprovalRule]]:
        """Check if command requires approval.

        Args:
            command: Command to check

        Returns:
            Tuple of (requires_approval, rule)
        """
        # Check if command is blocked
        for rule in DEFAULT_APPROVAL_RULES.values():
            if rule.pattern.lower() in command.lower():
                if rule.blocked:
                    return True, rule  # Blocked = always requires approval (and will be denied)

        # Check if command matches any approval rule
        for pattern, rule in DEFAULT_APPROVAL_RULES.items():
            if pattern.lower() in command.lower():
                # Check session-approved patterns (batch approval)
                if pattern in self.session_approved_patterns:
                    return False, None

                # Check if user has remembered choice
                if pattern in self.remembered_choices:
                    return False, None  # User said to remember, auto-approve

                # Check trusted patterns (default approvals)
                if pattern in self.trusted_patterns:
                    return False, None

                # Check if user has customized rule
                if pattern in self.user_rules:
                    user_rule = self.user_rules[pattern]
                    if user_rule.get("always_require", False):
                        return True, rule
                    else:
                        return False, None

                # Default: require approval
                return True, rule

        # No matching rule, no approval needed
        return False, None

    def is_blocked(self, command: str) -> bool:
        """Check if command is blocked entirely."""
        for rule in DEFAULT_APPROVAL_RULES.values():
            if rule.pattern.lower() in command.lower():
                return rule.blocked
        return False

    def log_decision(self, command: str, approved: bool, remembered: bool = False):
        """Log approval decision.

        Args:
            command: Command that was approved/denied
            approved: Whether it was approved
            remembered: Whether user chose to remember this
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "command": command,
            "approved": approved,
            "remembered": remembered,
        }

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def remember_choice(self, pattern: str, approve: bool):
        """Remember user's choice for future.

        Args:
            pattern: Command pattern to remember
            approve: Whether to auto-approve in future
        """
        self.remembered_choices[pattern] = approve
        self._save_config()

    def add_trusted_pattern(self, pattern: str, description: str = ""):
        """Add a trusted pattern that doesn't require approval.

        Args:
            pattern: Command pattern to trust
            description: Optional description
        """
        self.trusted_patterns[pattern] = {"enabled": True, "description": description}
        self._save_config()

    def remove_trusted_pattern(self, pattern: str):
        """Remove a trusted pattern.

        Args:
            pattern: Command pattern to remove from trusted
        """
        if pattern in self.trusted_patterns:
            del self.trusted_patterns[pattern]
            self._save_config()

    def approve_for_session(self, pattern: str):
        """Approve command pattern for current session (batch approval).

        Args:
            pattern: Command pattern to approve for session
        """
        self.session_approved_patterns.add(pattern)

    def clear_session_approvals(self):
        """Clear all session-level approvals."""
        self.session_approved_patterns.clear()

    def set_timeout(self, seconds: int):
        """Set approval timeout.

        Args:
            seconds: Timeout in seconds (0 to disable)
        """
        self.timeout_seconds = max(0, min(seconds, 300))  # Clamp 0-5 min
        self._save_config()

    def _save_config(self):
        """Save configuration to file."""
        config = {
            "rules": self.user_rules,
            "remembered": self.remembered_choices,
            "trusted_patterns": self.trusted_patterns,
            "timeout_seconds": self.timeout_seconds,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_approval_history(self, limit: int = 20) -> list[dict]:
        """Get recent approval decisions.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of approval log entries
        """
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Return most recent first, limited
        return list(reversed(entries[-limit:]))

    def clear_history(self):
        """Clear approval history."""
        if self.log_file.exists():
            self.log_file.unlink()

    def reset_remembered_choices(self):
        """Reset all remembered choices."""
        self.remembered_choices = {}
        self._save_config()
