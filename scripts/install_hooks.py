#!/usr/bin/env python3
"""Evidence Suite - Git Hooks Installer
Installs pre-commit hooks for mandatory testing.
"""

import os
import stat
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
GIT_DIR = PROJECT_ROOT / ".git"
HOOKS_DIR = GIT_DIR / "hooks"

PRE_COMMIT_HOOK = """#!/bin/sh
# Evidence Suite Pre-Commit Hook
# Rule: Any upgrade must test and benchmark metrics

echo "Running pre-commit checks..."
python scripts/pre_commit_check.py --no-benchmark

if [ $? -ne 0 ]; then
    echo ""
    echo "Pre-commit checks failed!"
    echo "Fix the issues above before committing."
    echo ""
    echo "To bypass (not recommended): git commit --no-verify"
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
"""

PRE_PUSH_HOOK = """#!/bin/sh
# Evidence Suite Pre-Push Hook
# Runs full benchmark before push

echo "Running pre-push benchmarks..."
python scripts/pre_commit_check.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Pre-push checks failed!"
    echo "Fix the issues above before pushing."
    echo ""
    echo "To bypass (not recommended): git push --no-verify"
    exit 1
fi

echo "Pre-push checks passed!"
exit 0
"""


def install_hooks():
    """Install git hooks."""
    if not GIT_DIR.exists():
        print("Error: Not a git repository. Run 'git init' first.")
        return False

    HOOKS_DIR.mkdir(exist_ok=True)

    # Install pre-commit hook
    pre_commit_path = HOOKS_DIR / "pre-commit"
    with open(pre_commit_path, "w", newline="\n") as f:
        f.write(PRE_COMMIT_HOOK)

    # Make executable (Unix)
    if os.name != "nt":
        st = os.stat(pre_commit_path)
        os.chmod(pre_commit_path, st.st_mode | stat.S_IEXEC)

    print(f"Installed pre-commit hook: {pre_commit_path}")

    # Install pre-push hook
    pre_push_path = HOOKS_DIR / "pre-push"
    with open(pre_push_path, "w", newline="\n") as f:
        f.write(PRE_PUSH_HOOK)

    if os.name != "nt":
        st = os.stat(pre_push_path)
        os.chmod(pre_push_path, st.st_mode | stat.S_IEXEC)

    print(f"Installed pre-push hook: {pre_push_path}")

    print("\nGit hooks installed successfully!")
    print("\nRule enforced: Any upgrade must test and benchmark metrics")
    print("  - pre-commit: Runs unit tests and pipeline verification")
    print("  - pre-push: Runs full benchmarks with regression detection")
    print("\nTo bypass (not recommended): git commit/push --no-verify")

    return True


def uninstall_hooks():
    """Uninstall git hooks."""
    pre_commit_path = HOOKS_DIR / "pre-commit"
    pre_push_path = HOOKS_DIR / "pre-push"

    if pre_commit_path.exists():
        pre_commit_path.unlink()
        print(f"Removed: {pre_commit_path}")

    if pre_push_path.exists():
        pre_push_path.unlink()
        print(f"Removed: {pre_push_path}")

    print("Git hooks uninstalled.")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install/uninstall git hooks")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall hooks")

    args = parser.parse_args()

    if args.uninstall:
        uninstall_hooks()
    else:
        install_hooks()
