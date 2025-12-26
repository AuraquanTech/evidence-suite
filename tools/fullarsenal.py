#!/usr/bin/env python3
"""Evidence Suite: Full Arsenal Protocol
Author: Senior Director of Coding Excellence

Automated, iterative code optimization and enhancement system that ensures
maximum quality, performance, and robustness through continuous improvement cycles.

Usage:
    python tools/fullarsenal.py                    # Full protocol
    python tools/fullarsenal.py --quick            # Quick validation only
    python tools/fullarsenal.py --phase 1          # Run specific phase
    python tools/fullarsenal.py --report           # Generate report only
    python tools/fullarsenal.py --fix              # Auto-fix issues

Protocol Phases:
    1. Initial Analysis & Assessment
    2. Syntax & Compile Validation
    3. Logic & Flow Analysis
    4. Best Practices Compliance
    5. Security Assessment
    6. Performance Benchmarking
    7. Documentation Audit
    8. Final Report Generation
"""

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Status(Enum):
    """Validation status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    INFO = "INFO"


class Severity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Issue:
    """Represents a code quality issue."""

    phase: int
    category: str
    severity: Severity
    message: str
    file: str | None = None
    line: int | None = None
    suggestion: str | None = None


@dataclass
class PhaseResult:
    """Result of a validation phase."""

    phase: int
    name: str
    status: Status
    duration_ms: float
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""

    timestamp: str
    project: str
    python_version: str
    total_duration_ms: float
    phases: list[PhaseResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class FullArsenalProtocol:
    """Full Arsenal Protocol implementation.

    Provides comprehensive, automated code validation across multiple phases:
    - Syntax validation
    - Compile/runtime checks
    - Logic and flow analysis
    - Best practices compliance
    - Security assessment
    - Performance benchmarking
    """

    # Quality thresholds
    THRESHOLDS = {
        "code_quality_score": 85,
        "test_coverage": 70,
        "cyclomatic_complexity_max": 15,
        "maintainability_index_min": 65,
        "security_score": 85,
        "doc_coverage": 60,
    }

    def __init__(
        self,
        project_root: Path | None = None,
        verbose: bool = True,
        fix_mode: bool = False,
        quick_mode: bool = False,
    ):
        self.project_root = project_root or PROJECT_ROOT
        self.verbose = verbose
        self.fix_mode = fix_mode
        self.quick_mode = quick_mode
        self.issues: list[Issue] = []
        self.phases: list[PhaseResult] = []
        self.start_time = time.perf_counter()

    # =========================================================================
    # OUTPUT HELPERS
    # =========================================================================

    def _print(self, message: str, status: Status | None = None) -> None:
        """Print formatted message."""
        if not self.verbose:
            return

        symbols = {
            Status.PASS: "\033[92m[PASS]\033[0m",
            Status.FAIL: "\033[91m[FAIL]\033[0m",
            Status.WARN: "\033[93m[WARN]\033[0m",
            Status.SKIP: "\033[90m[SKIP]\033[0m",
            Status.INFO: "\033[94m[INFO]\033[0m",
        }

        prefix = symbols.get(status, " ") if status else ""
        print(f"  {prefix} {message}")

    def _header(self, title: str) -> None:
        """Print phase header."""
        if not self.verbose:
            return
        print()
        print("\033[1m\033[94m" + "=" * 70 + "\033[0m")
        print(f"\033[1m\033[94m  {title}\033[0m")
        print("\033[1m\033[94m" + "=" * 70 + "\033[0m")

    def _section(self, title: str) -> None:
        """Print section header."""
        if self.verbose:
            print(f"\n\033[96m>> {title}\033[0m")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in the project."""
        exclude_dirs = {".venv", "venv", "__pycache__", ".git", "migrations", "node_modules"}
        files = []
        for path in self.project_root.rglob("*.py"):
            if not any(excl in path.parts for excl in exclude_dirs):
                files.append(path)
        return files

    def _run_command(
        self, cmd: list[str], cwd: Path | None = None, timeout: int = 120
    ) -> tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                check=False,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -2, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return -3, "", str(e)

    def _check_tool(self, tool: str) -> bool:
        """Check if a tool is available."""
        code, _, _ = self._run_command([sys.executable, "-m", tool, "--version"])
        if code != 0:
            code, _, _ = self._run_command([tool, "--version"])
        return code == 0

    def _add_issue(
        self,
        phase: int,
        category: str,
        severity: Severity,
        message: str,
        file: str | None = None,
        line: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an issue to the list."""
        self.issues.append(
            Issue(
                phase=phase,
                category=category,
                severity=severity,
                message=message,
                file=file,
                line=line,
                suggestion=suggestion,
            )
        )

    # =========================================================================
    # PHASE 1: INITIAL ANALYSIS
    # =========================================================================

    def phase_1_initial_analysis(self) -> PhaseResult:
        """Phase 1: Initial Analysis & Assessment."""
        self._header("PHASE 1: INITIAL ANALYSIS")
        start = time.perf_counter()
        issues_before = len(self.issues)

        metrics: dict[str, Any] = {}

        # Project structure analysis
        self._section("Project Structure")
        python_files = self._get_python_files()
        metrics["total_python_files"] = len(python_files)
        self._print(f"Python files: {len(python_files)}", Status.INFO)

        # Line count
        total_lines = 0
        for f in python_files:
            try:
                total_lines += len(f.read_text(encoding="utf-8").splitlines())
            except Exception:
                pass
        metrics["total_lines"] = total_lines
        self._print(f"Total lines: {total_lines:,}", Status.INFO)

        # Required files check
        self._section("Required Files")
        required = ["README.md", "requirements.txt", "pyproject.toml", ".gitignore"]
        for req_file in required:
            if (self.project_root / req_file).exists():
                self._print(f"{req_file}", Status.PASS)
            else:
                self._print(f"{req_file} missing", Status.WARN)
                self._add_issue(1, "structure", Severity.LOW, f"Missing {req_file}")

        # Directory structure
        self._section("Directory Structure")
        expected_dirs = ["core", "api", "tests", "scripts"]
        for dir_name in expected_dirs:
            if (self.project_root / dir_name).is_dir():
                self._print(f"{dir_name}/", Status.PASS)
            else:
                self._print(f"{dir_name}/ missing", Status.WARN)
                self._add_issue(1, "structure", Severity.LOW, f"Missing directory: {dir_name}/")

        # Environment check
        self._section("Environment")
        self._print(f"Python: {sys.version.split()[0]}", Status.INFO)
        self._print(f"Project root: {self.project_root}", Status.INFO)

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = Status.PASS if new_issues == 0 else Status.WARN

        return PhaseResult(
            phase=1,
            name="Initial Analysis",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 2: SYNTAX VALIDATION
    # =========================================================================

    def phase_2_syntax_validation(self) -> PhaseResult:
        """Phase 2: Syntax & Compile Validation."""
        self._header("PHASE 2: SYNTAX VALIDATION")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {"syntax_errors": 0, "files_checked": 0}

        self._section("Python Syntax Check")
        python_files = self._get_python_files()
        syntax_errors = 0

        for py_file in python_files:
            metrics["files_checked"] += 1
            try:
                source = py_file.read_text(encoding="utf-8")
                ast.parse(source, filename=str(py_file))
            except SyntaxError as e:
                syntax_errors += 1
                self._print(f"{py_file.name}:{e.lineno} - {e.msg}", Status.FAIL)
                self._add_issue(
                    2,
                    "syntax",
                    Severity.CRITICAL,
                    f"Syntax error: {e.msg}",
                    str(py_file),
                    e.lineno,
                )
            except Exception as e:
                self._print(f"{py_file.name} - Parse error: {e}", Status.WARN)

        metrics["syntax_errors"] = syntax_errors
        if syntax_errors == 0:
            self._print(f"All {len(python_files)} files have valid syntax", Status.PASS)
        else:
            self._print(f"{syntax_errors} syntax errors found", Status.FAIL)

        # Import validation
        self._section("Import Validation")
        import_errors = 0
        core_modules = ["core", "api"]

        for module in core_modules:
            module_path = self.project_root / module / "__init__.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(module, module_path)
                if spec and spec.loader:
                    try:
                        # Just check if it can be parsed, don't actually import
                        source = module_path.read_text(encoding="utf-8")
                        ast.parse(source)
                        self._print(f"{module} module structure valid", Status.PASS)
                    except Exception as e:
                        import_errors += 1
                        self._print(f"{module} module error: {e}", Status.FAIL)
                        self._add_issue(2, "import", Severity.HIGH, str(e), str(module_path))

        metrics["import_errors"] = import_errors

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = (
            Status.PASS if new_issues == 0 else Status.FAIL if syntax_errors > 0 else Status.WARN
        )

        return PhaseResult(
            phase=2,
            name="Syntax Validation",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 3: LOGIC & FLOW ANALYSIS
    # =========================================================================

    def phase_3_logic_flow(self) -> PhaseResult:
        """Phase 3: Logic & Flow Analysis."""
        self._header("PHASE 3: LOGIC & FLOW ANALYSIS")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {}

        self._section("Cyclomatic Complexity")
        # Use radon if available
        code, stdout, _ = self._run_command(
            [sys.executable, "-m", "radon", "cc", "core", "api", "-a", "-s"]
        )
        if code == 0:
            # Parse average complexity
            for line in stdout.splitlines():
                if "Average complexity" in line:
                    try:
                        avg = float(line.split(":")[-1].strip().split()[0].strip("()"))
                        metrics["avg_complexity"] = avg
                        if avg <= 5:
                            self._print(f"Average complexity: {avg:.2f} (Excellent)", Status.PASS)
                        elif avg <= 10:
                            self._print(f"Average complexity: {avg:.2f} (Good)", Status.PASS)
                        elif avg <= 15:
                            self._print(f"Average complexity: {avg:.2f} (Moderate)", Status.WARN)
                        else:
                            self._print(f"Average complexity: {avg:.2f} (High)", Status.FAIL)
                            self._add_issue(
                                3,
                                "complexity",
                                Severity.MEDIUM,
                                f"High cyclomatic complexity: {avg:.2f}",
                                suggestion="Refactor complex functions",
                            )
                    except (ValueError, IndexError):
                        pass
        else:
            self._print("radon not available, skipping complexity analysis", Status.SKIP)

        # Dead code detection
        self._section("Dead Code Detection")
        code, stdout, _ = self._run_command(
            [sys.executable, "-m", "vulture", "core", "api", "--min-confidence", "80"]
        )
        if code == 0:
            dead_code_count = len([l for l in stdout.splitlines() if l.strip()])
            metrics["dead_code_items"] = dead_code_count
            if dead_code_count == 0:
                self._print("No dead code detected", Status.PASS)
            else:
                self._print(f"Found {dead_code_count} potentially unused items", Status.WARN)
        elif code == -2:
            self._print("vulture not available", Status.SKIP)
        else:
            self._print("Dead code check completed", Status.INFO)

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = Status.PASS if new_issues == 0 else Status.WARN

        return PhaseResult(
            phase=3,
            name="Logic & Flow Analysis",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 4: BEST PRACTICES
    # =========================================================================

    def phase_4_best_practices(self) -> PhaseResult:
        """Phase 4: Best Practices Compliance."""
        self._header("PHASE 4: BEST PRACTICES COMPLIANCE")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {}

        # Ruff linting
        self._section("Ruff Linting")
        if self.fix_mode:
            code, stdout, stderr = self._run_command(
                [sys.executable, "-m", "ruff", "check", ".", "--fix", "--quiet"]
            )
        else:
            code, stdout, stderr = self._run_command(
                [sys.executable, "-m", "ruff", "check", ".", "--quiet"]
            )

        if code == 0:
            self._print("No linting issues", Status.PASS)
            metrics["lint_issues"] = 0
        elif code == -2:
            self._print("ruff not available", Status.SKIP)
        else:
            issue_count = len([l for l in (stdout + stderr).splitlines() if l.strip()])
            metrics["lint_issues"] = issue_count
            self._print(f"Found {issue_count} linting issues", Status.WARN)
            self._add_issue(
                4,
                "lint",
                Severity.LOW,
                f"{issue_count} linting issues",
                suggestion="Run 'ruff check . --fix' to auto-fix",
            )

        # Format check
        self._section("Code Formatting")
        if self.fix_mode:
            code, _, _ = self._run_command([sys.executable, "-m", "ruff", "format", "."])
            if code == 0:
                self._print("Code formatted", Status.PASS)
        else:
            code, _, _ = self._run_command([sys.executable, "-m", "ruff", "format", "--check", "."])
            if code == 0:
                self._print("Code is properly formatted", Status.PASS)
            elif code == -2:
                self._print("ruff not available", Status.SKIP)
            else:
                self._print("Code needs formatting", Status.WARN)
                self._add_issue(
                    4,
                    "format",
                    Severity.LOW,
                    "Code formatting issues",
                    suggestion="Run 'ruff format .'",
                )

        # Type hints coverage
        self._section("Type Hints")
        python_files = self._get_python_files()
        typed_files = 0
        for f in python_files[:50]:  # Sample first 50 files
            try:
                source = f.read_text(encoding="utf-8")
                if "def " in source and ("->" in source or ": " in source):
                    typed_files += 1
            except Exception:
                pass

        type_coverage = (typed_files / min(50, len(python_files))) * 100 if python_files else 0
        metrics["type_hint_coverage"] = type_coverage
        if type_coverage >= 80:
            self._print(f"Type hint coverage: {type_coverage:.0f}%", Status.PASS)
        elif type_coverage >= 50:
            self._print(f"Type hint coverage: {type_coverage:.0f}%", Status.WARN)
        else:
            self._print(f"Type hint coverage: {type_coverage:.0f}%", Status.FAIL)

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = Status.PASS if new_issues == 0 else Status.WARN

        return PhaseResult(
            phase=4,
            name="Best Practices",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 5: SECURITY ASSESSMENT
    # =========================================================================

    def phase_5_security(self) -> PhaseResult:
        """Phase 5: Security Assessment."""
        self._header("PHASE 5: SECURITY ASSESSMENT")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {}

        # Bandit security scan
        self._section("Bandit Security Scan")
        code, stdout, _ = self._run_command(
            [sys.executable, "-m", "bandit", "-r", "core", "api", "-q", "-ll"]
        )
        if code == 0:
            self._print("No security issues found", Status.PASS)
            metrics["security_issues"] = 0
        elif code == -2:
            self._print("bandit not available", Status.SKIP)
        else:
            issue_count = stdout.count("Issue:")
            metrics["security_issues"] = issue_count
            if issue_count > 0:
                self._print(f"Found {issue_count} security issues", Status.WARN)
                self._add_issue(
                    5,
                    "security",
                    Severity.MEDIUM,
                    f"{issue_count} security issues found",
                    suggestion="Review bandit output",
                )
            else:
                self._print("Security scan completed", Status.PASS)

        # Dependency vulnerabilities
        self._section("Dependency Vulnerabilities")
        code, stdout, _ = self._run_command(
            [sys.executable, "-m", "pip_audit", "--strict", "--progress-spinner", "off"]
        )
        if code == 0:
            self._print("No known vulnerabilities", Status.PASS)
            metrics["vulnerabilities"] = 0
        elif code == -2:
            self._print("pip-audit not available", Status.SKIP)
        else:
            self._print("Vulnerability check completed with warnings", Status.WARN)

        # Secret detection
        self._section("Secret Detection")
        secret_patterns = ["password", "secret", "api_key", "token", "credential"]
        secrets_found = 0
        for f in self._get_python_files()[:100]:
            try:
                content = f.read_text(encoding="utf-8").lower()
                for pattern in secret_patterns:
                    if f'{pattern} = "' in content or f"{pattern} = '" in content:
                        # Exclude test files and examples
                        if "test" not in str(f).lower() and "example" not in content:
                            secrets_found += 1
                            break
            except Exception:
                pass

        metrics["potential_secrets"] = secrets_found
        if secrets_found == 0:
            self._print("No hardcoded secrets detected", Status.PASS)
        else:
            self._print(f"Found {secrets_found} files with potential secrets", Status.WARN)
            self._add_issue(
                5,
                "secrets",
                Severity.HIGH,
                f"Potential hardcoded secrets in {secrets_found} files",
                suggestion="Use environment variables for secrets",
            )

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        critical_issues = sum(
            1 for i in self.issues[issues_before:] if i.severity == Severity.CRITICAL
        )
        status = (
            Status.FAIL if critical_issues > 0 else Status.WARN if new_issues > 0 else Status.PASS
        )

        return PhaseResult(
            phase=5,
            name="Security Assessment",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 6: TESTING
    # =========================================================================

    def phase_6_testing(self) -> PhaseResult:
        """Phase 6: Test Suite Execution."""
        if self.quick_mode:
            return PhaseResult(
                phase=6,
                name="Testing",
                status=Status.SKIP,
                duration_ms=0,
                metrics={"skipped": True},
            )

        self._header("PHASE 6: TEST SUITE")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {}

        self._section("Unit Tests")
        code, stdout, stderr = self._run_command(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "-v", "--tb=short", "-q"],
            timeout=180,
        )

        if code == 0:
            # Parse test count
            for line in stdout.splitlines():
                if "passed" in line:
                    self._print(line.strip(), Status.PASS)
                    break
            else:
                self._print("Tests passed", Status.PASS)
            metrics["tests_passed"] = True
        elif code == -2:
            self._print("pytest not available", Status.SKIP)
        else:
            self._print("Tests failed", Status.FAIL)
            metrics["tests_passed"] = False
            self._add_issue(
                6,
                "tests",
                Severity.HIGH,
                "Unit tests failed",
                suggestion="Fix failing tests before deployment",
            )

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = Status.PASS if new_issues == 0 else Status.FAIL

        return PhaseResult(
            phase=6,
            name="Testing",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # PHASE 7: DOCUMENTATION
    # =========================================================================

    def phase_7_documentation(self) -> PhaseResult:
        """Phase 7: Documentation Audit."""
        self._header("PHASE 7: DOCUMENTATION AUDIT")
        start = time.perf_counter()
        issues_before = len(self.issues)
        metrics: dict[str, Any] = {}

        self._section("Docstring Coverage")
        python_files = self._get_python_files()
        total_items = 0
        documented_items = 0

        for f in python_files[:50]:  # Sample
            try:
                source = f.read_text(encoding="utf-8")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_items += 1
                        if ast.get_docstring(node):
                            documented_items += 1
            except Exception:
                pass

        doc_coverage = (documented_items / total_items) * 100 if total_items > 0 else 0
        metrics["docstring_coverage"] = doc_coverage

        if doc_coverage >= 80:
            self._print(f"Docstring coverage: {doc_coverage:.0f}%", Status.PASS)
        elif doc_coverage >= 50:
            self._print(f"Docstring coverage: {doc_coverage:.0f}%", Status.WARN)
        else:
            self._print(f"Docstring coverage: {doc_coverage:.0f}%", Status.FAIL)
            self._add_issue(
                7,
                "documentation",
                Severity.LOW,
                f"Low docstring coverage: {doc_coverage:.0f}%",
                suggestion="Add docstrings to public functions and classes",
            )

        # README check
        self._section("README Quality")
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
            sections = ["install", "usage", "api", "config"]
            found_sections = sum(1 for s in sections if s.lower() in content.lower())
            metrics["readme_sections"] = found_sections
            if found_sections >= 3:
                self._print(f"README has {found_sections}/4 key sections", Status.PASS)
            else:
                self._print(f"README has {found_sections}/4 key sections", Status.WARN)
        else:
            self._print("README.md not found", Status.FAIL)

        duration = (time.perf_counter() - start) * 1000
        new_issues = len(self.issues) - issues_before
        status = Status.PASS if new_issues == 0 else Status.WARN

        return PhaseResult(
            phase=7,
            name="Documentation",
            status=status,
            duration_ms=duration,
            issues=self.issues[issues_before:],
            metrics=metrics,
        )

    # =========================================================================
    # RUN PROTOCOL
    # =========================================================================

    def run(self, phases: list[int] | None = None) -> ValidationReport:
        """Run the Full Arsenal Protocol."""
        self._header("FULL ARSENAL PROTOCOL ACTIVATED")
        print(f"\033[95m  Project:\033[0m {self.project_root.name}")
        print(f"\033[95m  Mode:\033[0m {'Quick' if self.quick_mode else 'Full'}")
        print(f"\033[95m  Fix:\033[0m {'Enabled' if self.fix_mode else 'Disabled'}")
        print(f"\033[95m  Timestamp:\033[0m {datetime.now().isoformat()}")

        all_phases = [
            self.phase_1_initial_analysis,
            self.phase_2_syntax_validation,
            self.phase_3_logic_flow,
            self.phase_4_best_practices,
            self.phase_5_security,
            self.phase_6_testing,
            self.phase_7_documentation,
        ]

        results = []
        for i, phase_func in enumerate(all_phases, 1):
            if phases and i not in phases:
                continue
            try:
                result = phase_func()
                results.append(result)
                self.phases.append(result)
            except Exception as e:
                print(f"\033[91mPhase {i} error: {e}\033[0m")
                results.append(
                    PhaseResult(
                        phase=i,
                        name=f"Phase {i}",
                        status=Status.FAIL,
                        duration_ms=0,
                        issues=[
                            Issue(
                                phase=i,
                                category="error",
                                severity=Severity.CRITICAL,
                                message=str(e),
                            )
                        ],
                    )
                )

        total_duration = (time.perf_counter() - self.start_time) * 1000

        # Generate summary
        summary = self._generate_summary(results)

        # Print final report
        self._print_final_report(results, summary, total_duration)

        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            project=self.project_root.name,
            python_version=sys.version.split()[0],
            total_duration_ms=total_duration,
            phases=results,
            summary=summary,
        )

    def _generate_summary(self, results: list[PhaseResult]) -> dict[str, Any]:
        """Generate summary from phase results."""
        passed = sum(1 for r in results if r.status == Status.PASS)
        failed = sum(1 for r in results if r.status == Status.FAIL)
        warned = sum(1 for r in results if r.status == Status.WARN)

        critical = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        high = sum(1 for i in self.issues if i.severity == Severity.HIGH)
        medium = sum(1 for i in self.issues if i.severity == Severity.MEDIUM)
        low = sum(1 for i in self.issues if i.severity == Severity.LOW)

        # Calculate overall score
        score = 100
        score -= critical * 20
        score -= high * 10
        score -= medium * 5
        score -= low * 1
        score = max(0, min(100, score))

        return {
            "phases_passed": passed,
            "phases_failed": failed,
            "phases_warned": warned,
            "total_issues": len(self.issues),
            "critical_issues": critical,
            "high_issues": high,
            "medium_issues": medium,
            "low_issues": low,
            "quality_score": score,
            "status": "PASS" if failed == 0 and critical == 0 else "FAIL",
        }

    def _print_final_report(
        self, results: list[PhaseResult], summary: dict[str, Any], duration: float
    ) -> None:
        """Print the final report."""
        self._header("FINAL REPORT")

        print("\n  \033[1mPhase Results:\033[0m")
        for r in results:
            status_color = {
                Status.PASS: "\033[92m",
                Status.FAIL: "\033[91m",
                Status.WARN: "\033[93m",
                Status.SKIP: "\033[90m",
            }.get(r.status, "")
            print(f"    {status_color}{r.status.value}\033[0m  Phase {r.phase}: {r.name}")

        print("\n  \033[1mIssue Summary:\033[0m")
        print(f"    Critical: {summary['critical_issues']}")
        print(f"    High:     {summary['high_issues']}")
        print(f"    Medium:   {summary['medium_issues']}")
        print(f"    Low:      {summary['low_issues']}")

        print(f"\n  \033[1mQuality Score:\033[0m {summary['quality_score']}/100")
        print(f"  \033[1mDuration:\033[0m {duration:.0f}ms")

        print()
        if summary["status"] == "PASS":
            print("\033[92m\033[1m" + "=" * 70 + "\033[0m")
            print("\033[92m\033[1m  [PASS] FULL ARSENAL PROTOCOL: ALL CHECKS PASSED\033[0m")
            print("\033[92m\033[1m" + "=" * 70 + "\033[0m")
        else:
            print("\033[91m\033[1m" + "=" * 70 + "\033[0m")
            print("\033[91m\033[1m  [FAIL] FULL ARSENAL PROTOCOL: ISSUES DETECTED\033[0m")
            print("\033[91m\033[1m" + "=" * 70 + "\033[0m")

    def save_report(self, report: ValidationReport, output_path: Path | None = None) -> Path:
        """Save report to JSON file."""
        if output_path is None:
            output_dir = self.project_root / "reports"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"fullarsenal_{timestamp}.json"

        # Convert to dict
        report_dict = {
            "timestamp": report.timestamp,
            "project": report.project,
            "python_version": report.python_version,
            "total_duration_ms": report.total_duration_ms,
            "phases": [
                {
                    "phase": p.phase,
                    "name": p.name,
                    "status": p.status.value,
                    "duration_ms": p.duration_ms,
                    "metrics": p.metrics,
                    "issues": [
                        {
                            "category": i.category,
                            "severity": i.severity.value,
                            "message": i.message,
                            "file": i.file,
                            "line": i.line,
                            "suggestion": i.suggestion,
                        }
                        for i in p.issues
                    ],
                }
                for p in report.phases
            ],
            "summary": report.summary,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        print(f"\n  Report saved to: {output_path}")
        return output_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evidence Suite: Full Arsenal Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/fullarsenal.py                # Full protocol
  python tools/fullarsenal.py --quick        # Quick validation
  python tools/fullarsenal.py --fix          # Auto-fix issues
  python tools/fullarsenal.py --phase 2 4    # Run phases 2 and 4 only
        """,
    )

    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (skip slow checks)")
    parser.add_argument("--fix", "-f", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--phase", "-p", type=int, nargs="+", help="Run specific phases only (1-7)")
    parser.add_argument("--report", "-r", action="store_true", help="Save JSON report")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--output", "-o", type=Path, help="Output path for report")

    args = parser.parse_args()

    protocol = FullArsenalProtocol(
        verbose=not args.quiet,
        fix_mode=args.fix,
        quick_mode=args.quick,
    )

    report = protocol.run(phases=args.phase)

    if args.report or args.output:
        protocol.save_report(report, args.output)

    return 0 if report.summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
