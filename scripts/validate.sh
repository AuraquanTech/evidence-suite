#!/usr/bin/env bash
# =============================================================================
# Evidence Suite: Comprehensive Validation Script
# Author: Senior Director of Coding Excellence
#
# Usage:
#   ./scripts/validate.sh              # Full validation
#   ./scripts/validate.sh --quick      # Quick checks only (syntax, format)
#   ./scripts/validate.sh --ci         # CI mode (strict, no fixes)
#   ./scripts/validate.sh --fix        # Auto-fix issues
#   ./scripts/validate.sh --security   # Security-focused checks
#   ./scripts/validate.sh --benchmark  # Include benchmark tests
#
# Exit Codes:
#   0 - All checks passed
#   1 - Validation failed
#   2 - Missing dependencies
# =============================================================================

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color
readonly BOLD='\033[1m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUICK_MODE=false
CI_MODE=false
FIX_MODE=false
SECURITY_MODE=false
BENCHMARK_MODE=false
VERBOSE=false

# Counters
PASSED=0
FAILED=0
SKIPPED=0
WARNINGS=0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_header() {
  echo ""
  echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${BLUE}  $1${NC}"
  echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

log_section() {
  echo ""
  echo -e "${CYAN}▶ $1${NC}"
}

log_pass() {
  echo -e "  ${GREEN}✓${NC} $1"
  ((PASSED++))
}

log_fail() {
  echo -e "  ${RED}✗${NC} $1"
  ((FAILED++))
}

log_skip() {
  echo -e "  ${YELLOW}○${NC} $1 (skipped)"
  ((SKIPPED++))
}

log_warn() {
  echo -e "  ${YELLOW}⚠${NC} $1"
  ((WARNINGS++))
}

log_info() {
  echo -e "  ${PURPLE}ℹ${NC} $1"
}

check_command() {
  command -v "$1" >/dev/null 2>&1
}

run_check() {
  local name="$1"
  local cmd="$2"
  local fix_cmd="${3:-}"

  if $FIX_MODE && [[ -n "$fix_cmd" ]]; then
    cmd="$fix_cmd"
  fi

  if eval "$cmd" >/dev/null 2>&1; then
    log_pass "$name"
    return 0
  else
    if $CI_MODE; then
      log_fail "$name"
      return 1
    else
      log_fail "$name"
      if $VERBOSE; then
        eval "$cmd" 2>&1 | head -20 || true
      fi
      return 1
    fi
  fi
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
  case $1 in
  --quick | -q)
    QUICK_MODE=true
    shift
    ;;
  --ci)
    CI_MODE=true
    shift
    ;;
  --fix | -f)
    FIX_MODE=true
    shift
    ;;
  --security | -s)
    SECURITY_MODE=true
    shift
    ;;
  --benchmark | -b)
    BENCHMARK_MODE=true
    shift
    ;;
  --verbose | -v)
    VERBOSE=true
    shift
    ;;
  --help | -h)
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -q, --quick      Quick checks only (syntax, format)"
    echo "  -f, --fix        Auto-fix issues where possible"
    echo "  --ci             CI mode (strict, no fixes)"
    echo "  -s, --security   Security-focused checks"
    echo "  -b, --benchmark  Include benchmark tests"
    echo "  -v, --verbose    Verbose output"
    echo "  -h, --help       Show this help"
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    exit 2
    ;;
  esac
done

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

cd "$PROJECT_ROOT"

log_header "EVIDENCE SUITE VALIDATION"
echo -e "${PURPLE}Project Root:${NC} $PROJECT_ROOT"
echo -e "${PURPLE}Mode:${NC} $(
  $QUICK_MODE && echo "Quick"
  $CI_MODE && echo "CI"
  $FIX_MODE && echo "Fix"
  $SECURITY_MODE && echo "Security"
  ! $QUICK_MODE && ! $CI_MODE && ! $FIX_MODE && ! $SECURITY_MODE && echo "Full"
)"
echo -e "${PURPLE}Python:${NC} $(python --version 2>/dev/null || echo 'Not found')"
echo -e "${PURPLE}Timestamp:${NC} $(date -Iseconds)"

# =============================================================================
# PHASE 1: SYNTAX VALIDATION
# =============================================================================

log_header "PHASE 1: SYNTAX VALIDATION"

log_section "Python Syntax Check"
python_files=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./__pycache__/*" 2>/dev/null | head -100)
syntax_errors=0
for file in $python_files; do
  if ! python -m py_compile "$file" 2>/dev/null; then
    log_fail "Syntax error in $file"
    ((syntax_errors++))
  fi
done
if [[ $syntax_errors -eq 0 ]]; then
  log_pass "All Python files have valid syntax"
fi

log_section "YAML Validation"
if check_command yamllint; then
  if yamllint -d relaxed . 2>/dev/null | grep -q "error"; then
    log_fail "YAML validation"
  else
    log_pass "YAML files valid"
  fi
else
  for yaml_file in $(find . -name "*.yml" -o -name "*.yaml" 2>/dev/null | grep -v node_modules | head -20); do
    if python -c "import yaml; yaml.safe_load(open('$yaml_file'))" 2>/dev/null; then
      :
    else
      log_fail "Invalid YAML: $yaml_file"
    fi
  done
  log_pass "YAML files parsed successfully"
fi

log_section "JSON Validation"
json_valid=true
for json_file in $(find . -name "*.json" -not -path "./node_modules/*" -not -path "./.venv/*" 2>/dev/null | head -20); do
  if ! python -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
    log_fail "Invalid JSON: $json_file"
    json_valid=false
  fi
done
$json_valid && log_pass "All JSON files valid"

log_section "TOML Validation"
if [[ -f pyproject.toml ]]; then
  if python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null; then
    log_pass "pyproject.toml valid"
  else
    log_fail "pyproject.toml invalid"
  fi
fi

# =============================================================================
# PHASE 2: CODE FORMATTING & LINTING
# =============================================================================

log_header "PHASE 2: CODE FORMATTING & LINTING"

log_section "Ruff Linter"
if check_command ruff; then
  if $FIX_MODE; then
    ruff check . --fix --quiet 2>/dev/null && log_pass "Ruff linting (with fixes)" || log_fail "Ruff linting"
  else
    ruff check . --quiet 2>/dev/null && log_pass "Ruff linting" || log_fail "Ruff linting"
  fi
else
  log_skip "Ruff not installed"
fi

log_section "Ruff Formatter"
if check_command ruff; then
  if $FIX_MODE; then
    ruff format . 2>/dev/null && log_pass "Ruff formatting (applied)" || log_fail "Ruff formatting"
  else
    ruff format --check . 2>/dev/null && log_pass "Ruff formatting" || log_fail "Ruff formatting (run with --fix)"
  fi
else
  log_skip "Ruff not installed"
fi

if ! $QUICK_MODE; then
  log_section "Type Checking (mypy)"
  if check_command mypy; then
    if mypy core api --config-file pyproject.toml --no-error-summary 2>/dev/null; then
      log_pass "Type checking"
    else
      log_warn "Type checking has issues (non-blocking)"
    fi
  else
    log_skip "mypy not installed"
  fi
fi

# =============================================================================
# PHASE 3: SECURITY CHECKS
# =============================================================================

if $SECURITY_MODE || ! $QUICK_MODE; then
  log_header "PHASE 3: SECURITY CHECKS"

  log_section "Bandit Security Scan"
  if check_command bandit; then
    if bandit -r core api -q -ll 2>/dev/null; then
      log_pass "Bandit security scan"
    else
      log_warn "Bandit found potential issues"
    fi
  else
    log_skip "Bandit not installed"
  fi

  log_section "Dependency Vulnerabilities"
  if check_command pip-audit; then
    if pip-audit --strict --progress-spinner off 2>/dev/null; then
      log_pass "No known vulnerabilities"
    else
      log_warn "Dependency vulnerabilities detected"
    fi
  elif check_command safety; then
    if safety check --full-report 2>/dev/null; then
      log_pass "No known vulnerabilities"
    else
      log_warn "Dependency vulnerabilities detected"
    fi
  else
    log_skip "pip-audit/safety not installed"
  fi

  log_section "Secrets Detection"
  if check_command detect-secrets; then
    if detect-secrets scan --baseline .secrets.baseline 2>/dev/null | grep -q '"results": {}'; then
      log_pass "No secrets detected"
    else
      log_warn "Potential secrets found"
    fi
  else
    # Simple grep-based check
    if grep -rE "(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]" core api --include="*.py" 2>/dev/null | grep -v "example\|test\|TODO" | head -1; then
      log_warn "Potential hardcoded secrets found"
    else
      log_pass "Basic secrets check"
    fi
  fi
fi

# =============================================================================
# PHASE 4: TESTS
# =============================================================================

if ! $QUICK_MODE; then
  log_header "PHASE 4: TESTS"

  log_section "Unit Tests"
  if check_command pytest; then
    if pytest tests/test_api.py -v --tb=short -q 2>/dev/null; then
      log_pass "Unit tests"
    else
      log_fail "Unit tests"
    fi
  else
    log_skip "pytest not installed"
  fi

  if $BENCHMARK_MODE; then
    log_section "Benchmark Tests"
    if [[ -f tests/benchmark.py ]]; then
      if python tests/benchmark.py -i 20 -w 2 2>/dev/null; then
        log_pass "Benchmark tests"
      else
        log_warn "Benchmark tests had issues"
      fi
    else
      log_skip "Benchmark tests not found"
    fi
  fi
fi

# =============================================================================
# PHASE 5: DOCUMENTATION & STRUCTURE
# =============================================================================

if ! $QUICK_MODE; then
  log_header "PHASE 5: DOCUMENTATION & STRUCTURE"

  log_section "Required Files"
  required_files=("README.md" "requirements.txt" "pyproject.toml")
  for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
      log_pass "$file exists"
    else
      log_warn "$file missing"
    fi
  done

  log_section "Project Structure"
  required_dirs=("core" "api" "tests" "scripts")
  for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
      log_pass "$dir/ directory exists"
    else
      log_warn "$dir/ directory missing"
    fi
  done
fi

# =============================================================================
# PHASE 6: PRE-COMMIT HOOKS
# =============================================================================

if ! $QUICK_MODE && ! $CI_MODE; then
  log_header "PHASE 6: PRE-COMMIT HOOKS"

  log_section "Pre-commit Installation"
  if check_command pre-commit; then
    if [[ -f .git/hooks/pre-commit ]]; then
      log_pass "Pre-commit hooks installed"
    else
      log_warn "Pre-commit hooks not installed (run: pre-commit install)"
    fi
  else
    log_skip "pre-commit not installed"
  fi
fi

# =============================================================================
# SUMMARY
# =============================================================================

log_header "VALIDATION SUMMARY"

total=$((PASSED + FAILED + SKIPPED))
echo ""
echo -e "  ${GREEN}Passed:${NC}   $PASSED"
echo -e "  ${RED}Failed:${NC}   $FAILED"
echo -e "  ${YELLOW}Skipped:${NC}  $SKIPPED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "  ${BLUE}Total:${NC}    $total"
echo ""

if [[ $FAILED -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
  echo -e "${GREEN}${BOLD}  ✓ ALL VALIDATION CHECKS PASSED${NC}"
  echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
  exit 0
else
  echo -e "${RED}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
  echo -e "${RED}${BOLD}  ✗ VALIDATION FAILED ($FAILED issues)${NC}"
  echo -e "${RED}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "${YELLOW}Tips:${NC}"
  echo "  - Run with --fix to auto-fix issues"
  echo "  - Run with --verbose for detailed output"
  echo "  - Run with --quick for fast checks only"
  exit 1
fi
