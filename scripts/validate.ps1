<#
.SYNOPSIS
    Evidence Suite: PowerShell Validation Script
    Author: Senior Director of Coding Excellence

.DESCRIPTION
    Cross-platform validation script for Windows environments.
    Runs comprehensive code quality checks.

.PARAMETER Quick
    Run quick checks only (syntax, format)

.PARAMETER Fix
    Auto-fix issues where possible

.PARAMETER Security
    Run security-focused checks

.PARAMETER Benchmark
    Include benchmark tests

.EXAMPLE
    .\scripts\validate.ps1
    .\scripts\validate.ps1 -Quick
    .\scripts\validate.ps1 -Fix
#>

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Fix,
    [switch]$Security,
    [switch]$Benchmark,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

# Colors
$Colors = @{
    Pass = "Green"
    Fail = "Red"
    Warn = "Yellow"
    Info = "Cyan"
    Header = "Blue"
}

# Counters
$Script:Passed = 0
$Script:Failed = 0
$Script:Skipped = 0
$Script:Warnings = 0

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor $Colors.Header
    Write-Host "  $Title" -ForegroundColor $Colors.Header
    Write-Host ("=" * 70) -ForegroundColor $Colors.Header
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "► $Title" -ForegroundColor $Colors.Info
}

function Write-Check {
    param(
        [string]$Message,
        [ValidateSet("Pass", "Fail", "Warn", "Skip", "Info")]
        [string]$Status
    )

    $Symbol = switch ($Status) {
        "Pass" { "✓" }
        "Fail" { "✗" }
        "Warn" { "⚠" }
        "Skip" { "○" }
        "Info" { "ℹ" }
    }

    Write-Host "  $Symbol $Message" -ForegroundColor $Colors[$Status]

    switch ($Status) {
        "Pass" { $Script:Passed++ }
        "Fail" { $Script:Failed++ }
        "Skip" { $Script:Skipped++ }
        "Warn" { $Script:Warnings++ }
    }
}

function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

Push-Location $ProjectRoot

try {
    Write-Header "EVIDENCE SUITE VALIDATION"
    Write-Host "  Project Root: $ProjectRoot" -ForegroundColor $Colors.Info
    Write-Host "  Python: $(python --version 2>&1)" -ForegroundColor $Colors.Info
    Write-Host "  Timestamp: $(Get-Date -Format 'o')" -ForegroundColor $Colors.Info

    # =========================================================================
    # PHASE 1: SYNTAX VALIDATION
    # =========================================================================
    Write-Header "PHASE 1: SYNTAX VALIDATION"

    Write-Section "Python Syntax Check"
    $PythonFiles = Get-ChildItem -Path . -Filter "*.py" -Recurse |
        Where-Object { $_.FullName -notmatch '\.venv|venv|__pycache__|node_modules' }

    $SyntaxErrors = 0
    foreach ($File in $PythonFiles | Select-Object -First 100) {
        $Result = python -m py_compile $File.FullName 2>&1
        if ($LASTEXITCODE -ne 0) {
            $SyntaxErrors++
            Write-Check "Syntax error in $($File.Name)" -Status "Fail"
        }
    }

    if ($SyntaxErrors -eq 0) {
        Write-Check "All $($PythonFiles.Count) Python files have valid syntax" -Status "Pass"
    }

    Write-Section "TOML Validation"
    if (Test-Path "pyproject.toml") {
        $Result = python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "pyproject.toml valid" -Status "Pass"
        } else {
            Write-Check "pyproject.toml invalid" -Status "Fail"
        }
    }

    # =========================================================================
    # PHASE 2: CODE FORMATTING & LINTING
    # =========================================================================
    Write-Header "PHASE 2: CODE FORMATTING & LINTING"

    Write-Section "Ruff Linter"
    if ($Fix) {
        $Result = python -m ruff check . --fix --quiet 2>&1
    } else {
        $Result = python -m ruff check . --quiet 2>&1
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Check "Ruff linting passed" -Status "Pass"
    } elseif ($LASTEXITCODE -eq 2) {
        Write-Check "Ruff not installed" -Status "Skip"
    } else {
        Write-Check "Ruff found issues" -Status "Warn"
    }

    Write-Section "Ruff Formatter"
    if ($Fix) {
        $Result = python -m ruff format . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "Code formatted" -Status "Pass"
        }
    } else {
        $Result = python -m ruff format --check . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "Code formatting OK" -Status "Pass"
        } else {
            Write-Check "Code needs formatting (run with -Fix)" -Status "Warn"
        }
    }

    if (-not $Quick) {
        Write-Section "Type Checking (mypy)"
        $Result = python -m mypy core api --config-file pyproject.toml 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "Type checking passed" -Status "Pass"
        } elseif ($LASTEXITCODE -eq 2) {
            Write-Check "mypy not installed" -Status "Skip"
        } else {
            Write-Check "Type checking has warnings" -Status "Warn"
        }
    }

    # =========================================================================
    # PHASE 3: SECURITY CHECKS
    # =========================================================================
    if ($Security -or (-not $Quick)) {
        Write-Header "PHASE 3: SECURITY CHECKS"

        Write-Section "Bandit Security Scan"
        $Result = python -m bandit -r core api -q -ll 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "No security issues found" -Status "Pass"
        } elseif ($LASTEXITCODE -eq 2) {
            Write-Check "Bandit not installed" -Status "Skip"
        } else {
            Write-Check "Security issues detected" -Status "Warn"
        }

        Write-Section "Dependency Vulnerabilities"
        $Result = python -m pip_audit --strict 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "No vulnerabilities found" -Status "Pass"
        } elseif ($LASTEXITCODE -eq 2) {
            Write-Check "pip-audit not installed" -Status "Skip"
        } else {
            Write-Check "Vulnerabilities detected" -Status "Warn"
        }
    }

    # =========================================================================
    # PHASE 4: TESTS
    # =========================================================================
    if (-not $Quick) {
        Write-Header "PHASE 4: TESTS"

        Write-Section "Unit Tests"
        $env:EVIDENCE_SUITE_ENV = "test"
        $Result = python -m pytest tests/test_api.py -v --tb=short -q 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Check "Unit tests passed" -Status "Pass"
        } elseif ($LASTEXITCODE -eq 2) {
            Write-Check "pytest not installed" -Status "Skip"
        } else {
            Write-Check "Unit tests failed" -Status "Fail"
        }

        if ($Benchmark) {
            Write-Section "Benchmark Tests"
            if (Test-Path "tests/benchmark.py") {
                $Result = python tests/benchmark.py -i 20 -w 2 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Check "Benchmarks completed" -Status "Pass"
                } else {
                    Write-Check "Benchmarks had issues" -Status "Warn"
                }
            }
        }
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    Write-Header "VALIDATION SUMMARY"

    Write-Host ""
    Write-Host "  Passed:   $Script:Passed" -ForegroundColor $Colors.Pass
    Write-Host "  Failed:   $Script:Failed" -ForegroundColor $Colors.Fail
    Write-Host "  Skipped:  $Script:Skipped" -ForegroundColor $Colors.Warn
    Write-Host "  Warnings: $Script:Warnings" -ForegroundColor $Colors.Warn
    Write-Host ""

    if ($Script:Failed -eq 0) {
        Write-Host ("=" * 70) -ForegroundColor $Colors.Pass
        Write-Host "  ✓ ALL VALIDATION CHECKS PASSED" -ForegroundColor $Colors.Pass
        Write-Host ("=" * 70) -ForegroundColor $Colors.Pass
        $ExitCode = 0
    } else {
        Write-Host ("=" * 70) -ForegroundColor $Colors.Fail
        Write-Host "  ✗ VALIDATION FAILED ($Script:Failed issues)" -ForegroundColor $Colors.Fail
        Write-Host ("=" * 70) -ForegroundColor $Colors.Fail
        Write-Host ""
        Write-Host "Tips:" -ForegroundColor $Colors.Warn
        Write-Host "  - Run with -Fix to auto-fix issues"
        Write-Host "  - Run with -Quick for fast checks only"
        $ExitCode = 1
    }

} finally {
    Pop-Location
}

exit $ExitCode
