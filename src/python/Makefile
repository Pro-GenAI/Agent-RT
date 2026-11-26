.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --extra dev

.PHONY: help install install-dev test test-chromium test-browsers test-matrix test-deps-matrix lint format format-check typecheck quality security-static security-attacks security vuln check verify require-bash

# GNU Make may use cmd.exe on Windows, so check Bash using native shell syntax.
require-bash:
ifeq ($(OS),Windows_NT)
	@where bash >NUL 2>&1 || (echo ERROR: Bash is required for this target. Install Bash and add it to PATH. & exit /b 1)
else
	@command -v bash >/dev/null 2>&1 || { echo 'ERROR: Bash is required for this target. Install Bash and add it to PATH.' >&2; exit 1; }
endif

help:
	@printf '%s\n' \
		'Agent RT Python developer commands:' \
		'  make install           Install locked development dependencies' \
		'  make install-dev       Alias for make install' \
		'  make test              Run the regression test suite' \
		'  make test-chromium     Run web UI tests in Chromium (AGENT_RT_CHROMIUM_PATH=/path uses your own)' \
		'  make test-browsers     Run web UI tests in every browser this OS can run' \
		'  make test-matrix       Run tests across all supported Python versions' \
		'  make test-deps-matrix  Run tests across supported dependency versions (mid-2024+)' \
		'  make lint              Run Ruff lint checks' \
		'  make format            Format Python code with Black' \
		'  make format-check      Check formatting without changing files' \
		'  make typecheck         Run mypy' \
		'  make quality           Run typecheck + lint + format-check' \
		'  make security-static   Run deterministic static security gate' \
		'  make security-attacks  Run offline adversarial runtime attacks' \
		'  make security          Run the full security/vulnerability gate' \
		'  make vuln              Alias for make security' \
		'  make verify            Run install + test + test-chromium + test-matrix + test-deps-matrix + security, stopping at the first failure' \
		'  make check             Run quality + tests + full security gate'

install:
	$(UV) sync --extra dev --locked

install-dev: install

test:
	$(RUN) pytest -n auto --dist worksteal

# Playwright is added for this run only and its browsers go to a temp folder, not the system.
# BROWSER_ARGS passes runner options, e.g. BROWSER_ARGS="--browsers-dir ~/agent-rt-browsers".
# Pinned: each Playwright release needs its own browser builds, and CircleCI caches them by this file.
PLAYWRIGHT_VERSION ?= 1.64.0
test-browsers:
	$(RUN) --with "playwright==$(PLAYWRIGHT_VERSION)" python scripts/test_browsers.py $(BROWSER_ARGS)

test-chromium:
	$(RUN) --with "playwright==$(PLAYWRIGHT_VERSION)" python scripts/test_browsers.py --engine chromium $(BROWSER_ARGS)

test-matrix: require-bash
	bash ./scripts/test-matrix.sh

test-deps-matrix: require-bash
	bash ./scripts/test-deps-matrix.sh

lint:
	$(RUN) ruff check src tests scripts

format:
	$(RUN) black src tests scripts

format-check:
	$(RUN) black --check src tests scripts

typecheck:
	$(RUN) mypy src

quality: typecheck lint format-check

security-static:
	$(RUN) python scripts/security_scan.py --fail-on medium

security-attacks:
	$(RUN) python scripts/security_attack_scan.py

security: require-bash
	@bash ../scripts/security-summary.sh \
		'Static scan' '$(RUN) python scripts/security_scan.py --fail-on medium' \
		'Adversarial runtime scan' '$(RUN) python scripts/security_attack_scan.py' \
		'Development security tools' '$(UV) run --python 3.12 --extra dev python scripts/dev_security_scan.py'

vuln: security

check: quality test security

# `make verify` runs without conda: a child process cannot deactivate the caller's shell,
# so drop conda's variables and its PATH entries (base and every env) for these steps.
ifneq ($(CONDA_EXE),)
CONDA_ROOT := $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(CONDA_EXE)))))
verify: export PATH := $(shell printf '%s' "$$PATH" | tr ':' '\n' | grep -v '^$(CONDA_ROOT)/' | paste -sd: -)
NO_CONDA := env -u CONDA_EXE -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PYTHON_EXE -u CONDA_PROMPT_MODIFIER -u _CE_CONDA -u _CE_M
endif

# Sequential recipe lines: make stops at the first failing step.
verify:
	@$(NO_CONDA) $(MAKE) install
	@$(NO_CONDA) $(MAKE) test
	@$(NO_CONDA) $(MAKE) test-chromium
	@$(NO_CONDA) $(MAKE) test-matrix
# 	@$(NO_CONDA) $(MAKE) test-deps-matrix
	@$(NO_CONDA) $(MAKE) security
