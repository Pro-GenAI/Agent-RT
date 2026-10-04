.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --extra dev

.PHONY: help install install-dev test test-matrix lint format format-check typecheck quality security-static security-attacks security vuln check

help:
	@printf '%s\n' \
		'Agent RT Python developer commands:' \
		'  make install           Install locked development dependencies' \
		'  make install-dev       Alias for make install' \
		'  make test              Run the regression test suite' \
		'  make test-matrix       Run tests across all supported Python versions' \
		'  make lint              Run Ruff lint checks' \
		'  make format            Format Python code with Black' \
		'  make format-check      Check formatting without changing files' \
		'  make typecheck         Run mypy' \
		'  make quality           Run typecheck + lint + format-check' \
		'  make security-static   Run deterministic static security gate' \
		'  make security-attacks  Run offline adversarial runtime attacks' \
		'  make security          Run the full security/vulnerability gate' \
		'  make vuln              Alias for make security' \
		'  make check             Run quality + tests + full security gate'

install:
	$(UV) sync --extra dev --locked

install-dev: install

test:
	$(RUN) pytest -n auto --dist worksteal

test-matrix:
	./scripts/test-matrix.sh

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

security: security-static security-attacks
	$(UV) run --python 3.12 --extra dev python scripts/dev_security_scan.py

vuln: security

check: quality test security
