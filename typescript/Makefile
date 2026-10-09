.DEFAULT_GOAL := help

NPM ?= npm

.PHONY: help install install-dev test test-matrix lint format format-check typecheck quality security-static security-attacks security vuln check require-bash

# GNU Make may use cmd.exe on Windows, so check Bash using native shell syntax.
require-bash:
ifeq ($(OS),Windows_NT)
	@where bash >NUL 2>&1 || (echo ERROR: Bash is required for this target. Install Bash and add it to PATH. & exit /b 1)
else
	@command -v bash >/dev/null 2>&1 || { echo 'ERROR: Bash is required for this target. Install Bash and add it to PATH.' >&2; exit 1; }
endif

help:
	@printf '%s\n' \
		'Agent RT TypeScript developer commands:' \
		'  make install           Install locked development dependencies' \
		'  make install-dev       Alias for make install' \
		'  make test              Run the regression test suite' \
		'  make test-matrix       Run tests across all supported Node versions' \
		'  make lint              Run ESLint checks' \
		'  make format            Format source/scripts with Prettier' \
		'  make format-check      Check formatting without changing files' \
		'  make typecheck         Run TypeScript type checking' \
		'  make quality           Run typecheck + lint + format-check/dead-code checks' \
		'  make security-static   Run deterministic static security gate' \
		'  make security-attacks  Run offline adversarial runtime attacks' \
		'  make security          Run the full security/vulnerability gate' \
		'  make vuln              Alias for make security' \
		'  make check             Run quality + tests + full security gate'

install:
	$(NPM) ci

install-dev: install

test:
	$(NPM) test

test-matrix: require-bash
	$(NPM) run test:matrix

lint:
	$(NPM) run lint

format:
	$(NPM) run format

format-check:
	$(NPM) run format:check

typecheck:
	$(NPM) run typecheck

quality: typecheck lint format-check
	$(NPM) run quality:dead-code

security-static:
	$(NPM) run security:scan:gate

security-attacks:
	$(NPM) run security:attacks

security: require-bash
	@bash ../scripts/security-summary.sh \
		'Static scan' '$(NPM) run security:scan:gate' \
		'Adversarial runtime scan' '$(NPM) run security:attacks' \
		'Dependency audit' '$(NPM) run security:deps' \
		'Secret scan' '$(NPM) run security:secrets' \
		'MCP scan' '$(NPM) run security:mcp'

vuln: security

check: quality test security
