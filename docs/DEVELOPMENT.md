# Development

This guide contains contributor workflows for the Python and TypeScript implementations of Agent RT. Package READMEs are intentionally focused on end users.

## Python

### Setup and tests

Python sources live in `python/src/`, tests in `python/tests/`, and repository utilities in `python/scripts/`. Python 3.10 through 3.14 are supported.

The preferred workflow uses the locked `uv` environment:

```bash
cd python
uv sync --extra dev --locked
make test
```

`make test` runs pytest with pytest-xdist using an automatic worker count and work-stealing distribution:

```bash
uv run --extra dev pytest -n auto --dist worksteal
```

Equivalent editable `pip` workflow:

```bash
cd python
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

The Python Makefile exposes the routine contributor commands:

```bash
make help
make install
make test
make test-matrix
make lint
make format
make format-check
make typecheck
make quality
make security-static
make security-attacks
make security
make check
```

| Target | What it does |
| --- | --- |
| `make install` | Install locked development dependencies (`uv sync --extra dev --locked`). |
| `make test` | Run the regression suite in parallel. |
| `make test-matrix` | Run the suite across all supported Python versions. |
| `make lint` / `make format` / `make format-check` | Ruff lint; Black formatting; Black check without changes. |
| `make typecheck` | Run mypy on `src`. |
| `make quality` | `typecheck` + `lint` + `format-check`. |
| `make security-static` | Deterministic static security gate (fails on medium or higher). |
| `make security-attacks` | Offline adversarial runtime scan. |
| `make security` | Both gates plus the third-party developer security scan; runs all checks and prints a final PASS/FAIL summary (failures appear in red on color terminals). |
| `make check` | `quality` + `test` + `security`. |

To run a single test file or test: `uv run --extra dev pytest tests/test_agent_loop.py -k name`.

Run the same regression suite across all supported Python versions:

```bash
cd python
./scripts/test-matrix.sh
```

Override the matrix when needed:

```bash
PYTHON_VERSIONS="3.13 3.14" ./scripts/test-matrix.sh
```

The matrix uses `uv`, ignores inherited Conda activation variables, and creates isolated locked environments from `uv.lock`. Both `make test-matrix` and `make security` require Bash on `PATH`; on Windows without Bash, the Make targets stop immediately with an actionable error.

The regression suite is pytest-native. Async tests use pytest-asyncio in auto mode with function-scoped test and fixture loops. Vector database integration coverage runs entirely on localhost through `tests/helpers/vector_mock.py` and `tests/test_vector_mock_integration.py`, covering Chroma, Milvus, Pinecone, Qdrant, and Weaviate through core Agent RT plus LangChain and LlamaIndex compatibility paths.

### Python security tooling

Run the repository security gates with:

```bash
uv run --extra dev python scripts/security_scan.py
uv run --extra dev python scripts/security_scan.py --json
uv run --extra dev python scripts/security_scan.py --fail-on medium
uv run --extra dev python scripts/security_attack_scan.py
uv run --extra dev python scripts/security_attack_scan.py --json
uv run --extra dev python scripts/dev_security_scan.py
```

With the editable `pip` workflow, run the same scripts directly with Python. `./scripts/dev_security_scan.py` is also supported and prefers tools installed in the project `.venv` before the launching interpreter or global `PATH`.

Use `--category code`, `--category agent-llm`, or `--category api-server` for focused static scans. `--include-tests` extends generic code scanning to tests and scripts.

The deterministic repository gates do not call a real LLM or external model API. `security_attack_scan.py` uses a malicious provider double to exercise prompt-injection-driven hidden tool calls, malformed arguments, permission/approval bypass attempts, restricted-data exfiltration, and denial-of-wallet behavior against real runtime paths.

The `dev` extra includes the `all` runtime integrations plus Ruff, Black, mypy, pytest/coverage, pre-commit/build/publishing tools, Bandit, pip-audit, detect-secrets, CycloneDX tooling, AgentSec, PromptFuzz, Fickling, PickleScan, and ModelScan where its Python-version constraints allow it.

Complementary tools can be run directly:

```bash
uv run --extra dev bandit -r src
uv run --extra dev pip-audit
uv run --extra dev detect-secrets scan
uv run --extra dev agentsec discover
uv run --extra dev agentsec scan <agent-installation>
uv run --extra dev promptfuzz list-attacks
uv run --extra dev fickling --check-safety <model-or-pickle-file>
uv run --extra dev picklescan --path <model-or-pickle-file>
uv run --extra dev modelscan --path <model-or-pickle-file>
uv run --extra dev cyclonedx-py environment -o sbom.json
```

PromptFuzz is used for its local corpus and local-target workflows. AgentSec inspects installed agent/MCP configurations. Fickling, PickleScan, and ModelScan inspect local serialized model artifacts.

### Python evaluation

HarmActionsEval is the default model-backed benchmark. It calls a real model through your OpenAI-compatible endpoint, so it incurs API cost and is non-deterministic; it is not part of the deterministic security gates or CI.

```bash
cd python
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="..."
# Optional for OpenAI-compatible providers:
export OPENAI_BASE_URL="https://your-provider.example/v1"

python scripts/evaluate_harness.py --scope benchmark
python scripts/evaluate_harness.py --scope benchmark --k 3 --limit 25
python scripts/evaluate_harness.py --scope benchmark --json
```

For offline regression and latency work:

```bash
python scripts/evaluate_harness.py --scope evals
python scripts/evaluate_harness.py --scope tests
python scripts/evaluate_harness.py --scope benchmark --benchmark performance --iterations 500
```

Scopes are `evals`, `tests`, `benchmark`, and `all`. The default benchmark is `harmactions`; `--benchmark performance` selects the local ToolRegistry dispatch microbenchmark. HarmActionsEval supports `--k`, `--offset`, `--limit`, `--cache-path`, `--output`, and `--json`.

### Python packaging

Package metadata is defined in `python/pyproject.toml`. Validate the publish path without uploading:

```bash
cd python
./pip_publish.sh --dry-run
```

`pip_publish.sh` runs the regression suite before building and validating the distribution.

## TypeScript

### Setup and tests

TypeScript sources live in `typescript/src/` (core in `index.ts`; companion contracts, framework adapters, and transports under `src/ext/`, mirroring Python's `ext` package), tests in `typescript/tests/`, and evaluation/publishing utilities in `typescript/scripts/`. Node 20, 22, 24, and 26 are supported.

Install reproducibly and run the runtime suite:

```bash
cd typescript
npm ci
npm test
```

The TypeScript Makefile exposes the same target names as Python (`make help` lists them), backed by npm scripts: `make quality` runs `typecheck`, `lint`, `format-check`, and dead-code detection, and `make security` runs the equivalent security checks individually, prints a final PASS/FAIL summary, and highlights failures in red on color terminals.

Run the suite across supported Node versions with `nvm`:

```bash
npm run test:matrix
```

The default matrix is Node 20, 22, 24, and 26. The script supports Unix `nvm` and `nvm-windows` from Git Bash. Each matrix install uses `npm ci --omit=optional --ignore-scripts`.

Override the matrix when needed:

```bash
NODE_VERSIONS="22 24 26" npm run test:matrix
```

Run code-quality checks with:

```bash
npm run typecheck
npm run lint
npm run quality:dead-code
npm run quality:check
npm run format
npm run format:check
```

Formatting is intentionally separate from `quality:check`.

The runtime tests use Node's built-in test runner. OpenAI HTTP/SSE behavior, embeddings, batches, model listing, error injection, and vector protocol coverage use `@copilotkit/aimock` where possible. `tests/vector-mock.js` extends aimock for Agent RT-specific Chroma v2, Milvus, and Weaviate query routes. The vector integration suite covers all five built-in vector backends without live services.

### TypeScript security tooling

Run security checks with:

```bash
npm run security:scan
npm run security:scan -- --json
npm run security:scan:gate
npm run security:attacks
npm run security:deps
npm run security:secrets
npm run security:mcp
npm run security:ai:static
npm run security:sbom
npm run security:gate
```

Use `--category code`, `--category agent-llm`, or `--category api-server` for focused built-in scans. `security:scan:gate` fails on medium-or-higher findings. `security:attacks` runs a deterministic malicious-model emulator. `security:deps` gates npm advisories, `security:secrets` scans for credentials, `security:mcp` covers MCP-specific issues, and `security:sbom` generates and validates `sbom.cdx.json`.

The deterministic security gates do not call a real LLM or external model API. `security:ai:static` is heuristic and review-oriented rather than part of the deterministic gate.

Development-only quality and security packages stay in `devDependencies`, not runtime dependencies or optional provider peers.

### TypeScript evaluation

The TypeScript benchmark runner delegates HarmActionsEval to the sibling Python runner so both runtimes use the same upstream benchmark and metrics:

```bash
cd typescript
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="..."
# Optional:
export OPENAI_BASE_URL="https://your-provider.example/v1"

npm run evaluate:benchmark
npm run evaluate:benchmark -- --k 3 --limit 25
npm run evaluate:benchmark -- --json
```

Offline commands:

```bash
npm run evaluate:evals
npm run evaluate:tests
npm run evaluate:performance -- --iterations 500
```

The underlying runner accepts `--scope evals|tests|benchmark|all`, `--benchmark harmactions|performance`, `--k`, `--offset`, `--limit`, `--cache-path`, `--output`, `--iterations`, and `--json`. Set `PYTHON` to override the interpreter used for HarmActionsEval.

### TypeScript packaging

Build output is written to `typescript/dist/`, with `dist/index.js` and `dist/index.d.ts` as package entry points.

Preview the package contents:

```bash
cd typescript
npm pack --dry-run
```

`scripts/npm_publish.sh` runs `npm ci`, the regression suite, and a package preview before publishing.

## Code conventions

- Keep Python and TypeScript semantics aligned and add focused tests in both languages when adding a runtime contract (see "Contract evolution and migration" in `Architecture-and-Extensions.md`).
- Provider payload builders may cache conversions only with strong references to the cached inputs.
- Define each helper name once per module; in Python and JavaScript a later definition silently replaces an earlier one.
- Keep deterministic gates offline: tests, security scans, and adversarial scans must not call a real LLM or external model API.
- Keep documentation current: `README.md` for user-facing capabilities, `docs/` for durable architecture, and `docs/MIGRATION.md` when a compatibility surface changes.

## Cross-platform CI

Cross-platform CI lives in `.github/workflows/cross-platform.yml` and keeps both runtimes covered on Linux, macOS, and Windows.

Both publish workflows depend on the cross-platform workflow so publishing does not proceed before the cross-platform release checks finish successfully.

## CircleCI on Apple Silicon macOS

CircleCI configuration lives in `.circleci/config.yml` and intentionally uses the macOS `m4pro.medium` resource class. Every job verifies `uname -m` is `arm64`. The workflow is manual-only through the `run_mac_ci` pipeline parameter.

The CircleCI workflow runs:

- Python regression tests across Python 3.10-3.14.
- Python built-in static security scanning, deterministic offline adversarial scanning, and the third-party developer security gate.
- TypeScript regression tests across Node 20, 22, 24, and 26.
- TypeScript type/lint/dead-code checks plus the deterministic security gate.

All deterministic security gates remain offline with respect to real LLM/model endpoints.
