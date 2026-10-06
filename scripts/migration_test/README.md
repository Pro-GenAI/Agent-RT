# Migration test runner

`migration_test.py` consumes `repo_list.csv` and tests the documented one-import-path Agent RT migration against third-party repositories.

For each CSV row, the runner clones the repository, records the tested commit, checks whether the repository is suitable for migration testing, runs its tests before migration, applies the Agent RT import/package rewrite, installs the local Agent RT package, and runs the tests again. Results are stored in `migration_results.csv` with these columns:

`Index, Package, Language, PLIndex, RepoURL, CommitHash, TestStatusBeforeMigration, TestStatusAfterMigration, TestResultsBeforeMigration, TestResultsAfterMigration`

The status columns use values such as `PASS`, `FAIL`, `SKIPPED`, `SETUP_FAILED`, `FILTERED_HEAVY`, and `NO_TESTS`. Detailed result fields contain compact JSON with diagnostic information and migration metadata.

Run this tooling only inside a disposable sandbox or VM. Tested repositories are untrusted, and dependency installation/test commands execute third-party code. The runner reduces credential exposure, but that is not a substitute for OS/container isolation or an explicit network policy.

## Usage

From this directory:

```bash
uv sync

# Run pending repositories sequentially.
uv run python migration_test.py

# Retry repositories whose previous result is retryable.
uv run python migration_test.py --retry-failed

# Rerun only specific repo_list.csv indices (values and ranges).
uv run python migration_test.py --indices 14,21,30-35

# Runs stop cleanly when free disk space drops below --min-free-gb (default 10).
# Stop with Ctrl+C at any time, then rerun the same command with --resume
# (also safe on the first run) to skip repositories it already finished.
uv run python migration_test.py --indices 14,21,30-35 --resume

# Rerun only repositories that passed before migration and failed after it.
uv run python migration_test.py --pass-fail

# Limit or select a range of work.
uv run python migration_test.py --start-index 1 --limit 10

# Run repositories concurrently.
uv run python migration_test.py --workers 8

# Summarize PASS/PASS (before/after migration) counts per package and language.
uv run python summarize_migration_results.py

# Enforce the checked-in uv lock while running.
uv run --frozen python migration_test.py --retry-failed --limit 13
```

Use `--restart` when you intentionally want to truncate `migration_results.csv` before a run.

Each run prints progress and also writes logs under `logs/` (override with `--log-dir`): `run-<timestamp>.log` mirrors the whole run, and `<timestamp>/<index>-<owner>_<repo>.log` contains one repository's full command lines and untruncated output. Start there when investigating a failed row.

Implementation details, setup heuristics, concurrency and cleanup invariants, provider mocking, migration rewrite rules, and maintenance requirements live in [`AGENTS.md`](./AGENTS.md).

## Runner tests

Offline regression tests for the runner's setup heuristics use only the standard library: `python -m unittest test_migration_runner`. They never run third-party code.
