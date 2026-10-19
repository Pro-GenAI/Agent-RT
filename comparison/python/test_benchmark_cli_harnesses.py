"""Regression tests for the isolated agent CLI benchmark (no external CLI required)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_cli_harnesses as cli


def args(**overrides):
    return argparse.Namespace(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0,
                              timeout=2.0, **overrides)


def row(name, **overrides):
    record = {"harness": name, "runs": 30, "failures": 0}
    for level in cli.DEFAULT_CONCURRENCY:
        record.update({f"c{level}_failures": 0, f"c{level}_batch_peak_rss_mib": 32.0,
                       f"c{level}_model_requests": level, f"c{level}_expected_min_requests": level})
    record.update(overrides)
    return record


def complete_rows():
    return [row(name) for name in cli.HARNESSES]


def fake_run():
    proc = SimpleNamespace(communicate=MagicMock(return_value=("", "")))
    return SimpleNamespace(proc=proc, started=time.perf_counter(), timeout=2.0,
                           wall=0.0, peak_rss=0, cpu={}, seen={})


def test_percentile_nearest_rank():
    assert cli.percentile(list(range(1, 21)), 95) == 19
    assert cli.percentile(list(range(1, 31)), 95) == 29
    assert cli.percentile([7], 95) == 7
    with pytest.raises(ValueError, match="at least one"):
        cli.percentile([], 95)


def test_publication_requires_every_harness_and_clean_batches():
    assert cli.publication_problems(complete_rows(), args()) == []
    assert cli.publication_problems([], args())
    assert cli.publication_problems(complete_rows()[:-1], args())
    assert cli.publication_problems([row("Agent RT")] * 5, args())

    incomplete = complete_rows()
    incomplete[0]["runs"] = 0
    assert "sequential" in " ".join(cli.publication_problems(incomplete, args()))
    incomplete = complete_rows()
    incomplete[1]["c8_failures"] = 1
    assert "batch" in " ".join(cli.publication_problems(incomplete, args()))
    incomplete = complete_rows()
    incomplete[1]["c8_model_requests"] = 0
    assert "requests" in " ".join(cli.publication_problems(incomplete, args()))
    incomplete = complete_rows()
    incomplete[1]["c4_batch_peak_rss_mib"] = ""
    assert "RSS" in " ".join(cli.publication_problems(incomplete, args()))


def test_live_batch_rss_is_not_sum_of_individual_peaks():
    runs = [fake_run(), fake_run()]
    mib = 2**20
    # Each tuple is (root alive, instantaneous RSS), ordered by sweep.
    observations = iter([(True, 10 * mib), (True, 20 * mib),
                         (True, 40 * mib), (True, 5 * mib),
                         (False, 0), (False, 0)])
    ledger = SimpleNamespace(requests=2)
    sandbox = SimpleNamespace(server=SimpleNamespace(ledger=ledger))
    samples = iter([
        {"success": True, "wall_ms": 100, "cpu_ms": 50, "peak_rss_mib": 40},
        {"success": True, "wall_ms": 100, "cpu_ms": 50, "peak_rss_mib": 20},
    ])
    with patch.object(cli, "make_sandbox", return_value=sandbox), \
         patch.object(cli, "start_run", side_effect=runs), \
         patch.object(cli, "sample", side_effect=lambda _: next(observations)), \
         patch.object(cli, "finish", side_effect=lambda *a: next(samples)), \
         patch.object(cli, "kill_run") as cleanup, \
         patch.object(cli, "close_sandbox") as close:
        batch = cli.run_batch("Agent RT", "fake", 2, args(), "concurrency-2")
    assert batch.peak_rss_mib == 45
    assert sum(s["peak_rss_mib"] for s in batch.samples) == 60
    assert batch.requests == batch.expected_min_requests == 2
    assert cleanup.call_count == 2
    close.assert_called_once_with(sandbox)


def test_shared_mock_missing_requests_fails_entire_batch():
    runs = [fake_run(), fake_run()]
    sandbox = SimpleNamespace(server=SimpleNamespace(ledger=SimpleNamespace(requests=1)))
    with patch.object(cli, "make_sandbox", return_value=sandbox), \
         patch.object(cli, "start_run", side_effect=runs), \
         patch.object(cli, "sample", return_value=(False, 0)), \
         patch.object(cli, "finish", return_value={"success": True, "error": ""}), \
         patch.object(cli, "kill_run"), patch.object(cli, "close_sandbox"):
        batch = cli.run_batch("Agent RT", "fake", 2, args(), "concurrency-2")
    assert batch.requests == 1
    assert batch.expected_min_requests == 2
    assert all(not item["success"] and "model requests" in item["error"] for item in batch.samples)


def test_partially_started_batch_is_cleaned_up():
    run = fake_run()
    sandbox = SimpleNamespace(server=SimpleNamespace(ledger=SimpleNamespace(requests=0)))
    with patch.object(cli, "make_sandbox", return_value=sandbox), \
         patch.object(cli, "start_run", side_effect=[run, RuntimeError("startup failed")]), \
         patch.object(cli, "kill_run") as cleanup, \
         patch.object(cli, "close_sandbox") as close:
        with pytest.raises(RuntimeError, match="startup failed"):
            cli.run_batch("Agent RT", "fake", 2, args(), "concurrency-2")
    cleanup.assert_called_once_with(run)
    run.proc.communicate.assert_called_once_with(timeout=5)
    close.assert_called_once_with(sandbox)


def test_sandbox_builder_failure_cleans_directory(tmp_path):
    seen = []
    (tmp_path / "sandbox").mkdir()
    class FakeServer:
        url = "http://127.0.0.1:1234"
        def shutdown(self):
            seen.append("shutdown")
        def server_close(self):
            seen.append("server_close")

    def fail_builder(binary, url, home):
        seen.append(str(home.parent))
        raise RuntimeError("bad configuration")

    with patch.object(cli.tempfile, "mkdtemp", return_value=str(tmp_path / "sandbox")), \
         patch.object(cli, "MockServer", FakeServer), \
         patch.dict(cli.HARNESSES, {"Agent RT": ("fake", fail_builder)}):
        with pytest.raises(RuntimeError, match="bad configuration"):
            cli.make_sandbox("Agent RT", "fake")
    assert seen[-2:] == ["shutdown", "server_close"]
    assert not (tmp_path / "sandbox").exists()


@pytest.mark.parametrize("flag,value", [
    ("--concurrency", "2"), ("--input-usd-per-mtok", "2.5"),
    ("--output-usd-per-mtok", "12"), ("--warmup", "0"),
])
def test_nonstandard_contract_cannot_publish(flag, value, tmp_path):
    argv = ["benchmark_cli_harnesses.py", "--publish", "--runs", "30",
            "--output-dir", str(tmp_path), flag, value]
    with patch.object(sys, "argv", argv), patch.object(cli, "resolve_binary") as resolve:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    resolve.assert_not_called()


def test_missing_cli_preflight_never_publishes(tmp_path):
    argv = ["benchmark_cli_harnesses.py", "--publish", "--runs", "30",
            "--output-dir", str(tmp_path)]
    with patch.object(sys, "argv", argv), \
         patch.object(cli, "resolve_binary", return_value=None), \
         patch.object(cli, "run_batch") as run:
        assert cli.main() == 1
    run.assert_not_called()
    assert not list(tmp_path.iterdir())
