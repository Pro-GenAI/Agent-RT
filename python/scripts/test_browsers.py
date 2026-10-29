"""Run the web UI browser tests in every browser this OS can run (`make test-browsers`).

Playwright's Chromium, Firefox and WebKit (Safari's engine) are downloaded into a temporary
folder (PLAYWRIGHT_BROWSERS_PATH), never installed system-wide. Branded Chrome and Edge are
Chromium-based, so Chromium covers them.
Ends with a per-browser PASS/FAIL/SKIP summary; exits non-zero if any browser failed.
`--engine chromium` tests only Chromium (`make test-chromium`); AGENT_RT_CHROMIUM_PATH points
it at your own Chromium/Chrome executable instead of downloading Playwright's.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "tests/test_web_browsers.py"
ENGINES = ("chromium", "firefox", "webkit")
# Playwright's Windows builds expect the Microsoft Visual C++ runtime (for example
# msvcp140_1.dll), which none of its bundles ship and only an administrator can install.
VC_RUNTIME_NOTE = (
    "needs the Microsoft Visual C++ Redistributable (x64), which only an administrator can "
    "install: https://aka.ms/vs/17/release/vc_redist.x64.exe"
)
CHROMIUM_PATH = os.environ.get("AGENT_RT_CHROMIUM_PATH") or None
SAFARI_NOTE = {
    "Darwin": "Playwright cannot drive Safari itself (safaridriver needs `sudo safaridriver --enable`); "
    "webkit covers its engine",
    "Linux": "Apple ships no Safari for Linux; webkit (Safari's engine) is tested instead",
    "Windows": "Apple ships no current Safari for Windows; webkit (Safari's engine) is tested instead",
}


def tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def install(engine: str, env: dict[str, str]) -> str | None:
    """Download one Playwright engine into the temporary folder; return an error or None."""
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", engine],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return None if result.returncode == 0 else tail(result.stdout + result.stderr)


def add_host_libraries(reason: str, libs_dir: Path) -> bool:
    """Fetch the Debian/Ubuntu packages Playwright says are missing into libs_dir, without sudo.

    `apt-get download` + `dpkg-deb -x` unpack them into our folder only, then the libraries are
    copied into WebKit's own bundled `sys/lib` folder (its launcher resets LD_LIBRARY_PATH to
    that folder), so the host system is unchanged. Returns True if anything was added.
    """
    if not (shutil.which("apt-get") and shutil.which("dpkg-deb")):
        return False
    match = re.search(r"apt-get install ([^\n║]+)", reason)
    packages = match.group(1).split() if match else []
    # Playwright may name only the files; Debian packages libNAME.so.N as libNAMEN (libNAME-N
    # when NAME ends in a digit), e.g. libyuv.so.0 -> libyuv0, libgav1.so.1 -> libgav1-1.
    for name, soversion in re.findall(r"\b(lib[\w.+-]*?)\.so\.(\d+)\b", reason):
        packages.append(f"{name}{'-' if name[-1].isdigit() else ''}{soversion}".lower())
    packages = [
        name
        for name in dict.fromkeys(packages)
        if re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", name)
    ]
    debs = libs_dir / "debs"
    debs.mkdir(parents=True, exist_ok=True)
    added = False
    for package in packages:
        print(f"Fetching host library {package} into {libs_dir} …", flush=True)
        fetched = subprocess.run(
            ["apt-get", "download", package],
            cwd=debs,
            capture_output=True,
            text=True,
            check=False,
        )
        if fetched.returncode != 0:
            print(tail(fetched.stderr, 3))
            continue
        for deb in debs.glob(f"{package}_*.deb"):
            subprocess.run(
                ["dpkg-deb", "-x", str(deb), str(libs_dir / "root")], check=True
            )
            added = True
    if added:
        copy_host_libraries(libs_dir)
    return added


def copy_host_libraries(libs_dir: Path) -> None:
    """Put previously fetched libraries into each downloaded WebKit bundle (also after updates)."""
    bundles = list(libs_dir.parent.glob("webkit-*/minibrowser-*/sys/lib"))
    for library in (libs_dir / "root").rglob("*.so*"):
        for bundle in bundles:
            shutil.copy2(library, bundle / library.name, follow_symlinks=True)


def probe(engine: str) -> str | None:
    """Launch the browser once; return why it cannot run, or None."""
    from playwright.sync_api import Error, sync_playwright

    try:
        with sync_playwright() as playwright:
            path = CHROMIUM_PATH if engine == "chromium" else None
            getattr(playwright, engine).launch(executable_path=path).close()
    except Error as exc:
        return str(exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--browsers-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "agent-rt-browsers",
        help="where Playwright browsers are downloaded (default: <temp>/agent-rt-browsers)",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        action="append",
        help="test only this engine (repeatable; default: all)",
    )
    parser.add_argument("pytest_args", nargs="*", help="extra pytest arguments")
    args = parser.parse_args()

    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            'Playwright is missing; run `make test-browsers` (it adds "playwright" for this run only).'
        )
        return 2

    args.browsers_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(args.browsers_dir)
    system = platform.system()
    print(f"Browsers folder: {args.browsers_dir}  (delete it to reclaim space)")

    results: list[tuple[str, str, str]] = []
    targets: list[str] = []
    for engine in args.engine or ENGINES:
        if engine == "chromium" and CHROMIUM_PATH:
            print(f"Using Chromium at {CHROMIUM_PATH} (skipping download)")
            targets.append(engine)
            continue
        print(f"Installing {engine} …", flush=True)
        error = install(engine, dict(os.environ))
        if error:
            results.append((engine, "FAIL", f"install failed:\n{error}"))
        else:
            targets.append(engine)

    libs_dir = args.browsers_dir / "host-libs"
    copy_host_libraries(libs_dir)
    runnable = []
    for target in targets:
        reason = probe(target)
        # WebKit on Linux may need a few system libraries; fetch them locally (a few rounds,
        # since an added library can reveal the next missing one).
        for _ in range(5):
            missing = ("missing dependencies", "error while loading shared libraries")
            if not reason or not any(marker in reason for marker in missing):
                break
            if not add_host_libraries(reason, libs_dir):
                break
            reason = probe(target)
        if reason is None:
            runnable.append(target)
        elif system == "Windows" and re.search(r"(msvcp|vcruntime)\d+\S*\.dll", reason):
            # A missing system runtime, not a bug here; installing it is outside this script.
            results.append((target, "SKIP", VC_RUNTIME_NOTE))
        else:
            results.append((target, "FAIL", f"cannot launch:\n{tail(reason)}"))

    for target in runnable:
        print(f"\n=== {target} ===", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                TESTS,
                "-p",
                "no:cacheprovider",
                "-q",
                *args.pytest_args,
            ],
            cwd=ROOT,
            env={**os.environ, "AGENT_RT_BROWSER": target},
            check=False,
        )
        results.append((target, "PASS" if completed.returncode == 0 else "FAIL", ""))
    if not args.engine:
        results.append(
            ("safari", "SKIP", SAFARI_NOTE.get(system, "not available on this OS"))
        )

    order = {name: index for index, name in enumerate(ENGINES)}
    results.sort(key=lambda row: order.get(row[0], len(order)))
    print("\nBrowser summary")
    for name, status, detail in results:
        first, _, rest = detail.partition("\n")
        print(f"  {status:<4}  {name:<14} {first}")
        if rest and status == "FAIL":
            print("\n".join("          " + line for line in rest.splitlines()))
    counts = {
        status: sum(row[1] == status for row in results)
        for status in ("PASS", "FAIL", "SKIP")
    }
    print(
        f"Totals: {counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
