import argparse
import concurrent.futures
import csv
import json
import pathlib
import re
import shutil
import subprocess
import tempfile

HEAVY = re.compile(
    r"(torch|torchvision|torchaudio|pytorch|transformers|sentence[-_ ]transformers|"
    r"tensorflow(?:-cpu|-gpu)?|jaxlib|diffusers|accelerate|bitsandbytes|xformers|"
    r"deepspeed|vllm|llama[-_]cpp[-_]python|ctransformers|onnxruntime(?:-gpu)?|"
    r"paddlepaddle|mxnet)",
    re.I,
)
MANIFEST = re.compile(
    r"(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.cfg|setup\.py|Pipfile|"
    r"poetry\.lock|uv\.lock|pdm\.lock|package\.json|package-lock\.json|"
    r"pnpm-lock\.yaml|yarn\.lock)$",
    re.I,
)
TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__)(/|$)|(^|/)(test_|.*\.(test|spec)\.)",
    re.I,
)
SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
PATTERNS = {
    "LangChain": re.compile(r"(langchain(?:[_./-]|$)|@langchain/)", re.I),
    "LlamaIndex": re.compile(r"(llama[_-]?index|@llamaindex/)", re.I),
    "OpenAI": re.compile(r"(from +openai|import +openai|[\"']openai[\"']|@openai/)", re.I),
    "Anthropic": re.compile(r"(from +anthropic|import +anthropic|@anthropic-ai/sdk|[\"']anthropic[\"'])", re.I),
    "OpenAI Agents": re.compile(r"(from +agents|import +agents|@openai/agents)", re.I),
    "AutoGen": re.compile(r"(autogen_agentchat|autogen_ext|pyautogen|autogen-agentchat)", re.I),
    "CrewAI": re.compile(r"(from +crewai|import +crewai|[\"']crewai[\"'])", re.I),
}


def read_manifest_text(path: pathlib.Path) -> str:
    """Decode a dependency manifest, honouring UTF-16/UTF-8 byte-order marks.

    Windows tools (for example `pip freeze > requirements.txt` in PowerShell)
    write UTF-16 files. Decoding those as UTF-8 yields NUL-interleaved text
    that no package regex matches, while pip/uv still install them.
    """
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if len(raw) >= 4 and raw[1:4:2] == b"\x00\x00" and raw[0] != 0:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def find_heavy_dependencies(repo: pathlib.Path) -> list[dict[str, str]]:
    """Return heavyweight dependency matches from dependency manifests in a checkout."""
    matches: set[tuple[str, str]] = set()
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo).as_posix()
        if not MANIFEST.search(relative):
            continue
        if any(part in {".git", ".venv", "venv", "node_modules"} for part in path.parts):
            continue
        try:
            text = read_manifest_text(path)
        except OSError:
            continue
        for match in HEAVY.finditer(text):
            matches.add((match.group(1).lower(), relative))
    return [
        {"package": package, "manifest": manifest}
        for package, manifest in sorted(matches)
    ]


def audit(job):
    framework, candidate = job
    work = pathlib.Path(tempfile.mkdtemp(prefix="candidate-audit-"))
    repo = work / "repo"
    result = {
        "framework": framework,
        "url": candidate["url"],
        "language": candidate.get("lang"),
        "size": candidate.get("size", 0),
        "stars": candidate.get("stars", 0),
        "heavy": [],
        "has_tests": False,
        "target": False,
        "target_test": False,
        "error": "",
    }
    try:
        clone = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
             "--quiet", candidate["url"], str(repo)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=35,
        )
        if clone.returncode:
            result["error"] = clone.stderr[-200:]
            return result

        tree = subprocess.run(
            ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=12,
        )
        paths = tree.stdout.splitlines()
        selected = [
            path for path in paths
            if MANIFEST.search(path) or pathlib.Path(path).suffix.lower() in SOURCE_EXTENSIONS
        ]
        selected.sort(
            key=lambda path: (
                0 if MANIFEST.search(path) else 1 if TEST_PATH.search(path) else 2,
                len(path),
            )
        )
        for path in selected[:450]:
            shown = subprocess.run(
                ["git", "-C", str(repo), "show", "HEAD:" + path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
                timeout=6,
            )
            text = shown.stdout
            if MANIFEST.search(path):
                result["heavy"].extend(match.group(1).lower() for match in HEAVY.finditer(text))
            if TEST_PATH.search(path):
                result["has_tests"] = True
            if PATTERNS[framework].search(text):
                result["target"] = True
                if TEST_PATH.search(path):
                    result["target_test"] = True
        result["heavy"] = sorted(set(result["heavy"]))
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("output")
    args = parser.parse_args()

    candidates = json.load(open("/tmp/agent_rt_candidates.json"))
    existing = {
        row["RepoURL"]
        for row in csv.DictReader(open("scripts/migration_test/repo_list.csv", encoding="utf-8"))
        if row.get("RepoURL")
    }
    pool = [
        item for item in candidates[args.framework]
        if item["url"] not in existing
        and item.get("lang") in {"Python", "TypeScript", "JavaScript"}
    ][args.start:args.end]

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(audit, [(args.framework, item) for item in pool]))

    pathlib.Path(args.output).write_text(json.dumps(results))
    valid = [
        item for item in results
        if not item["heavy"] and item["has_tests"] and item["target"] and not item["error"]
    ]
    print("audited", len(results), "valid", len(valid))
    for item in valid:
        print(
            item["language"],
            item["url"],
            "target-test" if item["target_test"] else "source-target",
            "size", item["size"],
        )

if __name__ == "__main__":
    main()
