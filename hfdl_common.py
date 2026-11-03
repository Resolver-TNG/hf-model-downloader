"""Shared helpers for the HF model download bundle (stdlib only)."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "model_manifest.json"

RECEIPT_NAME = "download_receipt.jsonl"
SUMS_NAME = "sha256sums.txt"
VERIFY_REPORT_NAME = "verify_report.json"
# Files that live in the destination root and are not model payload.
ROOT_META_NAMES = {RECEIPT_NAME, SUMS_NAME, VERIFY_REPORT_NAME}
# huggingface_hub keeps resume metadata under <local_dir>/.cache/huggingface
CACHE_DIRNAME = ".cache"

EVENT_PREFIX = "@@EVENT "

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")

_emit_lock = threading.Lock()
_events_enabled = False


def enable_events(flag: bool = True) -> None:
    global _events_enabled
    _events_enabled = flag


def emit(event: str, **fields: Any) -> None:
    """Print one machine-readable event line (only when events are enabled)."""
    if not _events_enabled:
        return
    payload = {"event": event, "t": round(time.time(), 3), **fields}
    line = EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _emit_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def human_bytes(n: int | float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if x < 1000 or unit == "PB":
            return f"{x:.2f} {unit}"
        x /= 1000
    return f"{x:.2f} PB"


def redact(text: str, secrets: list[str] | None = None) -> str:
    """Mask HF tokens (known secrets and anything that looks like one)."""
    for s in secrets or []:
        if s and len(s) >= 8:
            text = text.replace(s, "hf_****")
    return _TOKEN_RE.sub("hf_****", text)


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    for m in manifest["models"]:
        if not _REPO_ID_RE.match(m["repo_id"]):
            raise ValueError(f"invalid repo_id in manifest: {m['repo_id']!r}")
        if not _REVISION_RE.match(m["revision"]):
            raise ValueError(f"revision must be a 40-hex commit sha: {m['repo_id']} -> {m['revision']!r}")
    return manifest


def select_models(manifest: dict[str, Any], group: str | None, only: list[str] | None) -> list[dict[str, Any]]:
    models = manifest["models"]
    if only:
        wanted = set(only)
        selected = [m for m in models if m["repo_id"] in wanted]
        missing = wanted - {m["repo_id"] for m in selected}
        if missing:
            raise SystemExit(f"Unknown repo_id(s): {', '.join(sorted(missing))}")
    else:
        group = group or manifest.get("default_group", "core4")
        selected = [m for m in models if group in m.get("groups", [])]
        if not selected:
            groups = sorted(manifest.get("groups", {}).keys())
            raise SystemExit(f"No models selected for group={group!r}. Available groups: {groups}")
    return sorted(selected, key=lambda m: m.get("priority", 999))


def dir_usage(path: Path) -> int:
    """Bytes actually on disk under path, resume/.incomplete files included.

    Uses allocated blocks where the OS reports them (POSIX) so that a pre-allocated sparse
    .incomplete file is not counted as downloaded; falls back to apparent size (Windows).
    """
    total = 0
    stack = [str(path)]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            st = e.stat(follow_symlinks=False)
                            blocks = getattr(st, "st_blocks", None)
                            total += min(st.st_size, blocks * 512) if blocks is not None else st.st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def payload_usage(path: Path) -> int:
    """Bytes of finished payload only (skips the .cache resume metadata)."""
    total = 0
    for p in iter_payload_files(path):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def iter_payload_files(root: Path) -> Iterator[Path]:
    """Yield payload files under root, skipping .cache dirs at any depth and root-level meta files."""
    root = Path(root)
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            entries = sorted(os.scandir(cur), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name == CACHE_DIRNAME:
                        continue
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    if cur == root and e.name in ROOT_META_NAMES:
                        continue
                    yield Path(e.path)
            except OSError:
                continue


def nearest_existing_parent(p: Path) -> Path:
    p = Path(p)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return p


def safe_join(root: Path, rel: str) -> Path | None:
    """Join an untrusted relative path under root; None if it escapes root."""
    root_r = Path(root).resolve()
    cand = (root_r / rel).resolve()
    try:
        cand.relative_to(root_r)
    except ValueError:
        return None
    return cand


def read_token_from_stdin() -> str | None:
    line = sys.stdin.readline()
    tok = line.strip()
    return tok or None


# ---------------------------------------------------------------- ambient Hugging Face token
_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def ambient_token() -> tuple[str | None, str | None]:
    """Token that huggingface_hub will send on its own when no token is passed explicitly.

    Returns (token, source) where source is "env" (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) or
    "login" (saved by `hf auth login` / `huggingface_hub.login()`), or (None, None).
    A saved login is not used when HF_HUB_DISABLE_IMPLICIT_TOKEN is set, so it is not reported then.
    The token value is returned only so callers can redact it from logs; never print it.
    """
    env = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if env:
        return env, "env"
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").upper() in _TRUE_VALUES:
        return None, None
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None, None
    try:
        tok = get_token()
    except Exception:  # noqa: BLE001 - e.g. a failing OIDC exchange; treat as no token
        return None, None
    return (tok, "login") if tok else (None, None)


# ---------------------------------------------------------------- GGUF file variants
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)")
_SEP_CHARS = "-._/"


def gguf_variant_key(path: str) -> str:
    """'dir/model-Q4_K_M-00001-of-00003.gguf' -> 'dir/model-Q4_K_M' (split parts share one key)."""
    return _SPLIT_RE.sub("", path)[: -len(".gguf")]


def group_gguf_variants(files: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Group a repo's .gguf files into selectable variants (one quantisation, split parts together).

    files: (path, size) for every file in the repo. Returns a list of
    {"id", "label", "files", "size_bytes"} sorted by id; an empty list when the repo has fewer than two variants.
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for path, size in files:
        if path.endswith(".gguf"):
            groups.setdefault(gguf_variant_key(path), []).append((path, int(size or 0)))
    if len(groups) < 2:
        return []
    keys = sorted(groups)
    prefix = os.path.commonprefix(keys)
    cut = max((prefix.rfind(c) for c in _SEP_CHARS), default=-1) + 1  # back off to the last separator
    out = []
    for k in keys:
        label = k[cut:] or k
        out.append({"id": k, "label": label, "files": sorted(p for p, _ in groups[k]),
                    "size_bytes": sum(s for _, s in groups[k])})
    return out


def parse_variant_args(values: list[str]) -> dict[str, list[str]]:
    """['org/repo=VARIANT', ...] -> {'org/repo': ['VARIANT', ...]}"""
    out: dict[str, list[str]] = {}
    for v in values or []:
        repo, sep, var = v.partition("=")
        if not sep or not repo or not var:
            raise SystemExit(f"--variant must look like org/repo=VARIANT_ID, got {v!r}")
        out.setdefault(repo, []).append(var)
    return out


def resolve_variants(model: dict[str, Any], chosen: list[str] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick variants of a model. chosen=None means all. Returns (selected_variants, files_to_skip).

    A chosen value may be a variant id or its label. Raises ValueError for unknown or empty selections.
    """
    variants = model.get("gguf_variants") or []
    if not variants:
        if chosen:
            raise ValueError(f"{model['repo_id']} has no selectable files")
        return [], []
    if chosen is None:
        return list(variants), []
    if not chosen:
        raise ValueError(f"select at least one file for {model['repo_id']}")
    by_key = {}
    for v in variants:
        by_key[v["id"]] = v
        by_key.setdefault(v["label"], v)
    picked: list[dict[str, Any]] = []
    for c in chosen:
        v = by_key.get(c)
        if v is None:
            raise ValueError(f"unknown file variant for {model['repo_id']}: {c!r} (choose from: {', '.join(x['label'] for x in variants)})")
        if v not in picked:
            picked.append(v)
    skip = [f for v in variants if v not in picked for f in v["files"]]
    return picked, skip


def expected_bytes(model: dict[str, Any], selected_variants: list[dict[str, Any]]) -> int:
    """Download size for a model given its selected variants (the whole repo when it has none)."""
    if not model.get("gguf_variants"):
        return int(model.get("size_bytes", 0))
    return int(model.get("base_bytes", 0)) + sum(int(v["size_bytes"]) for v in selected_variants)


def fnmatch_escape(path: str) -> str:
    """Escape a literal path for fnmatch-style patterns (huggingface_hub allow/ignore_patterns)."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in path)
