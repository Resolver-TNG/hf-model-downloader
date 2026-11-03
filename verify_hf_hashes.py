#!/usr/bin/env python3
"""Verify downloaded model files against the hashes published by Hugging Face.

For every file of each pinned revision this compares:
  * file exists and size matches the Hub's size
  * LFS files (weights, tokenizer.json, ...): SHA-256 equals the Hub's lfs.sha256
  * regular git files (config, README, ...): git blob SHA-1 equals the Hub's blob_id
Extra files found locally are reported but are not an error.

This is different from sha256sums.txt, which is generated from your own copy and can only
detect corruption that happens *after* it was generated (for example during transfer).

Multi-file GGUF repos (manifest "gguf_variants"): the repo's other files are always checked. Of the
GGUF variants, --variant REPO_ID=VARIANT checks exactly those; without it, every variant that has at
least one file on disk is checked and variants with no file at all are reported as skipped (if none is
on disk, all are checked, so they show up as missing).

Examples:
  python3 verify_hf_hashes.py --dest /Volumes/MODELS/hf_models --group core4
  python3 verify_hf_hashes.py --dest ./hf_models --only nvidia/Qwen3.8-27B-NVFP4 --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hfdl_common as C

CHUNK = 16 * 1024 * 1024


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self._lock = threading.Lock()
        self._last_emit = 0.0

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            now = time.time()
            if now - self._last_emit >= 1.0:
                self._last_emit = now
                C.emit("verify_progress", bytes_done=self.done, bytes_total=self.total)

    def flush(self) -> None:
        C.emit("verify_progress", bytes_done=self.done, bytes_total=self.total)


def sha256_file(path: Path, progress: Progress) -> str:
    h = hashlib.sha256()
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    with path.open("rb", buffering=0) as f:
        while True:
            n = f.readinto(buf)
            if not n:
                break
            h.update(view[:n])
            progress.add(n)
    return h.hexdigest()


def git_blob_sha1(path: Path, size: int, progress: Progress) -> str:
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
            progress.add(len(chunk))
    return h.hexdigest()


def check_file(local_dir: Path, item: dict[str, Any], quick: bool, progress: Progress) -> dict[str, Any]:
    rel = item["file"]
    res: dict[str, Any] = {"file": rel, "ok": False, "reason": ""}
    path = C.safe_join(local_dir, rel)
    if path is None:
        res["reason"] = "unsafe path in remote listing"
        return res
    if not path.is_file():
        res["reason"] = "missing"
        return res
    size = path.stat().st_size
    if size != item["size"]:
        res["reason"] = f"size mismatch: local {size} / hub {item['size']}"
        return res
    if quick:
        progress.add(size)
        res.update(ok=True, reason="size only")
        return res
    try:
        if item["lfs_sha256"]:
            got = sha256_file(path, progress)
            want, algo = item["lfs_sha256"], "sha256"
        else:
            got = git_blob_sha1(path, size, progress)
            want, algo = item["blob_id"], "git-sha1"
    except OSError as e:
        res["reason"] = f"read error: {e}"
        return res
    if got != want:
        res["reason"] = f"{algo} mismatch: local {got[:16]}... / hub {want[:16]}..."
        return res
    res.update(ok=True, reason=algo)
    return res


def fetch_remote_files(repo_id: str, revision: str, token: str | None) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, token=token)
    items = []
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        items.append({
            "file": s.rfilename,
            "size": int(s.size or 0),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
            "blob_id": s.blob_id,
        })
    return items


def files_to_check(m: dict[str, Any], items: list[dict[str, Any]], local_dir: Path,
                   chosen: list[str] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Split the remote listing into files to verify and GGUF variants skipped (see module docstring)."""
    variants = m.get("gguf_variants") or []
    if not variants:
        return items, []
    if chosen is not None:
        picked, _ = C.resolve_variants(m, chosen)
    else:
        def on_disk(rel: str) -> bool:
            path = C.safe_join(local_dir, rel)
            return path is not None and path.is_file()

        picked = [v for v in variants if any(on_disk(f) for f in v["files"])]
        if not picked:
            picked = list(variants)
    variant_files = {f for v in variants for f in v["files"]}
    keep = {f for v in picked for f in v["files"]}
    checked = [i for i in items if i["file"] not in variant_files or i["file"] in keep]
    skipped = [v["label"] for v in variants if v not in picked]
    return checked, skipped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify downloaded models against Hugging Face hashes.")
    p.add_argument("--manifest", default=str(C.MANIFEST_PATH))
    p.add_argument("--dest", required=True)
    p.add_argument("--group", default=os.environ.get("MODEL_GROUP"))
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--variant", action="append", default=[], metavar="REPO_ID=VARIANT",
                   help="For a multi-file GGUF repo, verify only this file variant (repeatable)")
    p.add_argument("--quick", action="store_true", help="Check existence and size only (no hashing)")
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    p.add_argument("--token-stdin", action="store_true")
    p.add_argument("--events", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    C.enable_events(args.events)
    token = C.read_token_from_stdin() if args.token_stdin else None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    secrets = [token] if token else []

    def log(msg: str) -> None:
        print(C.redact(msg, secrets), flush=True)

    manifest = C.load_manifest(args.manifest)
    selected = C.select_models(manifest, args.group, args.only)
    variant_args = C.parse_variant_args(args.variant)
    stray = sorted(set(variant_args) - {m["repo_id"] for m in selected})
    if stray:
        raise SystemExit(f"--variant given for a model that is not selected: {', '.join(stray)}")
    for m in selected:
        try:
            C.resolve_variants(m, variant_args.get(m["repo_id"]))
        except ValueError as e:
            raise SystemExit(f"ERROR: {e}") from None
    dest = Path(args.dest).expanduser().resolve()
    if not dest.is_dir():
        log(f"ERROR: destination not found: {dest}")
        C.emit("done", ok=False, reason="dest_missing")
        return 2

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log("ERROR: huggingface_hub is not installed. Run: python3 -m pip install -r requirements.txt")
        C.emit("done", ok=False, reason="huggingface_hub_missing")
        return 3

    if not token:
        ambient, _ = C.ambient_token()  # used by huggingface_hub on its own; only needed for redaction here
        if ambient:
            secrets.append(ambient)

    log("Fetching file hashes from Hugging Face ...")
    plans: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    remote_all: dict[str, set[str]] = {}
    skipped_by_repo: dict[str, list[str]] = {}
    for m in selected:
        try:
            items = fetch_remote_files(m["repo_id"], m["revision"], token)
        except Exception as e:  # noqa: BLE001
            log(f"ERROR: cannot read metadata for {m['repo_id']}: {C.redact(repr(e), secrets)}")
            C.emit("done", ok=False, reason="metadata_failed", repo_id=m["repo_id"])
            return 2
        remote_all[m["repo_id"]] = {i["file"] for i in items}
        checked, skipped = files_to_check(m, items, dest / m["repo_id"], variant_args.get(m["repo_id"]))
        skipped_by_repo[m["repo_id"]] = skipped
        if skipped:
            log(f"{m['repo_id']}: skipping file variant(s) not selected / not on disk: {', '.join(skipped)}")
        plans.append((m, checked))

    total = sum(i["size"] for _, items in plans for i in items)
    progress = Progress(total)
    C.emit("verify_start", bytes_total=total, mode="quick" if args.quick else "full",
           models=[{"repo_id": m["repo_id"], "files": len(items), "bytes": sum(i["size"] for i in items)} for m, items in plans])
    log(f"Verifying {sum(len(i) for _, i in plans)} files, {C.human_bytes(total)} "
        f"({'size only' if args.quick else 'full hash'}), workers={args.workers}")

    report: dict[str, Any] = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              "mode": "quick" if args.quick else "full", "models": []}
    all_ok = True
    for m, items in plans:
        repo_id = m["repo_id"]
        local_dir = dest / repo_id
        bad: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(check_file, local_dir, it, args.quick, progress) for it in items]
            for fu in as_completed(futs):
                r = fu.result()
                if not r["ok"]:
                    bad.append(r)
                    log(f"  BAD  {repo_id}/{r['file']}: {r['reason']}")
                    C.emit("verify_file_bad", repo_id=repo_id, file=r["file"], reason=r["reason"])
        remote_names = remote_all[repo_id]  # skipped variants that are on disk are not "extra"
        extra = []
        if local_dir.is_dir():
            for p in C.iter_payload_files(local_dir):
                rel = p.relative_to(local_dir).as_posix()
                if rel not in remote_names:
                    extra.append(rel)
        ok = not bad
        all_ok &= ok
        log(f"{'OK  ' if ok else 'FAIL'} {repo_id}: {len(items) - len(bad)}/{len(items)} files"
            + (f", {len(extra)} extra local file(s)" if extra else ""))
        skipped = skipped_by_repo.get(repo_id, [])
        C.emit("verify_model_done", repo_id=repo_id, ok=ok, bad=len(bad), total=len(items), extra=len(extra),
               skipped_variants=len(skipped))
        report["models"].append({"repo_id": repo_id, "revision": m["revision"], "ok": ok,
                                 "files": len(items), "bad": bad, "extra_local_files": extra,
                                 "skipped_variants": skipped})
    progress.flush()
    report["ok"] = all_ok
    try:
        (dest / C.VERIFY_REPORT_NAME).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log(f"WARNING: could not write {C.VERIFY_REPORT_NAME}: {e}")
    log("RESULT: " + ("all files match Hugging Face" if all_ok else "MISMATCH FOUND (see above)"))
    C.emit("done", ok=all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
