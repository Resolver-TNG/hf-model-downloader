#!/usr/bin/env python3
"""Download pinned Hugging Face model snapshots listed in model_manifest.json.

Examples:
  python3 download_models.py --dry-run --group core4 --dest /Volumes/MODELS/hf_models
  python3 download_models.py --group core4 --dest /Volumes/MODELS/hf_models
  printf '%s\\n' "$HF_TOKEN" | python3 download_models.py --token-stdin --group all --dest ./hf_models
  # GGUF repo with several files: fetch only one of them (id or label from model_manifest.json "gguf_variants")
  python3 download_models.py --only Kujira/Swift-1.5-Qwen3.8-27B-CHiPS-Uncensored-GSQ-RCO-MTP-GGUF \\
      --variant Kujira/Swift-1.5-Qwen3.8-27B-CHiPS-Uncensored-GSQ-RCO-MTP-GGUF=IQ3_S-mtp --dest ./hf_models

Behaviour:
  * Each model is retried (network errors only) with exponential backoff; one failed
    model does not stop the others. The exit code is 1 if any model failed.
  * Resume is handled by huggingface_hub (local_dir + .cache/huggingface metadata).
  * --dry-run has no side effects (it does not create the destination directory).
  * Token: --token-stdin, else HF_TOKEN, else a login saved by `hf auth login` (used automatically by
    huggingface_hub). The value is never printed; the log only says which source is used.
  * --variant picks files of a multi-file GGUF repo; the repo's other files (README, licence, ...) are
    always fetched. Without --variant every file is fetched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import hfdl_common as C


def _log(msg: str, secrets: list[str]) -> None:
    print(C.redact(msg, secrets), flush=True)


def _is_fatal(exc: BaseException) -> bool:
    """Errors that retrying cannot fix (auth, missing repo/revision, disk full)."""
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"GatedRepoError", "RepositoryNotFoundError", "RevisionNotFoundError", "DisabledRepoError"}:
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (28, 13, 30):  # ENOSPC, EACCES, EROFS
        return True
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    return status in (401, 403, 404)


def _remaining_bytes(dest: Path, plan: list[dict[str, Any]]) -> int:
    rem = 0
    for p in plan:
        rem += max(0, p["expected"] - C.dir_usage(dest / p["model"]["repo_id"]))
    return rem


def _build_plan(selected: list[dict[str, Any]], variant_args: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Per model: selected file variants, files to skip and the expected download size."""
    known = {m["repo_id"] for m in selected}
    stray = sorted(set(variant_args) - known)
    if stray:
        raise SystemExit(f"--variant given for a model that is not selected: {', '.join(stray)}")
    plan = []
    for m in selected:
        try:
            variants, skip = C.resolve_variants(m, variant_args.get(m["repo_id"]))
        except ValueError as e:
            raise SystemExit(f"ERROR: {e}") from None
        plan.append({"model": m, "variants": variants, "skip": skip, "expected": C.expected_bytes(m, variants)})
    return plan


def build_parser() -> argparse.ArgumentParser:
    here = C.HERE
    p = argparse.ArgumentParser(description="Download pinned HF model snapshots.")
    p.add_argument("--manifest", default=str(here / "model_manifest.json"))
    p.add_argument("--dest", default=os.environ.get("DEST", str(here / "hf_models")))
    p.add_argument("--group", default=os.environ.get("MODEL_GROUP"), help="core3 / core4 / no_coder / all")
    p.add_argument("--only", action="append", default=[], help="Download only this repo_id (repeatable)")
    p.add_argument("--variant", action="append", default=[], metavar="REPO_ID=VARIANT",
                   help="For a multi-file GGUF repo, fetch only this file variant (id or label; repeatable)")
    p.add_argument("--dry-run", action="store_true", help="Show the plan only; no side effects")
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_MAX_WORKERS", "8")))
    p.add_argument("--space-factor", type=float, default=float(os.environ.get("SPACE_FACTOR", "1.25")),
                   help="Safety margin applied to the remaining bytes (warning only)")
    p.add_argument("--retries", type=int, default=5, help="Attempts per model for retryable errors")
    p.add_argument("--retry-wait", type=float, default=15.0, help="Base wait seconds (doubles each retry, max 300)")
    p.add_argument("--token-stdin", action="store_true", help="Read the HF token from the first line of stdin")
    p.add_argument("--high-performance", action="store_true", help="Set HF_XET_HIGH_PERFORMANCE=1")
    p.add_argument("--events", action="store_true", help="Emit machine-readable '@@EVENT' lines (used by the GUI)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    C.enable_events(args.events)

    token: str | None = C.read_token_from_stdin() if args.token_stdin else None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    secrets = [token] if token else []

    # These are read when huggingface_hub is imported, so set them first.
    if args.events:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if args.high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    manifest = C.load_manifest(args.manifest)
    selected = C.select_models(manifest, args.group, args.only)
    plan = _build_plan(selected, C.parse_variant_args(args.variant))
    dest = Path(args.dest).expanduser().resolve()

    payload = sum(p["expected"] for p in plan)
    remaining = _remaining_bytes(dest, plan) if dest.exists() else payload
    probe = C.nearest_existing_parent(dest)
    free = shutil.disk_usage(probe).free
    margin = int(remaining * args.space_factor)

    _log("Selected models:", secrets)
    for p in plan:
        m = p["model"]
        _log(f"- {m['repo_id']} @ {m['revision'][:12]}  {C.human_bytes(p['expected'])}", secrets)
        if m.get("gguf_variants"):
            names = ", ".join(v["label"] for v in p["variants"])
            _log(f"    files: {names}  ({len(p['variants'])} of {len(m['gguf_variants'])})", secrets)
    _log(f"Destination: {dest}", secrets)
    _log(f"Planned payload: {C.human_bytes(payload)}   remaining: {C.human_bytes(remaining)}", secrets)
    _log(f"Free space at {probe}: {C.human_bytes(free)}  (with x{args.space_factor:.2f} margin need {C.human_bytes(margin)})", secrets)
    C.emit("plan", dest=str(dest), payload=payload, remaining=remaining, free=free,
           models=[{"repo_id": p["model"]["repo_id"], "revision": p["model"]["revision"], "size_bytes": p["expected"],
                    "variants": [v["id"] for v in p["variants"]]} for p in plan])

    if free < remaining:
        _log("ERROR: free space is smaller than the remaining download size. Aborting.", secrets)
        C.emit("done", ok=False, reason="insufficient_space", failed=[])
        return 2
    if free < margin:
        _log("WARNING: free space is below the recommended margin.", secrets)

    if args.dry_run:
        _log("Dry-run only; nothing was created or downloaded.", secrets)
        C.emit("done", ok=True, dry_run=True, failed=[])
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log("ERROR: huggingface_hub is not installed. Run: python3 -m pip install -r requirements.txt", secrets)
        C.emit("done", ok=False, reason="huggingface_hub_missing", failed=[])
        return 3

    # With no explicit token huggingface_hub sends HF_TOKEN or the saved `hf auth login` token on its own.
    ambient, source = (None, None) if token else C.ambient_token()
    if ambient:
        secrets.append(ambient)
    if token:
        token_desc = "provided (not shown)"
    elif source == "env":
        token_desc = "from the HF_TOKEN environment variable (not shown)"
    elif source == "login":
        token_desc = "logged-in token saved by `hf auth login` (not shown; run `hf auth logout` to stop using it)"
    else:
        token_desc = "none (anonymous)"
    _log("HF token: " + token_desc, secrets)
    C.emit("token", source="explicit" if token else (source or "none"))
    if not (token or ambient):
        gated = [m["repo_id"] for m in selected if m.get("requires_token")]
        if gated:
            _log("WARNING: these models are gated and need an HF token from an account that accepted the licence: "
                 + ", ".join(gated), secrets)

    dest.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for p in plan:
        m = p["model"]
        repo_id, revision = m["repo_id"], m["revision"]
        local_dir = dest / repo_id
        local_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        ok = False
        last_err = ""
        C.emit("model_start", repo_id=repo_id, size_bytes=p["expected"])
        _log(f"\n=== Downloading {repo_id} @ {revision[:12]} ===", secrets)
        if p["skip"]:
            _log(f"skipping {len(p['skip'])} unselected file(s): " + ", ".join(p["skip"]), secrets)
        ignore = [C.fnmatch_escape(f) for f in p["skip"]] or None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type="model",
                    local_dir=str(local_dir),
                    token=token,
                    max_workers=args.max_workers,
                    ignore_patterns=ignore,
                )
                ok = True
                break
            except KeyboardInterrupt:
                _log("Interrupted.", secrets)
                C.emit("done", ok=False, reason="interrupted", failed=failed)
                return 130
            except Exception as e:  # noqa: BLE001 - classified below
                last_err = C.redact(repr(e), secrets)
                fatal = _is_fatal(e)
                if fatal and m.get("requires_token"):
                    last_err += "  (gated model: accept the licence on its Hugging Face page and use a token from that account)"
                _log(f"attempt {attempt}/{args.retries} failed: {last_err}", secrets)
                if fatal or attempt >= args.retries:
                    break
                wait = min(300.0, args.retry_wait * (2 ** (attempt - 1)))
                C.emit("model_retry", repo_id=repo_id, attempt=attempt, wait=wait, error=last_err)
                _log(f"retrying in {wait:.0f}s ...", secrets)
                time.sleep(wait)

        elapsed = round(time.time() - started, 2)
        rec: dict[str, Any] = {"repo_id": repo_id, "revision": revision, "elapsed_sec": elapsed,
                               "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if m.get("gguf_variants"):
            rec["variants"] = [v["id"] for v in p["variants"]]
        if ok:
            on_disk = C.payload_usage(local_dir)
            expected = p["expected"]
            # "extra": more than expected, e.g. other file variants kept from an earlier run
            size_state = "ok" if on_disk == expected else ("extra" if on_disk > expected else "short")
            rec.update(status="ok", bytes_on_disk=on_disk, expected_bytes=expected, size_state=size_state,
                       size_match=size_state != "short")
            note = {"ok": "", "extra": f"  (expected {C.human_bytes(expected)}; the folder also holds other files, e.g. variants fetched earlier)",
                    "short": f"  WARNING: expected {C.human_bytes(expected)}"}[size_state]
            _log(f"Done: {repo_id} ({elapsed:.1f}s) payload {C.human_bytes(on_disk)}" + note, secrets)
            C.emit("model_done", repo_id=repo_id, elapsed=elapsed, bytes_on_disk=on_disk,
                   size_match=size_state != "short", size_state=size_state)
        else:
            failed.append(repo_id)
            rec.update(status="failed", error=last_err)
            _log(f"FAILED: {repo_id}: {last_err}", secrets)
            C.emit("model_failed", repo_id=repo_id, error=last_err)
        with (dest / C.RECEIPT_NAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    if failed:
        _log(f"\nCompleted with failures: {', '.join(failed)}", secrets)
        C.emit("done", ok=False, reason="model_failed", failed=failed)
        return 1
    _log("\nAll selected downloads completed. Next: verify against Hugging Face hashes (verify_hf_hashes.py).", secrets)
    C.emit("done", ok=True, failed=[])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
