#!/usr/bin/env python3
"""Generate / verify sha256sums.txt for moving a model tree to another machine.

The output is compatible with `sha256sum -c` / `shasum -a 256 -c` (format: "<hash>  ./relative/path").
The `.cache` resume folders (at any depth) and the bundle's own meta files are excluded.

  python3 make_checksums.py generate /Volumes/MODELS/hf_models
  python3 make_checksums.py verify   /Volumes/MODELS/hf_models
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hfdl_common as C

CHUNK = 16 * 1024 * 1024


def _sha256(path: Path, on_bytes) -> str:
    h = hashlib.sha256()
    buf = bytearray(CHUNK)
    view = memoryview(buf)
    with path.open("rb", buffering=0) as f:
        while True:
            n = f.readinto(buf)
            if not n:
                break
            h.update(view[:n])
            on_bytes(n)
    return h.hexdigest()


class _Prog:
    def __init__(self, total: int) -> None:
        self.total, self.done, self._lock, self._last = total, 0, threading.Lock(), 0.0

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            if time.time() - self._last >= 1.0:
                self._last = time.time()
                C.emit("checksum_progress", bytes_done=self.done, bytes_total=self.total)


def _escape_name(rel: str) -> tuple[str, str]:
    """GNU sha256sum escapes backslash/newline in names and prefixes the line with a backslash."""
    if "\\" in rel or "\n" in rel:
        return "\\", rel.replace("\\", "\\\\").replace("\n", "\\n")
    return "", rel


def generate(root: Path, out: Path, workers: int) -> int:
    files = sorted(C.iter_payload_files(root), key=lambda p: p.relative_to(root).as_posix().encode())
    total = sum(p.stat().st_size for p in files)
    C.emit("checksum_start", op="generate", files=len(files), bytes_total=total)
    print(f"Hashing {len(files)} files, {C.human_bytes(total)}, workers={workers}", flush=True)
    prog = _Prog(total)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        digests = list(ex.map(lambda p: _sha256(p, prog.add), files))
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for p, d in zip(files, digests):
            prefix, name = _escape_name(p.relative_to(root).as_posix())
            f.write(f"{prefix}{d}  ./{name}\n")
    os.replace(tmp, out)
    print(f"Wrote: {out}", flush=True)
    C.emit("done", ok=True, files=len(files))
    return 0


def verify(root: Path, sums: Path, workers: int) -> int:
    entries: list[tuple[str, str]] = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        esc = line.startswith("\\")
        if esc:
            line = line[1:]
        digest, _, name = line.partition("  ")
        name = name[2:] if name.startswith("./") else name
        if esc:
            name = name.replace("\\n", "\n").replace("\\\\", "\\")
        entries.append((digest, name))
    sizes = {}
    for _, n in entries:
        p = C.safe_join(root, n)
        sizes[n] = p.stat().st_size if p and p.is_file() else 0
    total = sum(sizes.values())
    C.emit("checksum_start", op="verify", files=len(entries), bytes_total=total)
    print(f"Verifying {len(entries)} files, {C.human_bytes(total)}, workers={workers}", flush=True)
    prog = _Prog(total)

    def one(e: tuple[str, str]) -> tuple[str, str]:
        digest, name = e
        p = C.safe_join(root, name)
        if p is None or not p.is_file():
            return name, "missing"
        return name, "ok" if _sha256(p, prog.add) == digest else "MISMATCH"

    bad = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for name, status in ex.map(one, entries):
            if status != "ok":
                bad += 1
                print(f"  {status}: {name}", flush=True)
    listed = {n for _, n in entries}
    extra = [p.relative_to(root).as_posix() for p in C.iter_payload_files(root) if p.relative_to(root).as_posix() not in listed]
    for n in extra[:50]:
        print(f"  not in sums (extra): {n}", flush=True)
    print("RESULT: " + ("all files match" if not bad else f"{bad} problem file(s)"), flush=True)
    C.emit("done", ok=(bad == 0), bad=bad, extra=len(extra))
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", choices=["generate", "verify"])
    ap.add_argument("root")
    ap.add_argument("--sums", help=f"Checksum file (default: <root>/{C.SUMS_NAME})")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--events", action="store_true")
    a = ap.parse_args(argv)
    C.enable_events(a.events)
    root = Path(a.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        return 1
    sums = Path(a.sums).expanduser().resolve() if a.sums else root / C.SUMS_NAME
    if a.op == "generate":
        return generate(root, sums, a.workers)
    if not sums.is_file():
        print(f"ERROR: checksum file not found: {sums}", file=sys.stderr)
        return 1
    return verify(root, sums, a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
