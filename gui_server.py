#!/usr/bin/env python3
"""Local web GUI for the HF model download bundle (Python standard library only).

Start:   python3 gui_server.py            (or start_gui_windows.bat / start_gui_mac.command / start_gui_linux.sh)
Open:    the URL printed on start (it contains a one-time session key)

Security model
  * Listens on 127.0.0.1 only (there is deliberately no option to change that).
    For a remote box use:  ssh -L 8765:127.0.0.1:8765 user@host
  * Every API call needs the random per-run session key; the Host and Origin headers are
    checked as well, which blocks DNS-rebinding and cross-site requests from other tabs.
  * The Hugging Face token lives only in this process's memory. It is never written to disk,
    never placed in an environment variable of the worker (it is piped via stdin), never logged
    (all log output is redacted) and is cleared on request, on TTL expiry, after a job (optional)
    and when the server exits.
  * Workers are started with a fixed argv from validated parameters; only repo_ids that exist in
    the bundled manifest are accepted. No shell is used.
"""
from __future__ import annotations

import argparse
import collections
import hmac
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import hfdl_common as C

HERE = C.HERE
INDEX_PATH = HERE / "gui" / "index.html"
HOST = "127.0.0.1"
MAX_BODY = 64 * 1024
TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")

SCRIPTS = {
    "download": "download_models.py",
    "dry_run": "download_models.py",
    "verify": "verify_hf_hashes.py",
    "verify_quick": "verify_hf_hashes.py",
    "checksum_generate": "make_checksums.py",
    "checksum_verify": "make_checksums.py",
}
NEEDS_TOKEN = {"download", "verify", "verify_quick"}


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class Job:
    def __init__(self, kind: str, dest: str, models: list[dict[str, Any]], argv: list[str]) -> None:
        self.id = secrets.token_hex(4)
        self.kind = kind
        self.dest = dest
        self.argv_display = " ".join(argv)
        self.status = "running"          # running | finished | failed | stopped
        self.started = time.time()
        self.ended: float | None = None
        self.exit_code: int | None = None
        self.proc: subprocess.Popen | None = None
        self.stop_requested = False
        self.log: collections.deque[str] = collections.deque(maxlen=4000)
        self.log_base = 0                # absolute index of log[0]
        # "size" is the expected download size for this job (only the selected files of a GGUF repo)
        self.models = {m["repo_id"]: {"repo_id": m["repo_id"], "size": int(m.get("_expected", m.get("size_bytes", 0))),
                                       "bytes": 0, "state": "pending", "note": ""} for m in models}
        self.bytes_done = 0              # verify / checksum progress
        self.bytes_total = 0
        self.speed_samples: collections.deque[tuple[float, int]] = collections.deque(maxlen=60)
        self.final: dict[str, Any] = {}

    def add_log(self, line: str) -> None:
        if len(self.log) == self.log.maxlen:
            self.log_base += 1
        self.log.append(line)


class State:
    def __init__(self, manifest: dict[str, Any], default_dest: str) -> None:
        self.lock = threading.RLock()
        self.manifest = manifest
        self.default_dest = default_dest
        self.job: Job | None = None
        self._token: str | None = None
        self.token_info: dict[str, Any] = {}
        self.token_expires: float | None = None
        self.token_clear_on_finish = True
        self.notice = ""                 # one-line message about token auto-clearing
        self._ambient: tuple[float, str | None] = (0.0, None)

    # ---- token huggingface_hub would use by itself (HF_TOKEN or `hf auth login`) ----
    def ambient_source(self) -> str | None:
        """'env', 'login' or None. Only the source is kept here, never the token value. Cached for 20 s."""
        with self.lock:
            ts, src = self._ambient
            if time.time() - ts < 20:
                return src
        _tok, src = C.ambient_token()
        del _tok
        with self.lock:
            self._ambient = (time.time(), src)
        return src

    def token_available(self) -> bool:
        """True if a job would run with some token (GUI token, HF_TOKEN or a saved login)."""
        return bool(self.get_token() or self.ambient_source())

    # ---- token (memory only) ----
    def set_token(self, token: str, info: dict[str, Any], ttl_minutes: int, clear_on_finish: bool) -> None:
        with self.lock:
            self._token = token
            self.token_info = info
            self.token_expires = time.time() + ttl_minutes * 60 if ttl_minutes > 0 else None
            self.token_clear_on_finish = clear_on_finish
            self.notice = ""

    def clear_token(self, why: str = "") -> None:
        with self.lock:
            had = self._token is not None
            self._token = None
            self.token_info = {}
            self.token_expires = None
            if had and why:
                self.notice = why

    def get_token(self) -> str | None:
        with self.lock:
            if self._token and self.token_expires and time.time() >= self.token_expires:
                self.clear_token("Token was cleared automatically (TTL expired).")
            return self._token

    def token_status(self) -> dict[str, Any]:
        tok = self.get_token()
        ambient = self.ambient_source()
        with self.lock:
            st: dict[str, Any] = {"set": bool(tok), "ambient": ambient, "env_token": ambient == "env",
                                  "notice": self.notice}
            if tok:
                st.update(tail=tok[-4:], info=self.token_info, clear_on_finish=self.token_clear_on_finish,
                          expires_in=(max(0, int(self.token_expires - time.time())) if self.token_expires else None))
            return st


def whoami(token: str) -> dict[str, Any]:
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "hf-model-downloader-gui"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read())
    at = ((d.get("auth") or {}).get("accessToken")) or {}
    role = at.get("role")
    warn = ""
    perms: list[str] = []
    fg = at.get("fineGrained") or {}
    for scope in (fg.get("scoped") or []):
        perms += [str(p) for p in scope.get("permissions", [])]
    perms += [str(p) for p in (fg.get("global") or [])]
    if role in ("write", "admin") or any("write" in p or "manage" in p for p in perms):
        warn = "This token has write permissions. A read-only token is enough for downloading; prefer a Read / fine-grained read token."
    return {"user": d.get("name"), "role": role, "token_name": at.get("displayName"), "warning": warn}


class Handler(BaseHTTPRequestHandler):
    server_version = "HFDL-GUI"
    state: State
    session_key: str
    allowed_hosts: set[str]
    index_html: str

    def log_message(self, *_a: Any) -> None:  # never log: URLs contain the session key
        return

    # ---- helpers ----
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _err(self, code: int, msg: str) -> None:
        self._json(code, {"ok": False, "error": msg})

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "") in self.allowed_hosts

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin in {f"http://{h}" for h in self.allowed_hosts}

    def _key_ok(self) -> bool:
        return hmac.compare_digest(self.headers.get("X-Session-Key", ""), self.session_key)

    def _body(self) -> dict[str, Any] | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n > MAX_BODY:
            return None
        raw = self.rfile.read(n) if n else b"{}"
        try:
            obj = json.loads(raw or b"{}")
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None

    # ---- routing ----
    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._err(403, "bad host")
        u = urlparse(self.path)
        if u.path == "/":
            k = parse_qs(u.query).get("k", [""])[0]
            if not hmac.compare_digest(k, self.session_key):
                return self._send(403, "Open the URL printed in the terminal (it contains the session key).".encode(), "text/plain; charset=utf-8")
            html = self.index_html.replace("__SESSION_KEY__", self.session_key)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if not u.path.startswith("/api/"):
            return self._err(404, "not found")
        if not (self._key_ok() and self._origin_ok()):
            return self._err(403, "forbidden")
        q = parse_qs(u.query)
        try:
            if u.path == "/api/state":
                return self._json(200, api_state(self.state))
            if u.path == "/api/job":
                since = int(q.get("since", ["0"])[0])
                return self._json(200, api_job(self.state, since))
            if u.path == "/api/fs":
                return self._json(200, api_fs(q.get("path", [""])[0]))
        except Exception as e:  # noqa: BLE001
            return self._err(500, C.redact(repr(e), [self.state._token or ""]))
        return self._err(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._err(403, "bad host")
        u = urlparse(self.path)
        if not (u.path.startswith("/api/") and self._key_ok() and self._origin_ok()):
            return self._err(403, "forbidden")
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._err(415, "JSON only")
        body = self._body()
        if body is None:
            return self._err(400, "bad request body")
        st = self.state
        try:
            if u.path == "/api/dest/check":
                return self._json(200, api_dest_check(str(body.get("path", ""))))
            if u.path == "/api/fs/mkdir":
                return self._json(200, api_mkdir(str(body.get("parent", "")), str(body.get("name", ""))))
            if u.path == "/api/token":
                return self._json(*api_token_set(st, body))
            if u.path == "/api/token/clear":
                st.clear_token("Token cleared.")
                return self._json(200, {"ok": True})
            if u.path == "/api/start":
                return self._json(*api_start(st, body))
            if u.path == "/api/stop":
                return self._json(200, api_stop(st))
            if u.path == "/api/shutdown":
                api_stop(st)
                st.clear_token("")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return self._json(200, {"ok": True})
        except Exception as e:  # noqa: BLE001
            return self._err(500, C.redact(repr(e), [st._token or ""]))
        return self._err(404, "not found")


# ---------------------------------------------------------------- API implementations
def api_state(st: State) -> dict[str, Any]:
    man = st.manifest
    return {
        "ok": True,
        "groups": man.get("groups", {}),
        "default_group": man.get("default_group"),
        "models": [{k: m.get(k) for k in ("repo_id", "revision", "role", "license", "size_bytes", "groups", "priority", "file_count",
                                          "quant", "requires_token", "notes", "gguf_variants", "base_bytes")}
                   for m in man["models"]],
        "default_dest": st.default_dest,
        "env": {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count() or 1,
                "huggingface_hub": _pkg_version("huggingface_hub"), "hf_xet": _pkg_version("hf_xet"),
                "hf_transfer": _pkg_version("hf_transfer")},
        "token": st.token_status(),
    }


def api_fs(path: str) -> dict[str, Any]:
    shortcuts = [{"name": "Home", "path": str(Path.home())}]
    for cand in ("/Volumes", "/mnt", "/data", "/media", "/srv"):
        if Path(cand).is_dir():
            shortcuts.append({"name": cand, "path": cand})
    if os.name == "nt":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            if Path(f"{letter}:\\").exists():
                shortcuts.append({"name": f"{letter}:\\", "path": f"{letter}:\\"})
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_absolute():
        p = Path.home() / p
    p = p.resolve()
    while not p.is_dir() and p.parent != p:
        p = p.parent
    dirs: list[dict[str, str]] = []
    err = ""
    try:
        with os.scandir(p) as it:
            for e in it:
                try:
                    if e.is_dir():
                        dirs.append({"name": e.name, "path": str(Path(e.path))})
                except OSError:
                    continue
    except OSError as e:
        err = str(e)
    dirs.sort(key=lambda d: d["name"].lower())
    return {"ok": True, "path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs[:1000], "truncated": len(dirs) > 1000, "writable": os.access(p, os.W_OK),
            "shortcuts": shortcuts, "error": err}


def api_mkdir(parent: str, name: str) -> dict[str, Any]:
    if not name or name in (".", "..") or any(c in name for c in '/\\:*?"<>|\0'):
        return {"ok": False, "error": "invalid folder name"}
    p = Path(parent).expanduser()
    if not p.is_absolute() or not p.is_dir():
        return {"ok": False, "error": "parent folder not found"}
    target = p / name
    try:
        target.mkdir(exist_ok=True)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(target.resolve())}


def api_dest_check(path: str) -> dict[str, Any]:
    if not path.strip():
        return {"ok": False, "error": "empty path"}
    p = Path(path).expanduser()
    if not p.is_absolute():
        return {"ok": False, "error": "use an absolute path"}
    p = p.resolve()
    if p.exists() and not p.is_dir():
        return {"ok": False, "error": "that path is a file"}
    probe = C.nearest_existing_parent(p)
    usage = shutil.disk_usage(probe)
    return {"ok": True, "path": str(p), "exists": p.exists(), "writable": os.access(probe, os.W_OK),
            "free": usage.free, "total": usage.total, "probe": str(probe)}


def api_token_set(st: State, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    token = str(body.get("token", "")).strip()
    if not TOKEN_RE.match(token):
        return 400, {"ok": False, "error": "That does not look like a Hugging Face token (expected hf_ followed by letters/digits)."}
    ttl = int(body.get("ttl_minutes", 240))
    ttl = max(0, min(ttl, 7 * 24 * 60))
    try:
        info = whoami(token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return 400, {"ok": False, "error": "Hugging Face rejected this token (401/403). It is not stored."}
        return 502, {"ok": False, "error": f"Token check failed: HTTP {e.code}. It is not stored."}
    except Exception as e:  # noqa: BLE001
        return 502, {"ok": False, "error": f"Could not reach Hugging Face to check the token ({type(e).__name__}). It is not stored."}
    st.set_token(token, info, ttl, bool(body.get("clear_on_finish", True)))
    return 200, {"ok": True, "info": info, "token": st.token_status()}


def _int(body: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(body.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def api_start(st: State, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    kind = str(body.get("action", ""))
    if kind not in SCRIPTS:
        return 400, {"ok": False, "error": "unknown action"}
    dest_chk = api_dest_check(str(body.get("dest", "")))
    if not dest_chk["ok"]:
        return 400, dest_chk
    dest = dest_chk["path"]
    by_id = {m["repo_id"]: m for m in st.manifest["models"]}
    repo_ids = body.get("models") or []
    if not isinstance(repo_ids, list) or not all(isinstance(r, str) and r in by_id for r in repo_ids):
        return 400, {"ok": False, "error": "unknown model in selection"}
    models = [by_id[r] for r in sorted(set(repo_ids), key=lambda r: by_id[r].get("priority", 999))]
    if kind.startswith("checksum"):
        if not Path(dest).is_dir():
            return 400, {"ok": False, "error": "destination folder does not exist yet"}
    else:
        if not models:
            return 400, {"ok": False, "error": "select at least one model"}
        if kind != "dry_run" and kind.startswith("verify") and not Path(dest).is_dir():
            return 400, {"ok": False, "error": "destination folder does not exist yet"}
        if kind in ("download",) and not dest_chk["writable"]:
            return 400, {"ok": False, "error": "destination is not writable"}

    # file selection for multi-file GGUF repos: {"org/repo": ["variant id", ...]}; omitted = all files
    raw_var = body.get("variants") or {}
    if not isinstance(raw_var, dict):
        return 400, {"ok": False, "error": "bad file selection"}
    picked_ids = {m["repo_id"] for m in models}
    variant_argv: list[str] = []
    expected: dict[str, int] = {}
    for rid, ids in raw_var.items():
        if rid not in picked_ids or not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return 400, {"ok": False, "error": "bad file selection"}
    for m in models:
        chosen = raw_var.get(m["repo_id"])
        try:
            variants, _skip = C.resolve_variants(m, chosen)
        except ValueError as e:
            return 400, {"ok": False, "error": str(e)}
        expected[m["repo_id"]] = C.expected_bytes(m, variants)
        if chosen is not None and m.get("gguf_variants"):
            variant_argv += [a for v in variants for a in ("--variant", f"{m['repo_id']}={v['id']}")]
    models = [dict(m, _expected=expected[m["repo_id"]]) for m in models]

    if kind in NEEDS_TOKEN:
        gated = [m["repo_id"] for m in models if m.get("requires_token")]
        if gated and not st.token_available():
            return 400, {"ok": False, "error": "Gated model(s) selected: " + ", ".join(gated)
                         + ". Accept the licence on the model page with your Hugging Face account, then set a token in section 3"
                         + " (or log in once with `hf auth login`)."}

    script = str(HERE / SCRIPTS[kind])
    argv = [sys.executable, "-u", script]
    use_token = False
    if kind in ("download", "dry_run"):
        argv += ["--events", "--dest", dest,
                 "--max-workers", str(_int(body, "max_workers", 8, 1, 64)),
                 "--retries", str(_int(body, "retries", 5, 1, 20))]
        for m in models:
            argv += ["--only", m["repo_id"]]
        argv += variant_argv
        if kind == "dry_run":
            argv.append("--dry-run")
        elif body.get("high_performance"):
            argv.append("--high-performance")
    elif kind in ("verify", "verify_quick"):
        argv += ["--events", "--dest", dest, "--workers", str(_int(body, "workers", 8, 1, 32))]
        for m in models:
            argv += ["--only", m["repo_id"]]
        argv += variant_argv
        if kind == "verify_quick":
            argv.append("--quick")
    elif kind == "checksum_generate":
        argv += ["generate", dest, "--events", "--workers", str(_int(body, "workers", 8, 1, 32))]
    else:
        argv += ["verify", dest, "--events", "--workers", str(_int(body, "workers", 8, 1, 32))]

    token = st.get_token() if kind in NEEDS_TOKEN else None
    if token:
        argv.append("--token-stdin")
        use_token = True

    with st.lock:
        if st.job and st.job.status == "running":
            return 409, {"ok": False, "error": "another job is running"}
        job = Job(kind, dest, models, argv[3:])
        st.job = job

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # the GUI token (if any) goes through stdin, never through the environment
    if use_token:
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
    try:
        proc = subprocess.Popen(argv, cwd=str(HERE), env=env, stdin=subprocess.PIPE if use_token else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    except OSError as e:
        job.status, job.ended = "failed", time.time()
        job.add_log(f"cannot start worker: {e}")
        return 500, {"ok": False, "error": f"cannot start worker: {e}"}
    job.proc = proc
    if use_token and proc.stdin:
        try:
            proc.stdin.write(token + "\n")
            proc.stdin.flush()
            proc.stdin.close()
        except OSError:
            pass
    threading.Thread(target=_reader, args=(st, job, token), daemon=True).start()
    if kind in ("download",):
        threading.Thread(target=_sampler, args=(job, dest), daemon=True).start()
    return 200, {"ok": True, "job": job.id}


def _reader(st: State, job: Job, token: str | None) -> None:
    secrets_ = [token] if token else []
    assert job.proc and job.proc.stdout
    for raw in job.proc.stdout:
        line = C.redact(raw.rstrip("\r\n"), secrets_)
        if line.startswith(C.EVENT_PREFIX):
            try:
                _apply_event(job, json.loads(line[len(C.EVENT_PREFIX):]))
            except ValueError:
                pass
            continue
        with st.lock:
            job.add_log(line)
    rc = job.proc.wait()
    with st.lock:
        job.exit_code = rc
        job.ended = time.time()
        job.status = "stopped" if job.stop_requested else ("finished" if rc == 0 else "failed")
        if job.kind == "download":
            # the sampler stops once the job is no longer running, so settle the final numbers here
            for mm in job.models.values():
                if mm["state"] == "done":
                    mm["bytes"] = mm["size"]
            job.bytes_total = sum(m["size"] for m in job.models.values())
            job.bytes_done = sum(m["bytes"] for m in job.models.values())
        elif job.status == "finished" and job.bytes_total:
            job.bytes_done = job.bytes_total
        if token and st.token_clear_on_finish and job.kind in NEEDS_TOKEN:
            st.clear_token("Token was cleared automatically after the job finished.")


def _apply_event(job: Job, ev: dict[str, Any]) -> None:
    name = ev.get("event")
    rid = ev.get("repo_id")
    mm = job.models.get(rid) if rid else None
    if name == "plan":
        job.final["plan"] = {k: ev.get(k) for k in ("payload", "remaining", "free")}
    elif name == "model_start" and mm:
        mm["state"], mm["note"] = "downloading", ""
    elif name == "model_retry" and mm:
        mm["state"], mm["note"] = "retrying", f"attempt {ev.get('attempt')} failed, waiting {int(ev.get('wait', 0))}s"
    elif name == "model_done" and mm:
        mm["state"] = "done"
        state = ev.get("size_state") or ("ok" if ev.get("size_match", True) else "short")
        mm["note"] = {"ok": "", "extra": "folder also holds other files (e.g. variants fetched earlier)",
                      "short": "size differs from manifest"}.get(state, "")
        mm["bytes"] = mm["size"]
    elif name == "model_failed" and mm:
        mm["state"], mm["note"] = "failed", str(ev.get("error", ""))[:300]
    elif name in ("verify_start", "checksum_start"):
        job.bytes_total = int(ev.get("bytes_total", 0))
        job.bytes_done = 0
        for m in ev.get("models", []) or []:
            if m["repo_id"] in job.models:
                job.models[m["repo_id"]]["state"] = "verifying"
    elif name in ("verify_progress", "checksum_progress"):
        job.bytes_done = int(ev.get("bytes_done", 0))
        job.bytes_total = int(ev.get("bytes_total", job.bytes_total))
        job.speed_samples.append((time.time(), job.bytes_done))
    elif name == "verify_model_done" and mm:
        mm["state"] = "verified" if ev.get("ok") else "mismatch"
        mm["note"] = f"{ev.get('bad')} bad file(s)" if not ev.get("ok") else (f"{ev.get('extra')} extra local file(s)" if ev.get("extra") else "")
    elif name == "verify_file_bad":
        pass  # the worker also prints the line to the log
    elif name == "done":
        job.final["done"] = {k: v for k, v in ev.items() if k not in ("event", "t")}


def _sampler(job: Job, dest: str) -> None:
    """Estimate download progress from bytes on disk (cheap: a few hundred files)."""
    root = Path(dest)
    while job.status == "running":
        grand = 0
        for mm in job.models.values():
            if mm["state"] == "done":
                used = mm["size"]
            elif mm["state"] in ("downloading", "retrying", "failed"):
                used = min(mm["size"], C.dir_usage(root / mm["repo_id"]))
            else:
                used = min(mm["size"], C.dir_usage(root / mm["repo_id"])) if (root / mm["repo_id"]).exists() else 0
            grand += used
            if job.status == "running":
                mm["bytes"] = used
        if job.status != "running":
            break
        job.bytes_done = grand
        job.bytes_total = sum(m["size"] for m in job.models.values())
        job.speed_samples.append((time.time(), grand))
        time.sleep(2.0)


def api_stop(st: State) -> dict[str, Any]:
    with st.lock:
        job = st.job
        if not job or job.status != "running" or not job.proc:
            return {"ok": True, "stopped": False}
        job.stop_requested = True
        proc = job.proc
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
    return {"ok": True, "stopped": True}


def api_job(st: State, since: int) -> dict[str, Any]:
    with st.lock:
        job = st.job
        tok = st.token_status()
        if not job:
            return {"ok": True, "job": None, "token": tok}
        samples = list(job.speed_samples)
        speed = 0.0
        if len(samples) >= 2:
            t0, b0 = samples[max(0, len(samples) - 12)]
            t1, b1 = samples[-1]
            if t1 > t0:
                speed = max(0.0, (b1 - b0) / (t1 - t0))
        remaining = max(0, job.bytes_total - job.bytes_done)
        eta = int(remaining / speed) if speed > 1 and job.status == "running" else None
        start = max(0, since - job.log_base)
        lines = list(job.log)[start:]
        return {"ok": True, "token": tok, "job": {
            "id": job.id, "kind": job.kind, "status": job.status, "dest": job.dest,
            "started": job.started, "ended": job.ended, "exit_code": job.exit_code,
            "models": list(job.models.values()), "bytes_done": job.bytes_done, "bytes_total": job.bytes_total,
            "speed_bps": speed, "eta_s": eta, "final": job.final,
            "log": lines, "log_next": job.log_base + len(job.log)}}


def _open_browser(url: str) -> None:
    """Open the GUI page in the default browser shortly after the server starts listening."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("BROWSER")):
        # headless box: webbrowser would fall back to a text browser inside this terminal
        print("  (no display detected, so no browser was opened; use the SSH port forward above and open the URL on your own PC)", flush=True)
        return

    def run() -> None:
        time.sleep(0.5)
        try:
            ok = webbrowser.open(url)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            print("  (could not open a browser automatically; open the URL above by hand)", flush=True)

    threading.Thread(target=run, daemon=True).start()


def _ttl_watcher(st: State) -> None:
    while True:
        time.sleep(15)
        st.get_token()  # triggers lazy expiry


def main() -> int:
    ap = argparse.ArgumentParser(description="Local GUI for the HF model download bundle")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    ap.add_argument("--default-dest", default=os.environ.get("DEST", str(Path.home() / "hf_models")))
    args = ap.parse_args()

    if not INDEX_PATH.is_file():
        print(f"ERROR: GUI file not found: {INDEX_PATH}", file=sys.stderr)
        return 1
    manifest = C.load_manifest()
    state = State(manifest, args.default_dest)

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((HOST, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        print(f"ERROR: no free port in {args.port}..{args.port + 19}", file=sys.stderr)
        return 1
    port = server.server_address[1]
    key = secrets.token_urlsafe(24)
    Handler.state = state
    Handler.session_key = key
    Handler.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    Handler.index_html = INDEX_PATH.read_text(encoding="utf-8")
    threading.Thread(target=_ttl_watcher, args=(state,), daemon=True).start()

    url = f"http://127.0.0.1:{port}/?k={key}"
    print("HF model download GUI")
    print(f"  open : {url}")
    print(f"  bind : {HOST}:{port} only (not reachable from other machines)")
    print("  remote machine? use:  ssh -L {0}:127.0.0.1:{0} <user>@<host>   then open the URL above locally".format(port))
    print("  stop : Ctrl+C or the 'Quit' button in the page (clears the token)\n", flush=True)
    if not args.no_browser:
        _open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        api_stop(state)
        state.clear_token("")
        server.server_close()
    print("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
