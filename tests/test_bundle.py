"""Tests for the HF model download bundle.

Run (needs internet; downloads ~45 MB of tiny public models into a temp dir):
    python -m unittest discover -s tests -v

The suite copies the bundle to a temp dir and swaps in a tiny manifest, so the real multi-hundred-GB
manifest is never downloaded.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE))
import hfdl_common as C  # noqa: E402

TINY_A = {"repo_id": "prajjwal1/bert-tiny", "revision": "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
          "role": "test_a", "license": "mit", "size_bytes": 17991091, "groups": ["t1", "t2"], "priority": 1}
TINY_B = {"repo_id": "hf-internal-testing/tiny-random-bert", "revision": "f171d7baecaf37b5da5a3616d8833b9969753535",
          "role": "test_b", "license": "apache-2.0", "size_bytes": 28183891, "groups": ["t2"], "priority": 2}
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"
# tiny public GGUF repo with 7 single-file variants (Q2_K ... fp16), about 21 MB in total
TINY_GGUF_ID = "afrideva/tiny-random-LlamaForCausalLM-GGUF"
TINY_GGUF_REV = "a83b3e6463c0925f562aa9d4dd76563ca51551aa"
# Every subprocess gets an empty HF_HOME so a real `hf auth login` on the test machine cannot leak in.
EMPTY_HF_HOME = tempfile.mkdtemp(prefix="hfdl_hfhome_")
_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_TOKEN_PATH", "HF_OIDC_RESOURCE")


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _TOKEN_ENV}
    env.update(PYTHONIOENCODING="utf-8", HF_HOME=EMPTY_HF_HOME)
    env.update(extra)
    return env


def fake_login_home() -> str:
    """An HF_HOME that looks like `hf auth login` was run with FAKE_TOKEN."""
    d = tempfile.mkdtemp(prefix="hfdl_login_")
    Path(d, "token").write_text(FAKE_TOKEN, encoding="utf-8")
    return d


_TINY_GGUF: dict | None = None


def tiny_gguf_entry(groups: list[str], priority: int) -> dict:
    """Manifest entry for TINY_GGUF_ID, with variants built by the same code the dev script uses."""
    global _TINY_GGUF
    if _TINY_GGUF is None:
        url = f"https://huggingface.co/api/models/{TINY_GGUF_ID}/revision/{TINY_GGUF_REV}?blobs=true"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hfdl-tests"}), timeout=60) as r:
            d = json.loads(r.read())
        files = [(s["rfilename"], s.get("size") or 0) for s in d["siblings"]]
        variants = C.group_gguf_variants(files)
        total = sum(sz for _, sz in files)
        _TINY_GGUF = {"repo_id": TINY_GGUF_ID, "revision": TINY_GGUF_REV, "role": "test_gguf", "license": "-", "quant": "GGUF",
                      "size_bytes": total, "gguf_variants": variants,
                      "base_bytes": total - sum(v["size_bytes"] for v in variants)}
    return dict(_TINY_GGUF, groups=groups, priority=priority)


def variant(entry: dict, label: str) -> dict:
    return next(v for v in entry["gguf_variants"] if v["label"] == label)


def make_manifest(models, path: Path) -> None:
    path.write_text(json.dumps({"default_group": "t1", "groups": {"t1": "a", "t2": "a+b", "t3": "x", "t4": "gguf"},
                                "models": models}), encoding="utf-8")


def run(script: str, *args: str, stdin: str | None = None, cwd: Path = BUNDLE,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(cwd / script), *args], input=stdin, capture_output=True, text=True,
                          encoding="utf-8", env=env or clean_env(), timeout=600, cwd=str(cwd))


def events(out: str) -> list[dict]:
    return [json.loads(l[len(C.EVENT_PREFIX):]) for l in out.splitlines() if l.startswith(C.EVENT_PREFIX)]


class Unit(unittest.TestCase):
    def test_redact(self):
        self.assertNotIn(FAKE_TOKEN, C.redact(f"x {FAKE_TOKEN} y", [FAKE_TOKEN]))
        self.assertNotIn(FAKE_TOKEN, C.redact(f"Bearer {FAKE_TOKEN}"))  # pattern based even if not registered

    def test_safe_join(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNotNone(C.safe_join(Path(d), "a/b.txt"))
            self.assertIsNone(C.safe_join(Path(d), "../evil"))
            self.assertIsNone(C.safe_join(Path(d), "a/../../evil"))

    def test_payload_files_skip_nested_cache_and_meta(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "org/m/.cache/huggingface").mkdir(parents=True)
            (r / "org/m/.cache/huggingface/x.metadata").write_text("m")
            (r / "org/m/w.bin").write_text("w")
            (r / C.SUMS_NAME).write_text("s")
            (r / C.RECEIPT_NAME).write_text("r")
            got = sorted(p.relative_to(r).as_posix() for p in C.iter_payload_files(r))
            self.assertEqual(got, ["org/m/w.bin"])

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            make_manifest([dict(TINY_A, revision="main")], p)
            with self.assertRaises(ValueError):
                C.load_manifest(p)
            make_manifest([dict(TINY_A, repo_id="../../etc/passwd")], p)
            with self.assertRaises(ValueError):
                C.load_manifest(p)

    def test_shipped_manifest_loads_and_group_sums(self):
        man = C.load_manifest()
        sums = {g: sum(m["size_bytes"] for m in man["models"] if g in m["groups"]) for g in man["groups"]}
        self.assertTrue(sums["all"] > sums["no_coder"] > sums["core4"] > sums["core3"] > 0, sums)

    def test_shipped_manifest_structure(self):
        man = C.load_manifest()
        ids = [m["repo_id"] for m in man["models"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate repo_id")
        prios = [m["priority"] for m in man["models"]]
        self.assertEqual(len(prios), len(set(prios)), "duplicate priority")
        for m in man["models"]:
            self.assertGreater(m["size_bytes"], 0, m["repo_id"])
            self.assertTrue(m["groups"], m["repo_id"])
            for g in m["groups"]:
                self.assertIn(g, man["groups"], f"{m['repo_id']} uses undefined group {g}")
            self.assertTrue(m.get("quant"), f"{m['repo_id']} has no quant label")
            if m.get("gated"):
                self.assertTrue(m.get("requires_token"), f"{m['repo_id']} is gated but not marked requires_token")
        for g in man["groups"]:
            self.assertTrue(any(g in m["groups"] for m in man["models"]), f"group {g} is empty")

    def test_shipped_manifest_gguf_variants(self):
        man = C.load_manifest()
        multi = [m for m in man["models"] if m.get("gguf_variants")]
        self.assertGreaterEqual(len(multi), 2, "the Swift and Bolt-74B GGUF repos have several files")
        for m in multi:
            vs = m["gguf_variants"]
            self.assertGreaterEqual(len(vs), 2, m["repo_id"])
            self.assertEqual(m["base_bytes"] + sum(v["size_bytes"] for v in vs), m["size_bytes"], m["repo_id"])
            self.assertEqual(len({v["id"] for v in vs}), len(vs), m["repo_id"])
            self.assertEqual(len({v["label"] for v in vs}), len(vs), m["repo_id"])
            for v in vs:
                self.assertTrue(v["files"] and all(f.endswith(".gguf") for f in v["files"]), (m["repo_id"], v))

    def test_group_gguf_variants(self):
        files = [("README.md", 10), ("model-Q4_K_M-00001-of-00002.gguf", 100), ("model-Q4_K_M-00002-of-00002.gguf", 50),
                 ("model-Q8_0.gguf", 300), ("sub/model-IQ2.gguf", 40), ("mmproj-model-f16.gguf", 7)]
        vs = C.group_gguf_variants(files)
        by_id = {v["id"]: v for v in vs}
        self.assertEqual(set(by_id), {"model-Q4_K_M", "model-Q8_0", "sub/model-IQ2", "mmproj-model-f16"})
        self.assertEqual(by_id["model-Q4_K_M"]["files"], ["model-Q4_K_M-00001-of-00002.gguf", "model-Q4_K_M-00002-of-00002.gguf"])
        self.assertEqual(by_id["model-Q4_K_M"]["size_bytes"], 150)
        self.assertEqual(C.group_gguf_variants([("a.gguf", 1), ("README.md", 1)]), [], "a single variant is not selectable")
        # the shared prefix is cut back to a separator, so labels stay readable
        labels = sorted(v["label"] for v in C.group_gguf_variants([("x-IQ2_XS-mtp.gguf", 1), ("x-IQ3_S-mtp.gguf", 1)]))
        self.assertEqual(labels, ["IQ2_XS-mtp", "IQ3_S-mtp"])

    def test_resolve_variants(self):
        m = {"repo_id": "o/r", "size_bytes": 160, "base_bytes": 10, "gguf_variants": [
            {"id": "r-Q2", "label": "Q2", "files": ["r-Q2.gguf"], "size_bytes": 50},
            {"id": "r-Q8", "label": "Q8", "files": ["r-Q8-00001-of-00002.gguf", "r-Q8-00002-of-00002.gguf"], "size_bytes": 100}]}
        picked, skip = C.resolve_variants(m, None)
        self.assertEqual(len(picked), 2)
        self.assertEqual(skip, [])
        picked, skip = C.resolve_variants(m, ["Q2"])  # by label
        self.assertEqual([v["id"] for v in picked], ["r-Q2"])
        self.assertEqual(skip, ["r-Q8-00001-of-00002.gguf", "r-Q8-00002-of-00002.gguf"])
        self.assertEqual(C.expected_bytes(m, picked), 60)
        self.assertEqual(C.resolve_variants(m, ["r-Q8"])[0][0]["label"], "Q8")  # by id
        with self.assertRaises(ValueError):
            C.resolve_variants(m, [])
        with self.assertRaises(ValueError):
            C.resolve_variants(m, ["Q5"])
        with self.assertRaises(ValueError):
            C.resolve_variants({"repo_id": "o/plain", "size_bytes": 1}, ["Q2"])
        self.assertEqual(C.expected_bytes({"repo_id": "o/plain", "size_bytes": 7}, []), 7)

    def test_parse_variant_args_and_escape(self):
        import fnmatch
        self.assertEqual(C.parse_variant_args(["o/r=Q2", "o/r=Q8", "x/y=A"]), {"o/r": ["Q2", "Q8"], "x/y": ["A"]})
        for bad in (["o/r"], ["=Q2"], ["o/r="]):
            with self.assertRaises(SystemExit):
                C.parse_variant_args(bad)
        name = "model[v1]*?.gguf"
        self.assertTrue(fnmatch.fnmatchcase(name, C.fnmatch_escape(name)))
        self.assertFalse(fnmatch.fnmatchcase("modelv.gguf", C.fnmatch_escape(name)))


class AmbientToken(unittest.TestCase):
    """The token huggingface_hub uses on its own (`hf auth login`, HF_TOKEN) must be reported, never shown."""

    PROBE = ("import json, sys; sys.path.insert(0, sys.argv[1]); import hfdl_common as C, gui_server as G; "
             "st = G.State({'models': []}, '.'); s = st.token_status(); "
             "print(json.dumps({'common': C.ambient_token()[1], 'status': s, 'available': st.token_available()}))")

    def probe(self, env):
        r = subprocess.run([sys.executable, "-c", self.PROBE, str(BUNDLE)], capture_output=True, text=True,
                           encoding="utf-8", env=env, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(FAKE_TOKEN, r.stdout + r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_sources(self):
        none = self.probe(clean_env())
        self.assertIsNone(none["common"])
        self.assertIsNone(none["status"]["ambient"])
        self.assertFalse(none["available"])
        login = self.probe(clean_env(HF_HOME=fake_login_home()))
        self.assertEqual(login["common"], "login")
        self.assertEqual(login["status"]["ambient"], "login")
        self.assertTrue(login["available"])
        self.assertFalse(login["status"]["set"])
        env = self.probe(clean_env(HF_TOKEN=FAKE_TOKEN))
        self.assertEqual(env["status"]["ambient"], "env")
        off = self.probe(clean_env(HF_HOME=fake_login_home(), HF_HUB_DISABLE_IMPLICIT_TOKEN="1"))
        self.assertIsNone(off["status"]["ambient"], "a saved login is not used when implicit tokens are disabled")


class BrowserOpen(unittest.TestCase):
    def test_open_browser_rules(self):
        import io
        import contextlib
        import webbrowser
        import gui_server

        calls: list[str] = []
        real_open, real_platform, real_env = webbrowser.open, sys.platform, dict(os.environ)
        webbrowser.open = lambda u: calls.append(u) or True
        try:
            sys.platform = "linux"
            for k in ("DISPLAY", "WAYLAND_DISPLAY", "BROWSER"):
                os.environ.pop(k, None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gui_server._open_browser("http://127.0.0.1:1/?k=a")
            time.sleep(1.0)
            self.assertEqual(calls, [], "headless Linux must not launch a (text) browser")
            self.assertIn("no display", buf.getvalue())
            os.environ["DISPLAY"] = ":0"
            gui_server._open_browser("http://127.0.0.1:1/?k=b")
            time.sleep(1.0)
            self.assertEqual(calls, ["http://127.0.0.1:1/?k=b"])
        finally:
            webbrowser.open, sys.platform = real_open, real_platform
            os.environ.clear()
            os.environ.update(real_env)


class Workflow(unittest.TestCase):
    """End-to-end on two tiny public repos."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hfdl_test_"))
        cls.bundle = cls.tmp / "bundle"
        shutil.copytree(BUNDLE, cls.bundle, ignore=shutil.ignore_patterns("tests", ".venv", "__pycache__", "hf_models"))
        cls.gguf = tiny_gguf_entry(["t4"], 3)
        make_manifest([TINY_A, TINY_B, cls.gguf], cls.bundle / "model_manifest.json")
        cls.dest = cls.tmp / "models"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_dry_run_has_no_side_effects(self):
        ghost = self.tmp / "ghost" / "deep"
        r = run("download_models.py", "--dry-run", "--dest", str(ghost), "--group", "t2", "--events", cwd=self.bundle)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(ghost.exists(), "dry-run must not create the destination")
        self.assertTrue(any(e["event"] == "plan" for e in events(r.stdout)))

    def test_02_download_and_resume(self):
        r = run("download_models.py", "--dest", str(self.dest), "--group", "t2", "--events", cwd=self.bundle)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ev = events(r.stdout)
        done = [e for e in ev if e["event"] == "model_done"]
        self.assertEqual(len(done), 2)
        self.assertTrue(all(e["size_match"] for e in done), done)
        self.assertTrue((self.dest / TINY_A["repo_id"] / "config.json").is_file())
        # nested .cache exists, as documented by huggingface_hub
        self.assertTrue((self.dest / TINY_A["repo_id"] / ".cache").exists())
        # second run is a cheap no-op and still succeeds
        r2 = run("download_models.py", "--dest", str(self.dest), "--group", "t2", "--events", cwd=self.bundle)
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        receipts = (self.dest / C.RECEIPT_NAME).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(receipts), 4)

    def test_03_hf_hash_verify_ok_then_detects_same_size_corruption(self):
        r = run("verify_hf_hashes.py", "--dest", str(self.dest), "--group", "t2", "--events", cwd=self.bundle)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # flip one byte in a weight file without changing its size
        target = next((self.dest / TINY_A["repo_id"]).glob("*.safetensors"), None) or next((self.dest / TINY_A["repo_id"]).glob("*.bin"))
        data = bytearray(target.read_bytes())
        data[len(data) // 2] ^= 0xFF
        target.write_bytes(bytes(data))
        quick = run("verify_hf_hashes.py", "--dest", str(self.dest), "--only", TINY_A["repo_id"], "--quick", cwd=self.bundle)
        self.assertEqual(quick.returncode, 0, "quick mode checks size only and should not notice")
        full = run("verify_hf_hashes.py", "--dest", str(self.dest), "--only", TINY_A["repo_id"], "--events", cwd=self.bundle)
        self.assertEqual(full.returncode, 1, full.stdout)
        self.assertTrue(any(e["event"] == "verify_file_bad" and target.name in e["file"] for e in events(full.stdout)))
        self.assertTrue((self.dest / C.VERIFY_REPORT_NAME).is_file())
        # restore: re-download repairs it
        target.unlink()
        again = run("download_models.py", "--dest", str(self.dest), "--only", TINY_A["repo_id"], cwd=self.bundle)
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        ok = run("verify_hf_hashes.py", "--dest", str(self.dest), "--only", TINY_A["repo_id"], cwd=self.bundle)
        self.assertEqual(ok.returncode, 0, ok.stdout)

    def test_04_checksums_roundtrip_and_cache_excluded(self):
        g = run("make_checksums.py", "generate", str(self.dest), cwd=self.bundle)
        self.assertEqual(g.returncode, 0, g.stdout + g.stderr)
        sums = (self.dest / C.SUMS_NAME).read_text(encoding="utf-8")
        self.assertNotIn("/.cache/", sums, "resume metadata must not be listed")
        self.assertNotIn(C.RECEIPT_NAME, sums)
        v = run("make_checksums.py", "verify", str(self.dest), cwd=self.bundle)
        self.assertEqual(v.returncode, 0, v.stdout)
        victim = next(p for p in C.iter_payload_files(self.dest) if p.stat().st_size > 10)
        victim.write_bytes(victim.read_bytes()[:-1] + b"\x00" if victim.read_bytes()[-1:] != b"\x00" else victim.read_bytes()[:-1] + b"\x01")
        bad = run("make_checksums.py", "verify", str(self.dest), cwd=self.bundle)
        self.assertEqual(bad.returncode, 1, bad.stdout)
        self.assertIn("MISMATCH", bad.stdout)
        # the output is a valid `sha256sum -c` file: hash, two spaces, ./relative path
        first = sums.splitlines()[0]
        self.assertRegex(first, r"^[0-9a-f]{64}  \./")

    def test_05_failure_isolation_and_no_retry_on_missing_repo(self):
        ghost = dict(TINY_A, repo_id="prajjwal1/this-repo-does-not-exist-xyz", priority=0, groups=["t3"])
        good = dict(TINY_A, priority=1, groups=["t3"])
        make_manifest([ghost, good], self.bundle / "model_manifest_fail.json")
        d = self.tmp / "failcase"
        t0 = time.time()
        r = run("download_models.py", "--manifest", str(self.bundle / "model_manifest_fail.json"), "--dest", str(d),
                "--group", "t3", "--events", "--retries", "5", "--retry-wait", "30", cwd=self.bundle)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertLess(time.time() - t0, 90, "a missing repo is fatal and must not wait through the backoff")
        ev = events(r.stdout)
        self.assertTrue(any(e["event"] == "model_failed" and "does-not-exist" in e["repo_id"] for e in ev))
        self.assertTrue(any(e["event"] == "model_done" and e["repo_id"] == good["repo_id"] for e in ev), "the good model must still download")

    def test_06_token_is_never_printed(self):
        d = self.tmp / "tokencase"
        r = run("download_models.py", "--dest", str(d), "--only", TINY_A["repo_id"], "--token-stdin", "--retries", "1",
                "--events", cwd=self.bundle, stdin=FAKE_TOKEN + "\n")
        blob = r.stdout + r.stderr
        self.assertNotIn(FAKE_TOKEN, blob)
        self.assertNotIn(FAKE_TOKEN[3:], blob)
        rec = d / C.RECEIPT_NAME
        if rec.exists():
            self.assertNotIn(FAKE_TOKEN, rec.read_text(encoding="utf-8"))

    def test_07_logged_in_token_is_detected_and_hidden(self):
        # `hf auth login` saved FAKE_TOKEN; huggingface_hub sends it on its own, so the log must say so
        # (it used to claim "anonymous") and must never print it. HF rejects the fake token, so the
        # download itself is expected to fail.
        d = self.tmp / "logincase"
        r = run("download_models.py", "--dest", str(d), "--only", TINY_A["repo_id"], "--retries", "1", "--events",
                cwd=self.bundle, env=clean_env(HF_HOME=fake_login_home()))
        blob = r.stdout + r.stderr
        self.assertIn("logged-in token", blob)
        self.assertNotIn("anonymous", blob)
        self.assertNotIn(FAKE_TOKEN, blob)
        self.assertNotIn(FAKE_TOKEN[3:], blob)
        self.assertTrue(any(e["event"] == "token" and e["source"] == "login" for e in events(r.stdout)))
        # without a saved login the same command reports anonymous (re-uses test_02's files, so it is a no-op)
        anon = run("download_models.py", "--dest", str(self.dest), "--only", TINY_A["repo_id"], "--events", cwd=self.bundle)
        self.assertEqual(anon.returncode, 0, anon.stdout + anon.stderr)
        self.assertIn("none (anonymous)", anon.stdout)

    def test_08_gguf_variant_download_and_verify(self):
        g = self.gguf
        q2, q8 = variant(g, "Q2_K"), variant(g, "Q8_0")
        d = self.tmp / "gguf"
        rid = g["repo_id"]
        dry = run("download_models.py", "--dry-run", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q2_K", "--events", cwd=self.bundle)
        self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
        plan = next(e for e in events(dry.stdout) if e["event"] == "plan")
        self.assertEqual(plan["payload"], g["base_bytes"] + q2["size_bytes"])
        self.assertEqual(plan["models"][0]["variants"], [q2["id"]])
        bad = run("download_models.py", "--dry-run", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q5_X", cwd=self.bundle)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unknown file variant", bad.stdout + bad.stderr)

        r = run("download_models.py", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q2_K", "--events", cwd=self.bundle)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        done = next(e for e in events(r.stdout) if e["event"] == "model_done")
        self.assertEqual(done["size_state"], "ok", done)
        local = d / rid
        ggufs = sorted(p.name for p in local.glob("*.gguf"))
        self.assertEqual(ggufs, q2["files"], "only the selected GGUF file is fetched")
        self.assertTrue(any(p.suffix != ".gguf" for p in C.iter_payload_files(local)), "README etc. are always fetched")
        self.assertEqual(C.payload_usage(local), g["base_bytes"] + q2["size_bytes"])
        rec = json.loads((d / C.RECEIPT_NAME).read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(rec["variants"], [q2["id"]])

        # verify: explicit selection passes; no selection checks what is on disk and skips the rest
        v1 = run("verify_hf_hashes.py", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q2_K", "--events", cwd=self.bundle)
        self.assertEqual(v1.returncode, 0, v1.stdout)
        v2 = run("verify_hf_hashes.py", "--dest", str(d), "--only", rid, "--events", cwd=self.bundle)
        self.assertEqual(v2.returncode, 0, v2.stdout)
        md = next(e for e in events(v2.stdout) if e["event"] == "verify_model_done")
        self.assertEqual(md["skipped_variants"], len(g["gguf_variants"]) - 1)
        self.assertEqual(md["extra"], 0)
        v3 = run("verify_hf_hashes.py", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q8_0", "--events", cwd=self.bundle)
        self.assertEqual(v3.returncode, 1, "an explicitly selected variant that is not on disk is missing")
        self.assertTrue(any(e["event"] == "verify_file_bad" and e["file"] in q8["files"] and e["reason"] == "missing"
                            for e in events(v3.stdout)))

        # add a second variant later: resume keeps the first, and the "extra" size state is reported
        r2 = run("download_models.py", "--dest", str(d), "--only", rid, "--variant", f"{rid}=Q8_0", "--events", cwd=self.bundle)
        self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
        done2 = next(e for e in events(r2.stdout) if e["event"] == "model_done")
        self.assertEqual(done2["size_state"], "extra", done2)
        self.assertEqual(sorted(p.name for p in local.glob("*.gguf")), sorted(q2["files"] + q8["files"]))


class Gui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hfdl_gui_"))
        cls.bundle = cls.tmp / "bundle"
        shutil.copytree(BUNDLE, cls.bundle, ignore=shutil.ignore_patterns("tests", ".venv", "__pycache__", "hf_models"))
        gated = dict(TINY_B, repo_id="fake-org/gated-model", priority=3, groups=["t2"], gated="manual", requires_token=True)
        cls.gguf = tiny_gguf_entry(["t4"], 4)
        make_manifest([TINY_A, TINY_B, gated, cls.gguf], cls.bundle / "model_manifest.json")
        cls.port = 18765
        env = clean_env()
        cls.proc = subprocess.Popen([sys.executable, "-u", str(cls.bundle / "gui_server.py"), "--no-browser", "--port", str(cls.port)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", cwd=str(cls.bundle), env=env)
        cls.url = None
        for line in cls.proc.stdout:
            if "open :" in line:
                cls.url = line.split("open :", 1)[1].strip()
                break
        assert cls.url, "server did not print its URL"
        cls.key = cls.url.split("?k=", 1)[1]
        cls.base = cls.url.split("/?k=", 1)[0]
        cls.hostport = cls.base.split("//", 1)[1]

    @classmethod
    def tearDownClass(cls):
        try:
            cls.call("POST", "/api/shutdown", {})
        except Exception:
            pass
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, key=True, host=None, origin=None):
        headers = {}
        if key:
            headers["X-Session-Key"] = cls.key
        if host:
            headers["Host"] = host
        if origin:
            headers["Origin"] = origin
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(cls.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            e.close()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw.decode("utf-8", "replace")}

    def wait_job(self, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, d = self.call("GET", "/api/job?since=0")
            j = d.get("job")
            if j and j["status"] != "running":
                return j
            time.sleep(1)
        self.fail("job did not finish")

    def test_01_access_control(self):
        req = urllib.request.Request(self.base + "/")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 403)
        with urllib.request.urlopen(self.url, timeout=10) as r:
            html = r.read().decode("utf-8")
        self.assertIn(self.key, html)
        self.assertNotIn("__SESSION_KEY__", html)
        self.assertEqual(self.call("GET", "/api/state", key=False)[0], 403)
        self.assertEqual(self.call("GET", "/api/state", host="evil.example:80")[0], 403)
        self.assertEqual(self.call("POST", "/api/token/clear", {}, origin="http://evil.example")[0], 403)
        self.assertEqual(self.call("GET", "/api/state")[0], 200)
        self.assertEqual(self.call("GET", "/gui/index.html")[0], 404)

    def test_02_token_validation_and_not_stored_when_rejected(self):
        code, d = self.call("POST", "/api/token", {"token": "not-a-token"})
        self.assertEqual(code, 400)
        code, d = self.call("POST", "/api/token", {"token": FAKE_TOKEN})
        self.assertIn(code, (400, 502), d)  # HF rejects it (400) or is unreachable (502); never stored
        _, st = self.call("GET", "/api/state")
        self.assertFalse(st["token"]["set"])
        self.assertNotIn(FAKE_TOKEN, json.dumps(d))

    def test_03_request_validation(self):
        dest = str(self.tmp / "dl")
        self.assertEqual(self.call("POST", "/api/start", {"action": "rm -rf", "dest": dest, "models": []})[0], 400)
        self.assertEqual(self.call("POST", "/api/start", {"action": "download", "dest": "relative/path", "models": [TINY_A["repo_id"]]})[0], 400)
        self.assertEqual(self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": ["evil/not-in-manifest"]})[0], 400)
        self.assertEqual(self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": []})[0], 400)
        self.assertEqual(self.call("POST", "/api/stop", {})[1].get("stopped"), False)

    def test_03b_gated_model_requires_token(self):
        dest = str(self.tmp / "gated_dl")
        code, d = self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": ["fake-org/gated-model"]})
        self.assertEqual(code, 400, d)
        self.assertIn("Gated", d["error"])
        self.assertFalse(Path(dest).exists())
        # a dry run needs no token and must still work
        code, d = self.call("POST", "/api/start", {"action": "dry_run", "dest": dest, "models": ["fake-org/gated-model"]})
        self.assertEqual(code, 200, d)
        self.assertEqual(self.wait_job()["status"], "finished")
        _, st = self.call("GET", "/api/state")
        flagged = {m["repo_id"]: m["requires_token"] for m in st["models"]}
        self.assertTrue(flagged["fake-org/gated-model"])
        self.assertFalse(flagged[TINY_A["repo_id"]])

    def test_04_fs_browser(self):
        code, d = self.call("GET", "/api/fs?path=" + urllib.request.quote(str(self.tmp)))
        self.assertTrue(d["ok"])
        self.assertIn("bundle", [x["name"] for x in d["dirs"]])
        code, d = self.call("POST", "/api/fs/mkdir", {"parent": str(self.tmp), "name": "../x"})
        self.assertFalse(d["ok"])
        code, d = self.call("POST", "/api/fs/mkdir", {"parent": str(self.tmp), "name": "newdir"})
        self.assertTrue(d["ok"])
        self.assertTrue((self.tmp / "newdir").is_dir())

    def test_05_download_and_verify_through_api(self):
        dest = str(self.tmp / "dl")
        code, d = self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": [TINY_A["repo_id"]],
                                                    "max_workers": 4, "retries": 2, "high_performance": True})
        self.assertEqual(code, 200, d)
        self.assertEqual(self.call("POST", "/api/start", {"action": "dry_run", "dest": dest, "models": [TINY_A["repo_id"]]})[0], 409)
        j = self.wait_job()
        self.assertEqual(j["status"], "finished", j["log"][-10:])
        self.assertEqual(j["models"][0]["state"], "done")
        self.assertEqual(j["bytes_done"], j["bytes_total"])
        code, d = self.call("POST", "/api/start", {"action": "verify", "dest": dest, "models": [TINY_A["repo_id"]]})
        self.assertEqual(code, 200, d)
        j = self.wait_job()
        self.assertEqual(j["status"], "finished", j["log"][-10:])
        self.assertEqual(j["models"][0]["state"], "verified")
        code, d = self.call("POST", "/api/start", {"action": "checksum_generate", "dest": dest})
        self.assertEqual(code, 200, d)
        self.assertEqual(self.wait_job()["status"], "finished")

    def test_05b_gguf_file_selection_through_api(self):
        g, rid = self.gguf, self.gguf["repo_id"]
        q4 = variant(g, "Q4_K_M")
        _, st = self.call("GET", "/api/state")
        sm = next(m for m in st["models"] if m["repo_id"] == rid)
        self.assertEqual(len(sm["gguf_variants"]), len(g["gguf_variants"]))
        self.assertEqual(sm["base_bytes"], g["base_bytes"])
        self.assertIsNone(st["token"]["ambient"])
        dest = str(self.tmp / "gguf_dl")
        for bad in ({rid: []}, {rid: ["nope"]}, {rid: "Q2_K"}, {TINY_A["repo_id"]: ["x"]}, {"other/repo": ["Q2_K"]}, ["Q2_K"]):
            code, d = self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": [rid], "variants": bad})
            self.assertEqual(code, 400, (bad, d))
        self.assertFalse(Path(dest).exists())
        code, d = self.call("POST", "/api/start", {"action": "download", "dest": dest, "models": [rid], "variants": {rid: [q4["id"]]}})
        self.assertEqual(code, 200, d)
        j = self.wait_job()
        self.assertEqual(j["status"], "finished", j["log"][-10:])
        self.assertEqual(j["bytes_total"], g["base_bytes"] + q4["size_bytes"], "progress uses the selected size")
        self.assertEqual(j["bytes_done"], j["bytes_total"])
        self.assertEqual(sorted(p.name for p in (Path(dest) / rid).glob("*.gguf")), q4["files"])
        code, d = self.call("POST", "/api/start", {"action": "verify", "dest": dest, "models": [rid], "variants": {rid: [q4["id"]]}})
        self.assertEqual(code, 200, d)
        j = self.wait_job()
        self.assertEqual(j["status"], "finished", j["log"][-10:])
        self.assertEqual(j["models"][0]["state"], "verified")

    def test_06_stop_kills_the_worker(self):
        big = self.tmp / "bigdir"
        big.mkdir()
        with (big / "sparse.bin").open("wb") as f:
            f.truncate(12 * 1024 ** 3)  # sparse: cheap to create, takes seconds to hash
        code, d = self.call("POST", "/api/start", {"action": "checksum_generate", "dest": str(big), "workers": 1})
        self.assertEqual(code, 200, d)
        time.sleep(1.0)
        code, d = self.call("POST", "/api/stop", {})
        self.assertTrue(d["stopped"])
        j = self.wait_job(30)
        self.assertEqual(j["status"], "stopped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
