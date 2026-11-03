# hf-model-downloader

A small local tool that downloads a set of pinned [Hugging Face](https://huggingface.co) model snapshots in one go, with a browser GUI (or a plain CLI), resumable downloads and hash verification against the hashes Hugging Face publishes.

I wrote this for my own use when setting up a DGX box, and I'm putting it out as is. Use it however you like. (DGX をセットアップするときに個人的に使っているツールです。好きに使ってください。)

> Unofficial. This project is not affiliated with, endorsed by, or sponsored by Hugging Face, Inc. "Hugging Face" is a trademark of its owner.

日本語の詳しい使い方: [docs/USAGE.ja.md](docs/USAGE.ja.md)

## What it does

- Pick a destination folder, choose models (by group, or one by one, filtered by format and size) and download them with revisions pinned in [`model_manifest.json`](model_manifest.json).
- Resume an interrupted download by starting it again on the same destination.
- Verify afterwards: compare every file with the SHA-256 (LFS files) / git blob SHA-1 (plain files) that Hugging Face publishes, or make and check a `sha256sums.txt` for moving files between machines.
- Uses the official [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library (`snapshot_download`) and the public Hub API only.
- Local GUI only: the server listens on `127.0.0.1`, every request needs a random per-run key, and `Host` / `Origin` are checked. There is deliberately no option to expose it. For a remote machine use SSH port forwarding.
- An access token is optional (most listed repos are public and ungated). If you enter one it is validated with Hugging Face, kept in the server's memory only (never written to disk or to an environment variable), masked in logs, and cleared on request, on a timer, after a job, or on exit. Passing a token is also the recommended way to avoid [Hub rate limits](https://huggingface.co/docs/hub/rate-limits).

## Platform support

| Platform | Status |
|---|---|
| **Windows 11** | **Tested and the only supported platform.** |
| Linux | Run by the author, but no guarantee. |
| macOS | Not tested. The launcher is provided as is. |

The GUI text is currently in Japanese only; the CLI and this README are English.

Requires Python 3.10 or newer and network access to PyPI and huggingface.co.

## Quick start

- Windows: double-click `start_gui_windows.bat`
- Linux: `./start_gui_linux.sh` (add `--no-browser` on a headless box)
- macOS: `start_gui_mac.command` (untested; it also clears the quarantine flag on its own folder)

The first run creates a `.venv`, installs `requirements.txt`, starts the server and opens the browser at a URL like `http://127.0.0.1:8765/?k=<random key>`. Use the URL exactly as printed in the terminal.

CLI (no GUI):

```bash
# plan and free-space check; creates and downloads nothing
./download_models.sh --dry-run --group core4 --dest /path/to/models

# download
./download_models.sh --group core4 --dest /path/to/models

# verify against Hugging Face's published hashes
.venv/bin/python verify_hf_hashes.py --dest /path/to/models --group core4
```

Tests: `pip install pytest && python -m pytest tests`

## About the model list

[`model_manifest.json`](model_manifest.json) is a **snapshot taken on 2026-11-03**. Revisions and sizes go stale; edit or re-pin it for your own use. The `role` fields are the author's own evaluation labels, not a recommendation. The `license` field is a convenience note only.

Please read these before you use or pass on anything you download:

- **Models are not part of this repository and are not redistributed by it.** Each model is downloaded from its own Hugging Face repository and stays under that repository's license and terms. Some licenses limit commercial use, require attribution, or are gated. Check each one yourself.
- The list includes some community-modified models (for example "uncensored" or "abliterated" variants). They are public on Hugging Face at the time of writing; you are responsible for how you use them.
- Downloads are large (several hundred GB to multiple TB for the bigger groups). Check free space first (the dry run does).

## Disclaimer

Provided as is, without warranty (see [LICENSE](LICENSE)). The author does not take responsibility for how this tool or the downloaded models are used or redistributed. Make sure your use follows the [Hugging Face Terms of Service](https://huggingface.co/terms-of-service), [Content Policy](https://huggingface.co/content-policy) and the license of every model you download.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
