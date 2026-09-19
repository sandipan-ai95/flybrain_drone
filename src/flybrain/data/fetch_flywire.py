"""Download the FlyWire FAFB v783 public release from Google Cloud Storage.

The FlyWire Codex team publishes the raw connectome as gzipped CSVs at:

    https://storage.googleapis.com/flywire-data/codex/data/fafb/783/

License: **CC BY 4.0** (attribute Dorkenwald et al., 2024 / FlyWire consortium).

⚠️ This bucket layout is not officially guaranteed to be permanent. The
authoritative download route remains codex.flywire.ai. If a file 404s, sign
in there and grab it manually.

Usage
-----
    python -m flybrain.data.fetch_flywire            # neurons + connections only
    python -m flybrain.data.fetch_flywire --all      # everything (+ classifications, coords)
    python -m flybrain.data.fetch_flywire --dest data/raw/flywire
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"

CORE_FILES = ["neurons.csv.gz", "connections.csv.gz"]
EXTRA_FILES = ["classification.csv.gz", "consolidated_cell_types.csv.gz",
               "coordinates.csv.gz"]


def _download(url: str, dest: Path) -> None:
    """Streaming download with a progress bar (no third-party deps)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        chunk = 1 << 20  # 1 MiB
        with open(tmp, "wb") as out:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(f"\r  {dest.name}  "
                                     f"{done/1e6:6.1f}/{total/1e6:.1f} MB  {pct:5.1f}%")
                else:
                    sys.stdout.write(f"\r  {dest.name}  {done/1e6:6.1f} MB")
                sys.stdout.flush()
    tmp.replace(dest)
    sys.stdout.write("  ✓\n")


def fetch(dest: Path, files: list[str], overwrite: bool = False) -> None:
    print(f"[flywire-fetch] destination: {dest.resolve()}")
    for name in files:
        out = dest / name
        if out.exists() and not overwrite:
            size = out.stat().st_size / 1e6
            print(f"  {name} already present ({size:.1f} MB) — skipping")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"[flywire-fetch] GET {url}")
        try:
            _download(url, out)
        except Exception as e:
            print(f"\n  ❌ failed: {e}")
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
            raise
    print("[flywire-fetch] done.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=Path("data/raw/flywire"))
    p.add_argument("--all", action="store_true",
                   help="also download classification / cell types / coordinates")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    files = CORE_FILES + (EXTRA_FILES if a.all else [])
    fetch(a.dest, files, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
