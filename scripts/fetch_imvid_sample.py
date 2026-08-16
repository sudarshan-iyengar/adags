#!/usr/bin/env python3
"""Acquire and verify the ImViD "Opera" SAMPLE, reproducibly.

Authority: the ImViD lane of the 2026-08-16 directive, which authorizes
sample preflight, acquisition and an ingestion smoke.

WHY THIS IS A TRACKED SCRIPT. The DiVa-360 `chess_long` pilot is blocked
with "no reproducible acquisition path in the repository": that tranche
was collected by hand through a Dropbox browser UI, so the acquisition
cannot be repeated or audited
(`research-wiki/operations/elgs-m1-evidence-wiring-record.md`). The ImViD
sample does not have to inherit that problem — it is a single ungated
HTTPS asset with a publisher-computed digest, so the acquisition can be
one command with a hash gate.

WHAT IS AND IS NOT GATED. The SAMPLE is a plain public download from
GitHub Releases: no form, no login, no click-through, CC BY 4.0
(attribution). The FULL dataset is different — it requires an
application form emailed to the authors and manual approval — and this
script deliberately cannot fetch it. Acquiring the full dataset is a
user decision and a user action.

VERIFIED FACTS this script encodes (from the GitHub Releases API and the
repository's own README/LICENSE, not from prose):
  * asset `scene1_opera.zip`, release `v0.2`
  * 1,001,763,804 bytes
  * sha256 7cc2c5eba67da6a993e151c60418f79a446ef485122cae4e51917fe9fdbd682b
  * contents: Scene 1 videos, 300 frames, 5K at 60 FPS, H.264 MP4, plus
    COLMAP-native `cameras.txt` and `images.txt`
  * 39 cameras; the Opera scene was captured fixed-point only
  * NO masks, optical flow, depth or point cloud ship with it — the
    ImViD baseline computes flow and depth at training time from RGB,
    and a sparse cloud must be triangulated with COLMAP

FAIL-CLOSED. The download streams to a `.part` file and is renamed into
place ONLY after the digest matches; a mismatch leaves the partial file
under its `.part` name with a `.rejected` marker rather than anywhere it
could be mistaken for good data. An already-present archive is re-hashed
rather than re-downloaded, so the script is idempotent and a second run
is a verification pass. The archive is left read-only, per the standing
rule that raw datasets are read-only.

`--inspect` reads the zip CENTRAL DIRECTORY only. It does not extract:
it reports the entry inventory and the uncompressed total so the real
extraction cost is MEASURED before any extraction is authorized.

`--extract` unpacks the archive's members beside it. This is a small
operation and is NOT the large one people expect: the members are H.264
MP4s, so the zip is a container rather than a compressor (measured ratio
1.00x) and the extracted total is under 1 GiB. The ~557 GB figure from
the preflight describes DECODED FRAMES, which this does not produce.

Extraction is required rather than optional, and the reason was measured:
`ffprobe` on a 4 MiB prefix of `cam00.mp4` returns `moov atom not found`,
so these files are not `faststart` and their index sits at the END. A
reader cannot stream-decode them from inside the zip; the whole file has
to be present.

Usage:
  python3 scripts/fetch_imvid_sample.py --dest /apollo/users/sri/proj_adags/data/imvid
  python3 scripts/fetch_imvid_sample.py --dest <dir> --inspect
  python3 scripts/fetch_imvid_sample.py --dest <dir> --verify-only
  python3 scripts/fetch_imvid_sample.py --dest <dir> --verify-only --extract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

SAMPLE = {
    "name": "scene1_opera.zip",
    "url": (
        "https://github.com/Metaverse-AI-Lab-THU/ImViD/releases/download/"
        "v0.2/scene1_opera.zip"
    ),
    "release": "v0.2",
    "size_bytes": 1001763804,
    "sha256": "7cc2c5eba67da6a993e151c60418f79a446ef485122cae4e51917fe9fdbd682b",
    "license": "CC BY 4.0 (attribution)",
    "gated": False,
    "repository": "https://github.com/Metaverse-AI-Lab-THU/ImViD",
    "paper": "https://arxiv.org/pdf/2604.09473",
    "declared_contents": {
        "cameras": 39,
        "frames": 300,
        "fps": 60,
        "resolution": "5312x2988",
        "codec": "H.264 MP4",
        "calibration": "COLMAP-native cameras.txt + images.txt",
        "rig": "fixed-point only for the Opera scene",
        "ships_masks_flow_depth_pointcloud": False,
    },
}

_CHUNK = 1 << 20


def _sha256(path: Path, *, progress: bool = False) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
            if progress and total % (256 * _CHUNK) == 0:
                print(f"  hashed {total / 2**30:.2f} GiB", flush=True)
    return digest.hexdigest(), total


def _download(url: str, target: Path) -> int:
    part = target.with_suffix(target.suffix + ".part")
    total = 0
    with urlopen(url) as response, open(part, "wb") as out:
        declared = response.headers.get("Content-Length")
        print(f"  server declares Content-Length={declared}", flush=True)
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            out.write(block)
            total += len(block)
            if total % (128 * _CHUNK) == 0:
                print(f"  {total / 2**30:.2f} GiB", flush=True)
    digest, size = _sha256(part)
    if digest != SAMPLE["sha256"]:
        marker = part.with_suffix(part.suffix + ".rejected")
        marker.write_text(
            json.dumps({"expected": SAMPLE["sha256"], "got": digest, "bytes": size}, indent=2),
            encoding="utf-8",
        )
        raise ContractError(
            f"digest mismatch: expected {SAMPLE['sha256']}, got {digest} over "
            f"{size} bytes. The partial file is preserved at {part} with a "
            f".rejected marker and was NOT moved into place."
        )
    if size != SAMPLE["size_bytes"]:
        raise ContractError(
            f"size mismatch: expected {SAMPLE['size_bytes']}, got {size} "
            "(digest matched, which should be impossible -- refusing)"
        )
    part.rename(target)
    return size


def _inspect(archive: Path) -> dict:
    """Central-directory inventory only. Nothing is extracted."""
    by_suffix: dict[str, dict] = {}
    entries = 0
    compressed = 0
    uncompressed = 0
    largest: list[tuple[int, str]] = []
    names: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            entries += 1
            compressed += info.compress_size
            uncompressed += info.file_size
            suffix = Path(info.filename).suffix.lower() or "<none>"
            slot = by_suffix.setdefault(suffix, {"n": 0, "bytes": 0})
            slot["n"] += 1
            slot["bytes"] += info.file_size
            largest.append((info.file_size, info.filename))
            if len(names) < 80:
                names.append(info.filename)
    largest.sort(reverse=True)
    return {
        "entries": entries,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "by_suffix": by_suffix,
        "largest_entries": [{"bytes": b, "name": n} for b, n in largest[:15]],
        "sample_names": names,
    }


def _extract(archive: Path, dest: Path) -> dict:
    """Unpack members beside the archive, then hash what landed.

    Refuses a member whose normalized path escapes `dest` (the zip-slip
    guard) and refuses to overwrite an existing non-empty tree, so a
    second run is a no-op rather than a silent re-extract.
    """
    root = dest / "scene1_opera"
    if root.exists() and any(root.iterdir()):
        files = sorted(p for p in root.rglob("*") if p.is_file())
        return {
            "action": "already-extracted",
            "root": str(root),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }
    written: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (dest / info.filename).resolve()
            if dest.resolve() not in target.parents:
                raise ContractError(
                    f"archive member {info.filename!r} escapes {dest}; refusing"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while True:
                    block = src.read(_CHUNK)
                    if not block:
                        break
                    out.write(block)
            if target.stat().st_size != info.file_size:
                raise ContractError(
                    f"{info.filename}: wrote {target.stat().st_size} bytes, "
                    f"central directory declares {info.file_size}"
                )
            try:
                os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass
            written.append({
                "name": info.filename,
                "bytes": int(info.file_size),
                "sha256": _sha256(target)[0],
            })
    return {
        "action": "extracted",
        "root": str(root),
        "files": len(written),
        "bytes": sum(entry["bytes"] for entry in written),
        "members": written,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True,
                        help="persistent directory OUTSIDE the git repository")
    parser.add_argument("--inspect", action="store_true",
                        help="also read the zip central directory (no extraction)")
    parser.add_argument("--extract", action="store_true",
                        help="unpack the members (under 1 GiB; NOT frame decoding)")
    parser.add_argument("--verify-only", action="store_true",
                        help="never download; hash whatever is already present")
    parser.add_argument("--timestamp", default=None,
                        help="acquisition timestamp recorded in the manifest")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    if REPO_ROOT in dest.resolve().parents or dest.resolve() == REPO_ROOT:
        raise ContractError(
            f"--dest {dest} is inside the repository at {REPO_ROOT}; raw "
            "datasets are never stored in git"
        )
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / SAMPLE["name"]

    report: dict = {"schema": "imvid-sample-acquisition-v1", "asset": dict(SAMPLE)}
    report["destination"] = str(archive)
    report["acquired_at"] = args.timestamp

    if archive.exists():
        print(f"[imvid] present: {archive} -- re-hashing rather than re-downloading",
              flush=True)
        digest, size = _sha256(archive, progress=True)
        report["action"] = "verified-existing"
        report["sha256_observed"] = digest
        report["size_observed"] = size
        if digest != SAMPLE["sha256"]:
            raise ContractError(
                f"the archive already at {archive} has digest {digest}, not "
                f"{SAMPLE['sha256']}. Refusing to touch it -- inspect it by hand."
            )
    elif args.verify_only:
        raise ContractError(f"--verify-only but nothing at {archive}")
    else:
        print(f"[imvid] downloading {SAMPLE['url']}", flush=True)
        size = _download(SAMPLE["url"], archive)
        report["action"] = "downloaded"
        report["sha256_observed"] = SAMPLE["sha256"]
        report["size_observed"] = size

    # Raw datasets are read-only (AGENTS.md). Applied AFTER the digest
    # gate, so a rejected download never reaches this line.
    try:
        os.chmod(archive, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        report["archive_mode"] = "read-only (0444)"
    except OSError as exc:
        report["archive_mode"] = f"could not set read-only: {exc!r}"

    print(f"[imvid] VERIFIED sha256={report['sha256_observed']} "
          f"bytes={report['size_observed']}", flush=True)

    if args.inspect:
        print("[imvid] reading the central directory (no extraction)", flush=True)
        inventory = _inspect(archive)
        report["inventory"] = inventory
        ratio = (
            inventory["uncompressed_bytes"] / inventory["compressed_bytes"]
            if inventory["compressed_bytes"]
            else 0.0
        )
        print(f"  entries={inventory['entries']}  "
              f"uncompressed={inventory['uncompressed_bytes'] / 2**30:.2f} GiB  "
              f"ratio={ratio:.2f}x", flush=True)
        for suffix, slot in sorted(inventory["by_suffix"].items()):
            print(f"    {suffix:8s} n={slot['n']:5d}  {slot['bytes'] / 2**30:8.3f} GiB",
                  flush=True)

    if args.extract:
        print("[imvid] extracting members (H.264 MP4s; NOT frame decoding)", flush=True)
        extraction = _extract(archive, dest)
        report["extraction"] = extraction
        print(f"  {extraction['action']}: {extraction['files']} files, "
              f"{extraction['bytes'] / 2**30:.3f} GiB -> {extraction['root']}",
              flush=True)

    manifest = dest / "MANIFEST.imvid_sample.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[imvid] manifest -> {manifest}", flush=True)
    print("IMVID_JSON_BEGIN", flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print("IMVID_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
