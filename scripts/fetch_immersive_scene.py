#!/usr/bin/env python3
"""Acquire and verify one Google Immersive Light Field Video scene, reproducibly.

Authority: the Google-Immersive lane of the 2026-08-18 directive, which
authorizes read-only inventory plus ONE pilot-scene acquisition and a bounded
preprocessing smoke.

WHY THIS IS A TRACKED SCRIPT. Same reason as
``scripts/fetch_imvid_sample.py``: the DiVa-360 ``chess_long`` pilot is
blocked because its tranche was collected by hand through a browser UI and
"there is no reproducible acquisition path in the repository"
(``research-wiki/operations/elgs-m1-evidence-wiring-record.md``). This dataset
does not have to inherit that. The archives are plain public GCS objects, so
the acquisition is one command with a digest gate.

WHAT IS GATED. Nothing. Verified by direct request, not from prose: the bucket
``deepview_video_raw_data`` is publicly listable through the GCS JSON API and
every object returns HTTP 200 to an unauthenticated GET with
``Accept-Ranges``-style byte-range support (a ``Range: bytes=0-0`` probe
returns 206). No form, no login, no click-through, no credentials.

**DIGEST HANDLING IS DIFFERENT FROM THE ImViD SCRIPT AND THE DIFFERENCE
MATTERS.** ImViD's publisher provides a sha256 that could be hard-coded and
gated on. Google publishes no sha256; what the object carries is an MD5 (as
base64 in the GCS metadata) and a CRC32C. So this script gates on the
**publisher's own MD5**, fetched from the object metadata at run time, and
additionally records a sha256 it computes itself for our provenance. Gating on
a digest we invented would verify only that the file did not change between
two of our own reads, which is not integrity verification at all.

RESUMABLE. The download streams to a ``.part`` file and resumes with an HTTP
``Range`` request if one is already present, so an interrupted multi-gigabyte
transfer is restartable rather than restarted. The rename into place happens
ONLY after the MD5 matches; a mismatch leaves the partial file under its
``.part`` name with a ``.rejected`` marker.

``--inspect`` reads the zip central directory only and extracts nothing, so
the real extraction cost is MEASURED before any extraction is authorized.

Scene selection is BY RULE, not by content: ``02_Flames`` is the pilot because
it appears in SpacetimeGaussians' published 7-scene Immersive protocol and in
that repository's own example invocation. The choice is deliberately not based
on expected disappearance/return content, which nothing here has measured.

Usage:
  python3 scripts/fetch_immersive_scene.py --scene 02_Flames --dest <dir>
  python3 scripts/fetch_immersive_scene.py --scene 02_Flames --dest <dir> --verify-only
  python3 scripts/fetch_immersive_scene.py --scene 02_Flames --dest <dir> --verify-only --inspect
  python3 scripts/fetch_immersive_scene.py --inventory --out <manifest.json>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_visibility.errors import ContractError  # noqa: E402

BUCKET = "deepview_video_raw_data"
OBJECT_URL = "https://storage.googleapis.com/" + BUCKET + "/{name}"
META_URL = "https://storage.googleapis.com/storage/v1/b/" + BUCKET + "/o/{name}"
LIST_URL = (
    "https://storage.googleapis.com/storage/v1/b/" + BUCKET
    + "/o?fields=items(name,size,md5Hash,crc32c,updated)&maxResults=1000"
)

#: The 15 released scenes, with the sizes verified 2026-08-18 by GCS listing.
#: Recorded so a changed remote is detected rather than silently accepted.
VERIFIED_INVENTORY = {
    "01_Welder.zip": 10554565078,
    "02_Flames.zip": 5474948990,
    "03_Dog.zip": 1398414399,
    "04_Truck.zip": 1811297781,
    "05_Horse.zip": 3462199574,
    "06_Goats.zip": 1617389119,
    "07_Car.zip": 3122877276,
    "08_Pond.zip": 1780220296,
    "09_Alexa_Meade_Exhibit.zip": 3500462687,
    "10_Alexa_Meade_Face_Paint_1.zip": 10587538727,
    "11_Alexa_Meade_Face_Paint_2.zip": 3578161493,
    "12_Cave.zip": 4578867676,
    "13_Birds.zip": 4887272734,
    "14_Puppy.zip": 8927189887,
    "15_Branches.zip": 179620533,
}
VERIFIED_TOTAL_BYTES = 65461026250

#: Pilot scene BY RULE (STG's published protocol and its own example), not by
#: expected content.
PILOT_SCENE = "02_Flames"

_CHUNK = 1 << 20


def _sha256_and_md5(path: Path, *, progress: bool = False) -> tuple[str, str, int]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            sha.update(block)
            md5.update(block)
            total += len(block)
            if progress and total % (512 * _CHUNK) == 0:
                print(f"  hashed {total / 2**30:.2f} GiB", flush=True)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii"), total


def fetch_metadata(name: str) -> dict:
    with urlopen(META_URL.format(name=name), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_inventory() -> dict:
    with urlopen(LIST_URL, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    items = payload.get("items", [])
    listed = {item["name"]: int(item["size"]) for item in items}
    drift = {
        name: {"recorded": VERIFIED_INVENTORY.get(name), "remote": size}
        for name, size in listed.items()
        if VERIFIED_INVENTORY.get(name) != size
    }
    missing = sorted(set(VERIFIED_INVENTORY) - set(listed))
    return {
        "bucket": BUCKET,
        "n_objects": len(listed),
        "total_bytes": sum(listed.values()),
        "objects": [
            {
                "name": item["name"],
                "size_bytes": int(item["size"]),
                "md5_base64": item.get("md5Hash"),
                "crc32c_base64": item.get("crc32c"),
                "updated": item.get("updated"),
            }
            for item in sorted(items, key=lambda i: i["name"])
        ],
        "recorded_total_bytes": VERIFIED_TOTAL_BYTES,
        "drift_vs_recorded": drift,
        "recorded_names_missing_remotely": missing,
        "pilot_scene_by_rule": PILOT_SCENE,
    }


def _download(url: str, target: Path, *, expected_size: int) -> int:
    """Stream to ``target.part``, RESUMING if a partial file is present."""

    part = target.with_suffix(target.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expected_size:
        raise ContractError(
            f"existing partial {part} is {have} bytes, larger than the declared "
            f"{expected_size}; refusing to guess -- delete it deliberately"
        )
    if have == expected_size:
        print(f"  partial file is already complete at {have} bytes", flush=True)
        return have

    request = Request(url)
    mode = "wb"
    if have:
        request.add_header("Range", f"bytes={have}-")
        mode = "ab"
        print(f"  RESUMING from {have / 2**30:.2f} GiB", flush=True)
    try:
        response = urlopen(request, timeout=300)
    except HTTPError as exc:
        if have and exc.code == 416:
            raise ContractError(
                f"server rejected the resume range for {part} (416); the partial "
                "file may be inconsistent with the remote object"
            ) from exc
        raise
    with response, open(part, mode) as out:
        if have and response.status != 206:
            raise ContractError(
                f"resume requested but server returned {response.status}, not 206; "
                "refusing to append to a partial file from a full-body response"
            )
        total = have
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            out.write(block)
            total += len(block)
            if total % (256 * _CHUNK) == 0:
                print(f"  {total / 2**30:.2f} GiB", flush=True)
    return total


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
            if len(names) < 60:
                names.append(info.filename)
    largest.sort(reverse=True)
    return {
        "entries": entries,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "ratio": (compressed / uncompressed) if uncompressed else None,
        "by_suffix": by_suffix,
        "largest_entries": [{"bytes": b, "name": n} for b, n in largest[:15]],
        "sample_names": names,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default=PILOT_SCENE, help="scene stem, e.g. 02_Flames")
    parser.add_argument("--dest", default=None, help="destination directory (raw root)")
    parser.add_argument("--inventory", action="store_true", help="read-only listing; no download")
    parser.add_argument("--verify-only", action="store_true", help="re-hash an existing archive")
    parser.add_argument("--inspect", action="store_true", help="central-directory inventory")
    parser.add_argument("--out", default=None, help="manifest path")
    args = parser.parse_args(argv)

    report: dict = {"schema_version": "immersive-acquisition-v1"}

    if args.inventory:
        report["inventory"] = fetch_inventory()
        print(json.dumps({
            "n_objects": report["inventory"]["n_objects"],
            "total_bytes": report["inventory"]["total_bytes"],
            "drift": report["inventory"]["drift_vs_recorded"],
            "missing": report["inventory"]["recorded_names_missing_remotely"],
        }))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 0

    if args.dest is None:
        raise ContractError("--dest is required unless --inventory is given")

    name = args.scene if args.scene.endswith(".zip") else args.scene + ".zip"
    if name not in VERIFIED_INVENTORY:
        raise ContractError(
            f"{name!r} is not one of the 15 recorded scenes: "
            f"{sorted(VERIFIED_INVENTORY)}"
        )

    meta = fetch_metadata(name)
    declared_size = int(meta["size"])
    declared_md5 = meta["md5Hash"]
    recorded = VERIFIED_INVENTORY[name]
    if declared_size != recorded:
        raise ContractError(
            f"remote size {declared_size} differs from the recorded "
            f"{recorded} for {name}; the remote object changed and this "
            "script refuses to acquire silently"
        )
    report["remote"] = {
        "name": name,
        "url": OBJECT_URL.format(name=name),
        "size_bytes": declared_size,
        "publisher_md5_base64": declared_md5,
        "publisher_crc32c_base64": meta.get("crc32c"),
        "updated": meta.get("updated"),
        "gated": False,
    }

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name

    if not target.exists() and not args.verify_only:
        free = os.statvfs(dest) if hasattr(os, "statvfs") else None
        if free is not None:
            free_bytes = free.f_bavail * free.f_frsize
            report["free_bytes_before"] = free_bytes
            if free_bytes < 3 * declared_size:
                raise ContractError(
                    f"only {free_bytes / 2**30:.1f} GiB free for a "
                    f"{declared_size / 2**30:.1f} GiB download; refusing to "
                    "fill the filesystem (3x headroom required)"
                )
        print(f"downloading {name} ({declared_size / 2**30:.2f} GiB)", flush=True)
        _download(report["remote"]["url"], target, expected_size=declared_size)
        part = target.with_suffix(target.suffix + ".part")
        sha, md5, size = _sha256_and_md5(part, progress=True)
        if md5 != declared_md5:
            marker = part.with_suffix(part.suffix + ".rejected")
            marker.write_text(
                json.dumps({"expected_md5": declared_md5, "got_md5": md5,
                            "bytes": size}, indent=2),
                encoding="utf-8",
            )
            raise ContractError(
                f"MD5 mismatch against the PUBLISHER's digest: expected "
                f"{declared_md5}, got {md5} over {size} bytes. The partial file "
                f"is preserved at {part} with a .rejected marker and was NOT "
                "moved into place."
            )
        if size != declared_size:
            raise ContractError(
                f"size mismatch {size} vs {declared_size} while the MD5 matched, "
                "which should be impossible -- refusing"
            )
        part.rename(target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        report["acquired"] = True
    else:
        report["acquired"] = False

    if not target.exists():
        raise ContractError(f"archive not present: {target}")

    sha, md5, size = _sha256_and_md5(target, progress=True)
    report["local"] = {
        "path": str(target),
        "size_bytes": size,
        "sha256": sha,
        "md5_base64": md5,
        "publisher_md5_match": md5 == declared_md5,
        "read_only": not bool(target.stat().st_mode & stat.S_IWUSR),
    }
    if md5 != declared_md5:
        raise ContractError(
            f"local archive fails the publisher MD5: expected {declared_md5}, "
            f"got {md5}"
        )

    if args.inspect:
        report["archive_inventory"] = _inspect(target)

    print(json.dumps({
        "name": name,
        "size_bytes": size,
        "sha256": sha,
        "publisher_md5_match": True,
        "entries": report.get("archive_inventory", {}).get("entries"),
        "uncompressed_bytes": report.get("archive_inventory", {}).get("uncompressed_bytes"),
    }))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
