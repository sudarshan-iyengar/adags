#!/usr/bin/env python3
"""Resumable, manifest-backed acquisition of the anonymously-readable ImViD
Google Drive release into Apollo persistent storage.

Authority: the 2026-08-24 block directive (Lane D), which explicitly
authorizes downloading the complete anonymously accessible ImViD release
exposed by Drive folder ``1TrhrOrmFdvw-wTRPiVqlyWUWZrJJgHZe`` directly into
``/apollo/users/sri/proj_adags/data/imvid/raw/<folder>/``. Prior context:
``research-wiki/operations/dataset-admission-matrix-2026-08-18.md`` section
C1 (the enumeration that recorded 8 folders / 325 files / 1,181,076,959,285
bytes) and ``research-wiki/operations/imvid-sample-ingestion.md`` (the
300-frame Opera sample, already acquired -- this script never re-fetches it).

Why not ``gdown --folder``: at terabyte scale the binding requirements are
resumability, exact byte verification, per-file locking and an append-only
provenance manifest. ``gdown`` provides none of them and cannot be restarted
safely mid-file. ``gdown`` remains usable for an independent enumeration
cross-check; this script deliberately implements enumeration itself from the
public ``embeddedfolderview`` endpoint so acquisition has no third-party
dependency inside the Apollo container.

Design (fail-closed throughout):

  * ``enumerate``  walk the public folder tree; 1-byte HTTP Range probe per
                   file for its exact size; write an inventory JSON. No bulk
                   bytes are transferred.
  * ``download``   per-file O_EXCL lock, ``.part`` staging, Range resume,
                   exact byte-count verification, SHA-256, atomic rename,
                   read-only promotion, one O_APPEND manifest line.
  * ``status``     reconcile destination + manifest against the inventory.

Google Drive specifics that shape the implementation, all verified live on
2026-08-24 against ``scene1_opera/cam00.mp4``:

  * ``HEAD`` returns a ``Content-Length: 0`` virus-scan interstitial and is
    useless -- sizes come from ``GET`` with ``Range: bytes=0-0`` reading
    ``Content-Range``.
  * ``https://drive.usercontent.google.com/download?id=...&export=download&confirm=t``
    serves real bytes (``Content-Type: video/mp4``), answers ``206``, sets
    ``Accept-Ranges: bytes`` and honours arbitrary offsets (probed at
    3,224,000,000).
  * The publisher supplies NO checksum: no ETag, no Content-MD5, no
    X-Goog-Hash on the anonymous endpoints. The SHA-256 this script computes
    is therefore a LOCAL transfer-integrity record, never a comparison
    against a publisher hash. Correctness of the bytes rests on the exact
    byte-count match plus Drive's own TLS/TCP integrity.
  * A quota or rate-limit refusal arrives as an HTML body (often with a 200
    or 403), not as a transport error, so every response is content-type
    checked before a single byte is committed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

#: Bumped on ANY change to transfer or verification semantics. Recorded in
#: every manifest line so a later reader can tell which code produced it.
DOWNLOADER_VERSION = "imvid-fetch-1.2.0"

#: The user-supplied public folder. Not a secret; it is world-readable.
DEFAULT_ROOT_FOLDER_ID = "1TrhrOrmFdvw-wTRPiVqlyWUWZrJJgHZe"

#: Serves bytes. The ``uc?export=download`` form returns the interstitial.
DOWNLOAD_ENDPOINT = "https://drive.usercontent.google.com/download"
FOLDERVIEW_ENDPOINT = "https://drive.google.com/embeddedfolderview"

#: Frozen acquisition order (directive Lane D "Transfer order"), so useful
#: work can begin on Opera before the 1.1 TiB tail completes.
PRIORITY_ORDER: tuple[str, ...] = (
    "scene1_opera",
    "scene4_meeting",
    "scene7_playing",
    "scene2_laboratory",
    "scene5_rendition",
    "scene6_puppy",
    "scene3_classroom",
    "moving_rig",
)

#: The previously recorded anonymous inventory, for live comparison. A
#: mismatch is REPORTED, never silently accepted and never auto-corrected.
RECORDED_INVENTORY: dict[str, dict[str, int]] = {
    "moving_rig": {"files": 39, "mp4": 39, "bytes": 131_492_109_120},
    "scene1_opera": {"files": 41, "mp4": 39, "bytes": 125_649_776_270},
    "scene2_laboratory": {"files": 41, "mp4": 39, "bytes": 81_340_649_443},
    "scene3_classroom": {"files": 40, "mp4": 38, "bytes": 409_317_428_086},
    "scene4_meeting": {"files": 41, "mp4": 39, "bytes": 122_672_447_671},
    "scene5_rendition": {"files": 41, "mp4": 39, "bytes": 113_041_967_198},
    "scene6_puppy": {"files": 41, "mp4": 39, "bytes": 115_934_621_018},
    "scene7_playing": {"files": 41, "mp4": 39, "bytes": 81_627_960_479},
}
RECORDED_TOTAL_FILES = 325
RECORDED_TOTAL_BYTES = 1_181_076_959_285

_UA = "Mozilla/5.0 (X11; Linux x86_64) adags-imvid-fetch/1.0"
#: A lock older than this is treated as abandoned. Generous by design: the
#: largest file in the release is ~3.2 GB, which is ~3 minutes even at a
#: degraded 20 MB/s, so two hours cannot be reached by a live transfer. A
#: worker killed mid-file (SIGKILL skips the `finally`) would otherwise leave
#: a lock that blocks that file permanently -- silently, since a locked file
#: is skipped rather than reported as an error.
LOCK_STALE_SECONDS = 7200
#: Escalating waits after a Drive quota/rate refusal, in seconds. MEASURED
#: cause: two workers pulling ~42 MB/s sustained tripped a limit after ~62 GiB
#: in ~24 minutes, while a probe from a DIFFERENT host still served bytes for
#: an untouched file -- so the limit is per-IP against the requesting host,
#: not per-file and not account-wide. Escalating rather than fixed because the
#: reset horizon is unknown and a fixed short retry would be the "hammer Drive"
#: behaviour the acquisition rules forbid.
QUOTA_BACKOFF_SECONDS: tuple[int, ...] = (900, 1800, 3600, 3600, 7200)
_CHUNK = 8 * 1024 * 1024
_ENTRY_SPLIT = re.compile(r'(?=<div class="flip-entry" id="entry-)')
_ENTRY_ID = re.compile(r'id="entry-([A-Za-z0-9_-]+)"')
_ENTRY_TITLE = re.compile(r'flip-entry-title">([^<]*)<')
_CONTENT_RANGE = re.compile(r"bytes\s+\d+-\d+/(\d+)")


class TransferError(RuntimeError):
    """Any fail-closed refusal. Never raised for a recoverable quota pause."""


class QuotaPause(RuntimeError):
    """Drive declined to serve. Partials are preserved; the caller backs off."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 120.0):
    request = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def _fetch_text(url: str, timeout: float = 120.0) -> str:
    with _open(url, timeout=timeout) as response:
        raw = response.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def _file_url(file_id: str) -> str:
    return f"{DOWNLOAD_ENDPOINT}?id={file_id}&export=download&confirm=t"


def _classify_refusal(status: int, content_type: str) -> None:
    """Raise QuotaPause when Drive answered with an interstitial/refusal.

    An HTML body on a bulk-file endpoint is ALWAYS a refusal (quota, rate
    limit, login wall, or an error page) -- never data. Treating it as data
    is how a 'successful' download becomes a 6 KB HTML file on disk, which
    the byte-count check would catch, but far later and after wasted work.
    """

    if status in (403, 429, 500, 502, 503, 504):
        raise QuotaPause(f"Drive declined with HTTP {status}")
    if "text/html" in content_type.lower():
        raise QuotaPause(
            f"Drive returned an HTML body (content-type {content_type!r}) instead of file "
            "bytes -- quota, rate limit, login wall or error page"
        )


def probe_size(file_id: str, timeout: float = 120.0) -> tuple[int, str]:
    """Return ``(total_bytes, content_type)`` from a 1-byte Range GET.

    Uses ``Range: bytes=0-0`` because HEAD on this endpoint returns the
    virus-scan interstitial with ``Content-Length: 0``. Expects HTTP 206 and
    a parseable ``Content-Range``; anything else fails closed rather than
    guessing a size, since the size is what later verifies the transfer.
    """

    try:
        with _open(_file_url(file_id), headers={"Range": "bytes=0-0"}, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            content_range = response.headers.get("Content-Range", "")
            response.read(1)
    except urllib.error.HTTPError as exc:
        _classify_refusal(exc.code, exc.headers.get("Content-Type", "") if exc.headers else "")
        raise TransferError(f"size probe failed for {file_id}: HTTP {exc.code}") from exc

    _classify_refusal(status, content_type)
    if status != 206:
        raise TransferError(
            f"size probe for {file_id} returned HTTP {status}, expected 206 -- "
            "the endpoint is not honouring Range and resume cannot be trusted"
        )
    match = _CONTENT_RANGE.search(content_range)
    if not match:
        raise TransferError(
            f"size probe for {file_id} returned an unparseable Content-Range: {content_range!r}"
        )
    return int(match.group(1)), content_type


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def list_folder(folder_id: str) -> list[tuple[str, str]]:
    """Return ``[(entry_id, title), ...]`` for one public Drive folder.

    Parsed from the public ``embeddedfolderview`` HTML. The id and the title
    are extracted from the SAME DOM element rather than from two independent
    ordered lists, because a positional pairing would silently mis-associate
    every file if the endpoint ever reordered or dropped one field.
    """

    html = _fetch_text(f"{FOLDERVIEW_ENDPOINT}?id={folder_id}#list")
    entries: list[tuple[str, str]] = []
    for block in _ENTRY_SPLIT.split(html):
        id_match = _ENTRY_ID.search(block)
        title_match = _ENTRY_TITLE.search(block)
        if id_match and title_match:
            entries.append((id_match.group(1), title_match.group(1).strip()))
    return entries


def cmd_enumerate(args: argparse.Namespace) -> int:
    root = list_folder(args.root_folder_id)
    if not root:
        raise TransferError(
            f"root folder {args.root_folder_id} enumerated to ZERO entries -- it is not "
            "anonymously readable from this host, or the endpoint changed shape"
        )
    print(f"root folder {args.root_folder_id}: {len(root)} top-level entries", flush=True)

    folders: dict[str, Any] = {}
    total_files = 0
    total_bytes = 0
    for folder_id, folder_name in sorted(root, key=lambda item: item[1]):
        entries = list_folder(folder_id)
        files = []
        for file_id, name in entries:
            size, content_type = probe_size(file_id)
            files.append(
                {
                    "drive_file_id": file_id,
                    "name": name,
                    "relative_path": f"{folder_name}/{name}",
                    "expected_bytes": size,
                    "content_type": content_type,
                }
            )
            time.sleep(args.probe_delay)
        files.sort(key=lambda item: item["name"])
        folder_bytes = sum(item["expected_bytes"] for item in files)
        mp4 = sum(1 for item in files if item["name"].lower().endswith(".mp4"))
        folders[folder_name] = {
            "drive_folder_id": folder_id,
            "file_count": len(files),
            "mp4_count": mp4,
            "total_bytes": folder_bytes,
            "files": files,
        }
        total_files += len(files)
        total_bytes += folder_bytes
        print(
            f"  {folder_name:20s} id={folder_id}  files={len(files):3d}  mp4={mp4:3d}  "
            f"bytes={folder_bytes:,}",
            flush=True,
        )

    inventory = {
        "schema": "imvid-drive-inventory-v1",
        "downloader_version": DOWNLOADER_VERSION,
        "root_folder_id": args.root_folder_id,
        "enumerated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "folder_count": len(folders),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "folders": folders,
    }

    comparison = compare_against_recorded(inventory)
    inventory["comparison_against_recorded"] = comparison

    out = Path(args.inventory)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nTOTAL  folders={len(folders)}  files={total_files}  bytes={total_bytes:,}")
    print(f"recorded  files={RECORDED_TOTAL_FILES}  bytes={RECORDED_TOTAL_BYTES:,}")
    print(f"inventory written: {out}")
    for line in comparison["notes"]:
        print(f"  NOTE: {line}")
    return 0


def compare_against_recorded(inventory: dict[str, Any]) -> dict[str, Any]:
    """Diff the live enumeration against the recorded 325-file inventory.

    Reports differences; never rewrites either side. A live release that has
    changed since 2026-08-18 is a finding about the publisher, not an error
    to be normalised away.
    """

    notes: list[str] = []
    live = inventory["folders"]
    for name in sorted(set(live) | set(RECORDED_INVENTORY)):
        if name not in RECORDED_INVENTORY:
            notes.append(f"{name}: present live, ABSENT from the recorded inventory")
            continue
        if name not in live:
            notes.append(f"{name}: in the recorded inventory, ABSENT live")
            continue
        want = RECORDED_INVENTORY[name]
        got = live[name]
        if got["file_count"] != want["files"]:
            notes.append(
                f"{name}: file count {got['file_count']} live vs {want['files']} recorded"
            )
        if got["mp4_count"] != want["mp4"]:
            notes.append(f"{name}: mp4 count {got['mp4_count']} live vs {want['mp4']} recorded")
        if got["total_bytes"] != want["bytes"]:
            notes.append(
                f"{name}: bytes {got['total_bytes']:,} live vs {want['bytes']:,} recorded "
                f"(delta {got['total_bytes'] - want['bytes']:+,})"
            )
    if inventory["total_files"] != RECORDED_TOTAL_FILES:
        notes.append(
            f"TOTAL file count {inventory['total_files']} live vs {RECORDED_TOTAL_FILES} recorded"
        )
    if inventory["total_bytes"] != RECORDED_TOTAL_BYTES:
        notes.append(
            f"TOTAL bytes {inventory['total_bytes']:,} live vs {RECORDED_TOTAL_BYTES:,} recorded"
        )
    if not notes:
        notes.append("live enumeration matches the recorded inventory exactly")
    return {"matches_recorded": len(notes) == 1 and "matches" in notes[0], "notes": notes}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    """One O_APPEND JSON line. Append-only; nothing is ever rewritten."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_lock(lock_path: Path) -> bool:
    """O_EXCL per-file claim so two workers never fetch the same file."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (0, 1):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if attempt == 1:
                return False
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between the failed create and the stat; retry
            if age < LOCK_STALE_SECONDS:
                return False
            print(f"    STEALING a stale lock ({age/3600:.1f} h old): {lock_path.name} -- "
                  "its worker died without releasing it", flush=True)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    else:
        return False
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "host": os.uname().nodename if hasattr(os, "uname") else "",
                   "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
    return True


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(entry: dict[str, Any], dest_root: Path, manifest_path: Path,
                 locks_dir: Path, *, timeout: float = 300.0) -> str:
    """Fetch one file resumably. Returns 'done' | 'skipped' | 'locked'.

    Raises QuotaPause on a Drive refusal WITHOUT deleting the partial, so the
    next invocation resumes from the recorded offset. Raises TransferError on
    a byte-count mismatch or a conflicting destination -- both are conditions
    a retry cannot fix and must not be papered over.
    """

    relative = entry["relative_path"]
    expected = int(entry["expected_bytes"])
    final_path = dest_root / relative
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    lock_path = locks_dir / (relative.replace("/", "__") + ".lock")

    if final_path.exists():
        actual = final_path.stat().st_size
        if actual == expected:
            return "skipped"
        raise TransferError(
            f"conflicting destination {final_path}: {actual:,} bytes on disk, "
            f"{expected:,} expected. Refusing to overwrite; inspect and remove deliberately."
        )

    if not _acquire_lock(lock_path):
        return "locked"

    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        offset = part_path.stat().st_size if part_path.exists() else 0
        if offset > expected:
            raise TransferError(
                f"partial {part_path} is {offset:,} bytes, larger than the expected "
                f"{expected:,}. Refusing to resume from a corrupt partial."
            )
        if offset == expected:
            print(f"    {relative}: partial already complete, promoting", flush=True)
        else:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                response = _open(_file_url(entry["drive_file_id"]), headers=headers, timeout=timeout)
            except urllib.error.HTTPError as exc:
                _classify_refusal(exc.code, exc.headers.get("Content-Type", "") if exc.headers else "")
                raise TransferError(f"{relative}: HTTP {exc.code}") from exc

            with response:
                _classify_refusal(response.status, response.headers.get("Content-Type", ""))
                if offset and response.status != 206:
                    raise TransferError(
                        f"{relative}: resume requested from {offset:,} but the server answered "
                        f"HTTP {response.status} instead of 206 -- refusing to append to a "
                        "partial with a full-file response, which would corrupt it"
                    )
                started = time.time()
                written = 0
                with open(part_path, "ab") as handle:
                    while True:
                        block = response.read(_CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                rate = written / max(time.time() - started, 1e-6) / 1e6
                print(f"    {relative}: +{written:,} bytes at {rate:.1f} MB/s", flush=True)

        actual = part_path.stat().st_size
        if actual != expected:
            raise QuotaPause(
                f"{relative}: transfer ended early at {actual:,} of {expected:,} bytes "
                f"({100.0 * actual / max(expected, 1):.2f}%). Partial PRESERVED for resume."
            )

        digest = _sha256_of(part_path)
        os.replace(part_path, final_path)
        os.chmod(final_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        _append_manifest(
            manifest_path,
            {
                "schema": "imvid-transfer-manifest-v1",
                "downloader_version": DOWNLOADER_VERSION,
                "source_folder": relative.split("/", 1)[0],
                "drive_file_id": entry["drive_file_id"],
                "relative_path": relative,
                "name": entry["name"],
                "expected_bytes": expected,
                "observed_bytes": actual,
                "sha256": digest,
                "sha256_is_local_transfer_integrity_only": True,
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print(f"    {relative}: COMPLETE {actual:,} bytes sha256={digest[:16]}...", flush=True)
        return "done"
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def cmd_download(args: argparse.Namespace) -> int:
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    dest_root = Path(args.dest_root)
    manifest_path = Path(args.manifest)
    locks_dir = Path(args.locks_dir)

    folders = inventory["folders"]
    order = [name for name in PRIORITY_ORDER if name in folders]
    order += [name for name in sorted(folders) if name not in order]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - set(folders)
        if unknown:
            raise TransferError(f"--only names folders absent from the inventory: {sorted(unknown)}")
        order = [name for name in order if name in wanted]

    done = skipped = 0
    for folder_name in order:
        entries = folders[folder_name]["files"]
        print(f"\n[{folder_name}] {len(entries)} files, "
              f"{folders[folder_name]['total_bytes']:,} bytes", flush=True)
        for entry in entries:
            attempt = 0
            while True:
                try:
                    outcome = download_one(entry, dest_root, manifest_path, locks_dir,
                                           timeout=args.timeout)
                    break
                except QuotaPause as exc:
                    print(f"    QUOTA/RATE PAUSE: {exc}", flush=True)
                    if attempt >= len(QUOTA_BACKOFF_SECONDS) or args.no_backoff:
                        print("    Backoff exhausted. Every completed byte and partial is "
                              "preserved; re-run this exact command to resume.", flush=True)
                        return 3
                    wait = QUOTA_BACKOFF_SECONDS[attempt]
                    attempt += 1
                    print(f"    backing off {wait}s (attempt {attempt}/"
                          f"{len(QUOTA_BACKOFF_SECONDS)}) then resuming from the recorded "
                          "offset -- nothing is deleted", flush=True)
                    time.sleep(wait)
            if outcome == "done":
                done += 1
                if args.sleep_between_files:
                    time.sleep(args.sleep_between_files)
            elif outcome == "skipped":
                skipped += 1
            if args.max_files and done >= args.max_files:
                print(f"\n--max-files {args.max_files} reached; stopping cleanly.", flush=True)
                return 0
    print(f"\ndownloaded={done}  already-present={skipped}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    dest_root = Path(args.dest_root)
    complete_bytes = partial_bytes = missing_bytes = 0
    complete = partial = missing = 0
    rows = []
    for folder_name, folder in sorted(inventory["folders"].items()):
        f_done = f_part = f_missing = 0
        f_bytes = 0
        for entry in folder["files"]:
            final_path = dest_root / entry["relative_path"]
            part_path = final_path.with_suffix(final_path.suffix + ".part")
            expected = int(entry["expected_bytes"])
            if final_path.exists() and final_path.stat().st_size == expected:
                f_done += 1
                f_bytes += expected
                complete += 1
                complete_bytes += expected
            elif part_path.exists():
                got = part_path.stat().st_size
                f_part += 1
                f_bytes += got
                partial += 1
                partial_bytes += got
                missing_bytes += expected - got
            else:
                f_missing += 1
                missing += 1
                missing_bytes += expected
        rows.append((folder_name, f_done, f_part, f_missing, f_bytes, folder["total_bytes"]))

    print(f"{'folder':22s} {'done':>5s} {'part':>5s} {'miss':>5s} {'have_bytes':>18s} {'total_bytes':>18s}")
    for name, d, p, m, have, total in rows:
        print(f"{name:22s} {d:5d} {p:5d} {m:5d} {have:18,d} {total:18,d}")
    print(f"\ncomplete={complete} partial={partial} missing={missing}")
    print(f"bytes complete={complete_bytes:,}  in partials={partial_bytes:,}  "
          f"remaining={missing_bytes:,}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_imvid_release.py",
        description="Resumable acquisition of the public ImViD Drive release into Apollo storage.",
    )
    parser.add_argument("--version", action="version", version=DOWNLOADER_VERSION)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    default_root = "/apollo/users/sri/proj_adags/data/imvid/raw"
    default_inv = "/apollo/users/sri/proj_adags/data/imvid/imvid_drive_inventory.json"
    default_man = "/apollo/users/sri/proj_adags/data/imvid/imvid_transfer_manifest.jsonl"
    default_locks = "/apollo/users/sri/proj_adags/data/imvid/.locks"

    enum = sub.add_parser("enumerate", help="walk the public tree; probe exact sizes; write inventory")
    enum.add_argument("--root-folder-id", default=DEFAULT_ROOT_FOLDER_ID)
    enum.add_argument("--inventory", default=default_inv)
    enum.add_argument("--probe-delay", type=float, default=0.15,
                      help="seconds between size probes; keeps request rate polite")
    enum.set_defaults(func=cmd_enumerate)

    down = sub.add_parser("download", help="resumable transfer in the frozen priority order")
    down.add_argument("--inventory", default=default_inv)
    down.add_argument("--dest-root", default=default_root)
    down.add_argument("--manifest", default=default_man)
    down.add_argument("--locks-dir", default=default_locks)
    down.add_argument("--only", nargs="*", default=None, help="restrict to these source folders")
    down.add_argument("--max-files", type=int, default=0, help="stop cleanly after N new files")
    down.add_argument("--timeout", type=float, default=300.0)
    down.add_argument("--sleep-between-files", type=float, default=0.0,
                      help="seconds to pause after each completed file; lowers the "
                           "sustained request rate that trips Drive's per-IP limit")
    down.add_argument("--no-backoff", action="store_true",
                      help="exit immediately on a quota refusal instead of backing off "
                           "and resuming (the default is to self-heal)")
    down.set_defaults(func=cmd_download)

    stat_p = sub.add_parser("status", help="reconcile destination against the inventory")
    stat_p.add_argument("--inventory", default=default_inv)
    stat_p.add_argument("--dest-root", default=default_root)
    stat_p.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TransferError, QuotaPause) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
