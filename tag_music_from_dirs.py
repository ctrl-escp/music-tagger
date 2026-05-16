#!/usr/bin/env python3
"""Set MP3 ID3 tags from album directory and file names.

Default mode is a dry run. Add --apply to write tags.

The script intentionally has no third-party dependencies. It implements the
small subset of ID3v2.3/v2.4 needed for reading and replacing common text
frames while preserving artwork and other unrelated frames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


FIELD_ORDER = ("artist", "album", "title", "track", "year")
# User-facing fields mapped to their ID3 frame ids. Year differs between
# ID3v2.3 (TYER) and ID3v2.4 (TDRC), so both are accepted when reading.
FIELD_FRAMES = {
    "artist": {"TPE1"},
    "album": {"TALB"},
    "title": {"TIT2"},
    "track": {"TRCK"},
    "year": {"TYER", "TDRC"},
}
TARGET_FRAMES = set().union(*FIELD_FRAMES.values())


@dataclass(frozen=True)
class Tags:
    """Tags inferred from path names and command-line overrides."""

    artist: str
    album: str
    title: str
    track: str | None
    year: str | None


@dataclass(frozen=True)
class Analysis:
    """Per-file plan: what tags are available and which fields should change."""

    path: Path
    tags: Tags | None
    values: dict[str, str]
    replace_fields: list[str]
    existing_fields: dict[str, str] | None = None
    cache_key: str | None = None
    cache_entry: dict | None = None
    matched_artist_title: bool = True
    error: str | None = None


class Progress:
    """Single-line terminal progress indicator.

    The line is kept shorter than the terminal width to avoid wrapping; once a
    terminal wraps, carriage-return redraws cannot reliably erase the old line.
    """

    def __init__(self, total: int, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.start = time.monotonic()
        self.last_draw = 0.0
        self.last_len = 0

    def draw(self, inspected: int, updates: int, skipped: int, errors: int, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_draw < 0.2:
            return
        self.last_draw = now
        percent = (inspected / self.total * 100) if self.total else 100.0
        elapsed = max(0.001, now - self.start)
        rate = inspected / elapsed
        terminal_width = max(40, shutil.get_terminal_size(fallback=(80, 20)).columns)
        suffix = f" {inspected}/{self.total} {percent:5.1f}% upd={updates} skip={skipped} err={errors} {rate:4.0f}/s"
        bar_width = max(8, terminal_width - len(suffix) - 4)
        filled = int(bar_width * inspected / self.total) if self.total else bar_width
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"[{bar}]{suffix}"
        if len(line) >= terminal_width:
            suffix = f" {percent:5.1f}% u={updates} s={skipped} e={errors}"
            bar_width = max(8, terminal_width - len(suffix) - 4)
            filled = int(bar_width * inspected / self.total) if self.total else bar_width
            bar = "#" * filled + "-" * (bar_width - filled)
            line = f"[{bar}]{suffix}"
        line = line[: terminal_width - 1]
        sys.stderr.write("\r\x1b[2K" + line)
        sys.stderr.flush()
        self.last_len = len(line)

    def done(self, inspected: int, updates: int, skipped: int, errors: int) -> None:
        if not self.enabled:
            return
        self.draw(inspected, updates, skipped, errors, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


def syncsafe_to_int(data: bytes) -> int:
    """Decode the 4-byte syncsafe integer used by ID3 tag headers."""

    return ((data[0] & 0x7F) << 21) | ((data[1] & 0x7F) << 14) | ((data[2] & 0x7F) << 7) | (data[3] & 0x7F)


def int_to_syncsafe(value: int) -> bytes:
    """Encode an integer as the 4-byte syncsafe form used by ID3 tag headers."""

    return bytes([(value >> 21) & 0x7F, (value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F])


def frame_size(data: bytes, major: int) -> int:
    """Decode a frame-size field for ID3v2.3 or ID3v2.4."""

    if major == 4:
        return syncsafe_to_int(data)
    return int.from_bytes(data, "big")


def encode_frame_size(size: int, major: int) -> bytes:
    """Encode a frame-size field for the original tag's ID3 version."""

    if major == 4:
        return int_to_syncsafe(size)
    return size.to_bytes(4, "big")


def text_payload(text: str, major: int) -> bytes:
    """Build a text-frame payload using an encoding supported by the tag version."""

    if major == 4:
        return b"\x03" + text.encode("utf-8")
    return b"\x01" + text.encode("utf-16")


def make_text_frame(frame_id: str, text: str, major: int) -> bytes:
    payload = text_payload(text, major)
    return frame_id.encode("ascii") + encode_frame_size(len(payload), major) + b"\x00\x00" + payload


def frame_id_for_field(field: str, major: int) -> str:
    """Return the ID3 frame id used when writing a logical field."""

    if field == "artist":
        return "TPE1"
    if field == "album":
        return "TALB"
    if field == "title":
        return "TIT2"
    if field == "track":
        return "TRCK"
    if field == "year":
        return "TDRC" if major == 4 else "TYER"
    raise KeyError(field)


def tag_flags(blob: bytes) -> int:
    if len(blob) >= 10 and blob[:3] == b"ID3":
        return blob[5]
    return 0


def split_existing_tag(blob: bytes) -> tuple[int, bytes, bytes]:
    """Return (major_version, tag_body, audio_bytes) for a whole MP3 file."""

    if len(blob) < 10 or blob[:3] != b"ID3":
        return 3, b"", blob
    major = blob[3]
    if major not in (3, 4):
        return 3, b"", blob[10 + syncsafe_to_int(blob[6:10]) :]
    tag_size = syncsafe_to_int(blob[6:10])
    return major, blob[10 : 10 + tag_size], blob[10 + tag_size :]


def read_id3_header_and_body(path: Path) -> tuple[int, int, bytes]:
    """Read only the ID3 header/body, avoiding loading the full MP3 for scans."""

    with path.open("rb") as handle:
        header = handle.read(10)
        if len(header) < 10 or header[:3] != b"ID3":
            return 3, 0, b""
        major = header[3]
        flags = header[5]
        tag_size = syncsafe_to_int(header[6:10])
        if major not in (3, 4):
            return 3, flags, b""
        return major, flags, handle.read(tag_size)


def iter_frames(tag_body: bytes, major: int, flags: int = 0):
    """Yield parsed ID3 frames as (frame_id, raw_frame, payload)."""

    offset = 0
    if flags & 0x40 and len(tag_body) >= 4:
        if major == 4:
            ext_size = syncsafe_to_int(tag_body[:4])
            offset = max(ext_size, 4)
        else:
            ext_size = int.from_bytes(tag_body[:4], "big")
            offset = 4 + ext_size

    while offset + 10 <= len(tag_body):
        header = tag_body[offset : offset + 10]
        frame_id = header[:4].decode("latin1", errors="ignore")
        if not re.fullmatch(r"[A-Z0-9]{4}", frame_id):
            break
        size = frame_size(header[4:8], major)
        end = offset + 10 + size
        if size <= 0 or end > len(tag_body):
            break
        yield frame_id, tag_body[offset:end], tag_body[offset + 10 : end]
        offset = end


def preserved_frames(tag_body: bytes, major: int, flags: int = 0, replace_frames: set[str] | None = None) -> list[bytes]:
    """Return raw frames except the ones that will be rewritten."""

    replace_frames = replace_frames or TARGET_FRAMES
    return [raw for frame_id, raw, _payload in iter_frames(tag_body, major, flags) if frame_id not in replace_frames]


def decode_text_frame(payload: bytes) -> str:
    """Decode an ID3 text-frame payload into a Python string."""

    if not payload:
        return ""
    encoding = payload[0]
    data = payload[1:]
    codec = {
        0: "latin1",
        1: "utf-16",
        2: "utf-16-be",
        3: "utf-8",
    }.get(encoding, "latin1")
    return data.decode(codec, errors="replace").replace("\x00", "").strip()


def read_existing_fields(path: Path) -> dict[str, str]:
    """Read the supported text fields currently stored in an MP3."""

    major, flags, old_body = read_id3_header_and_body(path)
    fields: dict[str, str] = {}
    if not old_body:
        return fields
    for frame_id, _raw, payload in iter_frames(old_body, major, flags):
        if frame_id not in TARGET_FRAMES:
            continue
        text = decode_text_frame(payload)
        if not text:
            continue
        for field, frame_ids in FIELD_FRAMES.items():
            if frame_id in frame_ids and field not in fields:
                fields[field] = text
    return fields


def inferred_values(tags: Tags) -> dict[str, str]:
    """Convert optional tag attributes into the dict used by planning/writing."""

    values = {
        "artist": tags.artist,
        "album": tags.album,
        "title": tags.title,
    }
    if tags.track:
        values["track"] = tags.track
    if tags.year:
        values["year"] = tags.year
    return values


def looks_like_hebrew_mojibake(text: str) -> bool:
    """Detect common Hebrew mojibake from cp1255 bytes decoded as Latin-1."""

    if not text:
        return False
    has_hebrew = any("\u0590" <= char <= "\u05ff" for char in text)
    has_cp1255_latin = any("\u00e0" <= char <= "\u00fa" for char in text)
    return has_cp1255_latin and not has_hebrew


def missing_fields(tags: Tags, existing: dict[str, str]) -> list[str]:
    return [field for field, value in inferred_values(tags).items() if value and not existing.get(field, "").strip()]


def repairable_mojibake_fields(tags: Tags, existing: dict[str, str]) -> list[str]:
    """Find existing fields that are present but should be replaced as mojibake."""

    return [
        field
        for field, value in inferred_values(tags).items()
        if value and looks_like_hebrew_mojibake(existing.get(field, ""))
    ]


def write_id3(path: Path, tags: Tags, replace_fields: list[str]) -> None:
    """Rewrite selected text frames while preserving audio and unrelated frames."""

    blob = path.read_bytes()
    flags = tag_flags(blob)
    major, old_body, audio = split_existing_tag(blob)
    if flags & 0x80:
        raise ValueError("existing tag uses unsynchronisation; skipping to avoid corrupting preserved frames")

    replace_frames = set().union(*(FIELD_FRAMES[field] for field in replace_fields))
    frames = preserved_frames(old_body, major, flags, replace_frames)
    values = inferred_values(tags)
    for field in replace_fields:
        value = values.get(field)
        if value:
            frames.append(make_text_frame(frame_id_for_field(field, major), value, major))

    body = b"".join(frames) + (b"\x00" * 2048)
    header = b"ID3" + bytes([major, 0, 0]) + int_to_syncsafe(len(body))
    tmp = path.with_name(path.name + ".tagtmp")
    tmp.write_bytes(header + body + audio)
    os.replace(tmp, path)


def clean_album_part(text: str) -> tuple[str, str | None]:
    """Remove common release-junk from the album part and extract a year."""

    album = text.strip()
    year = None

    start = re.match(r"^(?P<year>(?:19|20)\d{2})\s+(.+)$", album)
    if start:
        year = start.group("year")
        album = start.group(2).strip()

    for pattern in (
        r"\s*@\s*\d{2,4}\s*$",
        r"\s+Mp3\s*(?:\(\s*\d+\s*kbps\s*\))?\s*$",
        r"\s*\[\s*(?!19\d{2}|20\d{2})[^\]]+\]\s*$",
        r"\s*\(\s*\d+\s*kbps\s*\)\s*$",
    ):
        album = re.sub(pattern, "", album, flags=re.IGNORECASE).strip()

    paren_year = re.search(r"[\s([]((?:19|20)\d{2})[\])]\s*$", album)
    if paren_year:
        year = year or paren_year.group(1)
        album = album[: paren_year.start()].strip()

    trailing_year = re.search(r"\s+((?:19|20)\d{2})\s*$", album)
    if trailing_year:
        year = year or trailing_year.group(1)
        album = album[: trailing_year.start()].strip()

    return album.strip(" -_"), year


def parse_album_dir(name: str) -> tuple[str, str, str | None]:
    """Infer artist, album, and optional year from an album directory name."""

    if " - " in name:
        artist, album_part = name.split(" - ", 1)
        album, year = clean_album_part(album_part)
        return artist.strip(), album or artist.strip(), year
    album, year = clean_album_part(name)
    return album, album, year


def parse_track_title(path: Path) -> tuple[str | None, str]:
    """Infer track number and title from ordinary album filenames."""

    stem = path.stem.strip()
    match = re.match(r"^(?P<track>\d{1,3})(?P<sep>\s*[-._]\s*|\s+)(?P<title>.+)$", stem)
    if match:
        track_num = int(match.group("track"))
        sep = match.group("sep")
        if track_num and (len(match.group("track")) <= 2 or re.search(r"[-._]", sep)):
            return str(track_num), match.group("title").strip()
    match = re.match(r"^.+?\s*-\s*(?P<track>\d{1,3})\s*-\s*(?P<title>.+)$", stem)
    if match:
        return str(int(match.group("track"))), match.group("title").strip()
    return None, stem


def normalize_rel_dir(text: str) -> str:
    """Normalize a command-line relative directory prefix."""

    normalized = text.strip().replace("\\", "/").strip("/")
    return "" if normalized in ("", ".") else normalized


def matches_rel_prefix(rel_path: Path, prefix: str) -> bool:
    normalized = normalize_rel_dir(prefix)
    if not normalized:
        return True
    rel = rel_path.as_posix()
    return rel == normalized or rel.startswith(normalized + "/")


def uses_artist_title_filename(path: Path, root: Path, artist_title_dirs: tuple[str, ...]) -> bool:
    """Return whether this path should parse filenames as '<artist> - <title>'."""

    rel_path = path.relative_to(root)
    return any(matches_rel_prefix(rel_path, prefix) for prefix in artist_title_dirs)


def parse_artist_title_filename(path: Path) -> tuple[str | None, str | None, str]:
    """Parse playlist/radio-style filenames as optional track, artist, title."""

    stem = path.stem.strip()
    track = None
    rest = stem
    match = re.match(r"^\d+[._](?P<track>\d{1,3})\s*[-._]\s*(?P<rest>.+)$", stem)
    if match and " - " in match.group("rest"):
        track = str(int(match.group("track")))
        rest = match.group("rest").strip()
    else:
        match = re.match(r"^(?P<track>\d{1,3})\s*[-._]\s*(?P<rest>.+)$", stem)
        if match and " - " in match.group("rest"):
            track = str(int(match.group("track")))
            rest = match.group("rest").strip()
    if " - " in rest:
        artist, title = rest.split(" - ", 1)
        return track, artist.strip(), title.strip()
    return track, None, rest


def album_dir_for(path: Path) -> Path:
    """Return the directory whose name should be treated as the album source."""

    if re.fullmatch(r"(?i)(?:cd|disc|disk)\s*\d+", path.parent.name.strip()):
        return path.parent.parent
    return path.parent


def should_skip(path: Path, root: Path, excluded: set[str], skip_root_files: bool) -> str | None:
    """Return a reason to skip a file, or None if it should be processed."""

    rel_parts = path.relative_to(root).parts
    if skip_root_files and len(rel_parts) == 1:
        return "loose file in root"
    if any(part in excluded for part in rel_parts[:-1]):
        return "excluded directory"
    return None


def iter_mp3s(root: Path) -> list[Path]:
    """Return MP3 files below root in stable order."""

    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".mp3")


def load_cache(root: Path) -> dict:
    """Load the per-root .tag-cache file used to avoid rereading unchanged tags."""

    cache_path = root / ".tag-cache"
    if not cache_path.exists():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "files": {}}
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return {"version": 1, "files": {}}
    return data


def write_cache(root: Path, cache: dict) -> None:
    """Atomically write the per-root .tag-cache file."""

    cache_path = root / ".tag-cache"
    tmp = root / ".tag-cache.tmp"
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, cache_path)


def file_signature(path: Path) -> tuple[int, int]:
    """Return the cache invalidation signature for a file."""

    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def cached_existing_fields(path: Path, root: Path, cache_files: dict) -> tuple[dict[str, str], str, dict | None]:
    """Return existing fields from cache or by reading the file.

    The returned cache_entry is None when the current cache entry was valid;
    otherwise callers should store it after planning.
    """

    key = path.relative_to(root).as_posix()
    size, mtime_ns = file_signature(path)
    cached = cache_files.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("size") == size
        and cached.get("mtime_ns") == mtime_ns
        and isinstance(cached.get("fields"), dict)
    ):
        return dict(cached["fields"]), key, None

    fields = read_existing_fields(path)
    entry = {"size": size, "mtime_ns": mtime_ns, "fields": fields}
    return fields, key, entry


def planned_update(
    path: Path,
    root: Path,
    overwrite: bool,
    cache_files: dict,
    artist_title_dirs: tuple[str, ...],
    manual_overrides: dict[str, str],
    selected_replace_fields: tuple[str, ...],
    replace_hebrew_mojibake: bool,
) -> Analysis:
    """Build the edit plan for one file.

    This function does all read-only work needed for a file and is safe to run
    in worker threads. Actual writes stay serialized in main().
    """
    try:
        # Start with the regular album-directory and track-filename rules.
        album_dir = album_dir_for(path)
        artist, album, year = parse_album_dir(album_dir.name)
        track, title = parse_track_title(path)
        matched_artist_title = True

        # Optional playlist/radio mode: the filename owns artist/title, while
        # the containing directory still owns album.
        if uses_artist_title_filename(path, root, artist_title_dirs):
            filename_track, filename_artist, filename_title = parse_artist_title_filename(path)
            matched_artist_title = filename_artist is not None
            artist = filename_artist or artist
            title = filename_title
            track = filename_track

        # Manual --set-* values win over anything inferred from paths.
        values = inferred_values(Tags(artist=artist, album=album, title=title, track=track, year=year))
        values.update(manual_overrides)
        tags = Tags(
            artist=values.get("artist", artist),
            album=values.get("album", album),
            title=values.get("title", title),
            track=values.get("track", track),
            year=values.get("year", year),
        )
        cache_key = None
        cache_entry = None
        existing = None
        if overwrite:
            # --overwrite means replace every field we can infer.
            replace_fields = [field for field in FIELD_ORDER if field in values]
        else:
            existing, cache_key, cache_entry = cached_existing_fields(path, root, cache_files)
            selected_fields = [field for field in FIELD_ORDER if field in selected_replace_fields and field in values]
            manual_fields = [field for field in FIELD_ORDER if field in manual_overrides and field in values]
            if selected_fields:
                # --replace-field is forceful for matching files, but with
                # artist-title mode we avoid writing bogus artist/title values
                # when the filename has no " - " separator.
                mojibake = repairable_mojibake_fields(tags, existing) if replace_hebrew_mojibake else []
                replace_fields = selected_fields if matched_artist_title else []
                replace_fields += [field for field in mojibake if field not in replace_fields]
            elif manual_fields:
                # --set-* should only touch the manually selected fields.
                replace_fields = manual_fields
            else:
                # Default mode is conservative: fill blanks and optionally
                # repair Hebrew mojibake, but leave normal existing tags alone.
                missing = missing_fields(tags, existing)
                mojibake = repairable_mojibake_fields(tags, existing) if replace_hebrew_mojibake else []
                replace_fields = missing + [field for field in mojibake if field not in missing]
        return Analysis(
            path=path,
            tags=tags,
            values=values,
            replace_fields=replace_fields,
            existing_fields=existing,
            cache_key=cache_key,
            cache_entry=cache_entry,
            matched_artist_title=matched_artist_title,
        )
    except Exception as exc:
        return Analysis(path=path, tags=None, values={}, replace_fields=[], error=str(exc))


def default_worker_count() -> int:
    return min(32, max(4, (os.cpu_count() or 4) * 4))


def print_tags(path: Path, root: Path | None = None) -> None:
    """Print supported ID3 tags for one file."""

    label = path.name if root is None else path.relative_to(root).as_posix()
    print(label)
    fields = read_existing_fields(path)
    if not fields:
        print("  (no supported ID3 text tags)")
        return
    for field in FIELD_ORDER:
        if field in fields:
            print(f"  {field}: {fields[field]}")
    for field in sorted(k for k in fields if k not in FIELD_ORDER):
        print(f"  {field}: {fields[field]}")


def view_tags(target: Path, limit: int = 0) -> int:
    """CLI implementation for --view."""

    if target.is_file():
        if target.suffix.lower() != ".mp3":
            print(f"ERROR: not an MP3 file: {target}", file=sys.stderr)
            return 1
        print_tags(target)
        return 0
    if not target.is_dir():
        print(f"ERROR: not found: {target}", file=sys.stderr)
        return 1

    count = 0
    errors = 0
    for path in iter_mp3s(target):
        try:
            if count:
                print()
            print_tags(path, target)
            count += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR: {path.relative_to(target).as_posix()}: {exc}", file=sys.stderr)
        if limit and count >= limit:
            break
    print(f"\nviewed: {count}; errors: {errors}")
    return 1 if errors else 0


def main() -> int:
    """Parse CLI options, plan changes, and optionally write tags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="music root, default: current directory")
    parser.add_argument("--view", action="store_true", help="print current MP3 tags for a file or directory, then exit")
    parser.add_argument("--apply", action="store_true", help="write tags; without this, only prints a dry run")
    parser.add_argument("--overwrite", action="store_true", help="replace inferred fields even when existing tags are present")
    parser.add_argument("--set-artist", metavar="TEXT", help="force artist for every selected file")
    parser.add_argument("--set-album", metavar="TEXT", help="force album for every selected file")
    parser.add_argument("--set-title", metavar="TEXT", help="force title for every selected file")
    parser.add_argument("--set-track", metavar="TEXT", help="force track for every selected file")
    parser.add_argument("--set-year", metavar="TEXT", help="force year for every selected file")
    parser.add_argument(
        "--replace-field",
        action="append",
        choices=FIELD_ORDER,
        default=[],
        help="replace only this inferred field; repeatable",
    )
    parser.add_argument("--exclude", action="append", default=[], help="directory name to skip; repeatable")
    parser.add_argument("--quiet", action="store_true", help="only print errors and the final summary")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    parser.add_argument("--skip-root-files", action="store_true", help="skip MP3 files directly inside the selected root")
    parser.add_argument("--replace-hebrew-mojibake", action="store_true", help="replace fields that look like Hebrew decoded as Latin-1/cp1252")
    parser.add_argument("--workers", type=int, default=default_worker_count(), help="parallel tag readers for default fill-missing mode")
    parser.add_argument(
        "--artist-title-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="parse filenames under relative DIR as '<artist> - <title>'; repeatable, use '.' for the selected root",
    )
    parser.add_argument("--limit", type=int, default=0, help="only show/process the first N matching files")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.view:
        return view_tags(root, args.limit)

    # The default "." run treats root-level MP3s as loose singles and skips
    # them. When the user names a specific directory, root-level MP3s are
    # usually the intended target and are processed.
    skip_root_files = args.skip_root_files or args.root == "."
    excluded = set(args.exclude)
    artist_title_dirs = tuple(normalize_rel_dir(item) for item in args.artist_title_dir)
    manual_overrides = {
        field: value
        for field, value in (
            ("artist", args.set_artist),
            ("album", args.set_album),
            ("title", args.set_title),
            ("track", args.set_track),
            ("year", args.set_year),
        )
        if value is not None
    }
    selected_replace_fields = tuple(args.replace_field)
    cache = load_cache(root)
    cache_files = cache.setdefault("files", {})
    cache_dirty = False
    candidates = []
    changed = skipped = excluded_skipped = errors = inspected = 0

    # Build the candidate list up front so the progress bar has a real total.
    for path in iter_mp3s(root):
        reason = should_skip(path, root, excluded, skip_root_files)
        if reason:
            excluded_skipped += 1
            continue
        candidates.append(path)

    # Scanning existing tags is I/O bound, so do that in parallel. Writes happen
    # later in this loop, one file at a time.
    workers = 1 if args.overwrite else max(1, args.workers)
    progress = Progress(len(candidates), enabled=not args.no_progress)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = [
        executor.submit(
            planned_update,
            path,
            root,
            args.overwrite,
            cache_files,
            artist_title_dirs,
            manual_overrides,
            selected_replace_fields,
            args.replace_hebrew_mojibake,
        )
        for path in candidates
    ]
    try:
        for future in as_completed(futures):
            inspected += 1
            analysis = future.result()
            path = analysis.path
            rel = path.relative_to(root)
            if analysis.error:
                errors += 1
                print(f"ERROR: {rel}: {analysis.error}", file=sys.stderr)
                progress.draw(inspected, changed, skipped, errors)
                continue

            tags = analysis.tags
            values = analysis.values
            replace_fields = analysis.replace_fields
            if tags is None:
                errors += 1
                print(f"ERROR: {rel}: internal analysis error", file=sys.stderr)
                progress.draw(inspected, changed, skipped, errors)
                continue

            if not replace_fields:
                # Even files that do not need changes can refresh the cache.
                if analysis.cache_key and analysis.cache_entry:
                    cache_files[analysis.cache_key] = analysis.cache_entry
                    cache_dirty = True
                skipped += 1
                progress.draw(inspected, changed, skipped, errors)
                continue

            change_part = ", ".join(f"{field}={values[field]!r}" for field in replace_fields)
            if not args.quiet:
                if not args.no_progress:
                    progress.done(inspected, changed, skipped, errors)
                action = "overwrite" if args.overwrite else ("set" if manual_overrides else ("repair" if args.replace_hebrew_mojibake else "fill"))
                print(f"{rel}: {action} {change_part}")

            if args.apply:
                try:
                    write_id3(path, tags, replace_fields)
                except Exception as exc:
                    errors += 1
                    print(f"ERROR: {rel}: {exc}", file=sys.stderr)
                    progress.draw(inspected, changed, skipped, errors)
                    continue
                size, mtime_ns = file_signature(path)
                fields = {}
                if analysis.existing_fields:
                    fields.update(analysis.existing_fields)
                fields.update({field: values[field] for field in replace_fields})
                cache_files[rel.as_posix()] = {"size": size, "mtime_ns": mtime_ns, "fields": fields}
                cache_dirty = True
            elif analysis.cache_key and analysis.cache_entry:
                # Dry-runs still populate .tag-cache, which makes later runs
                # much faster while keeping the default mode non-destructive.
                cache_files[analysis.cache_key] = analysis.cache_entry
                cache_dirty = True
            changed += 1
            progress.draw(inspected, changed, skipped, errors)
            if args.limit and changed >= args.limit:
                for pending in futures:
                    pending.cancel()
                break
    finally:
        executor.shutdown(cancel_futures=True)

    progress.done(inspected, changed, skipped, errors)

    if cache_dirty:
        write_cache(root, cache)

    mode = "updated" if args.apply else "would update"
    total_skipped = skipped + excluded_skipped
    excluded_part = f"; excluded: {', '.join(sorted(excluded))}" if excluded else ""
    print(f"\n{mode}: {changed}; skipped: {total_skipped}; errors: {errors}{excluded_part}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
