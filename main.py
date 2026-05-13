#!/usr/bin/env python3
"""
music-meta — scan, report, and fix audio file metadata

Usage:
  music-meta.py <root> scan
  music-meta.py <root> missing
  music-meta.py <root> wrong
  music-meta.py <root> dirs
  music-meta.py <root> fix-missing [--dry-run]
  music-meta.py <root> fix-field <field> (--value VALUE | --from-dir)
                        [--path-filter PATTERN] [--dry-run]

Requirements: pip install mutagen rich
"""

import argparse
import fnmatch
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from mutagen.asf import ASF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, TALB, TDRC, TIT2, TPE1, TRCK, ID3NoHeaderError
    from mutagen.mp4 import MP4
except ImportError:
    sys.exit("mutagen not installed: pip install mutagen rich")

try:
    from rich.console import Console
    from rich.table import Table

    console = Console()
except ImportError:
    console = None


AUDIO_EXT = {".mp3", ".m4a", ".wma", ".flac"}
CORE_FIELDS = ["artist", "album", "title"]
ALL_FIELDS = ["artist", "album", "title", "year", "track"]


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Meta:
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    year: Optional[str] = None
    track: Optional[str] = None

    def missing_core(self) -> list[str]:
        return [f for f in CORE_FIELDS if not getattr(self, f)]


# ── Reading metadata ──────────────────────────────────────────────────────────


def _clean(v) -> Optional[str]:
    s = str(v).strip() if v is not None else ""
    return s or None


def read_meta(path: Path) -> Meta:
    m = Meta()
    ext = path.suffix.lower()
    try:
        if ext == ".flac":
            f = FLAC(path)

            def fg(key):
                v = f.get(key.lower()) or f.get(key.upper()) or []
                return _clean(v[0]) if v else None

            m.artist = fg("artist")
            m.album = fg("album")
            m.title = fg("title")
            m.year = fg("date") or fg("year")
            tr = fg("tracknumber")
            m.track = tr.split("/")[0] if tr else None

        elif ext == ".mp3":
            try:
                f = ID3(path)
            except ID3NoHeaderError:
                return m
            m.artist = _clean(str(f["TPE1"])) if "TPE1" in f else None
            m.album = _clean(str(f["TALB"])) if "TALB" in f else None
            m.title = _clean(str(f["TIT2"])) if "TIT2" in f else None
            yr = _clean(str(f["TDRC"])) if "TDRC" in f else None
            m.year = yr[:4] if yr and len(yr) >= 4 else yr
            tr = _clean(str(f["TRCK"])) if "TRCK" in f else None
            m.track = tr.split("/")[0] if tr else None

        elif ext == ".m4a":
            f = MP4(path)
            tags = f.tags or {}

            def mp(k):
                v = tags.get(k, [])
                return _clean(v[0]) if v else None

            m.artist = mp("©ART")
            m.album = mp("©alb")
            m.title = mp("©nam")
            yr = mp("©day")
            m.year = yr[:4] if yr and len(yr) >= 4 else yr
            trkn = tags.get("trkn")
            m.track = str(trkn[0][0]) if trkn else None

        elif ext == ".wma":
            f = ASF(path)
            tags = f.tags or {}

            def af(k):
                v = tags.get(k, [])
                return _clean(str(v[0])) if v else None

            m.artist = af("Author")
            m.album = af("WM/AlbumTitle")
            m.title = af("Title")
            m.year = af("WM/Year")
            m.track = af("WM/TrackNumber")

    except Exception:
        pass
    return m


# ── Writing metadata ──────────────────────────────────────────────────────────


def write_meta(path: Path, updates: dict, dry_run: bool = False) -> bool:
    if not updates or dry_run:
        return True
    ext = path.suffix.lower()
    try:
        if ext == ".flac":
            f = FLAC(path)
            fmap = {
                "artist": "artist",
                "album": "album",
                "title": "title",
                "year": "date",
                "track": "tracknumber",
            }
            for fld, val in updates.items():
                if val and fld in fmap:
                    f[fmap[fld]] = [val]
            f.save()

        elif ext == ".mp3":
            try:
                f = ID3(path)
            except ID3NoHeaderError:
                f = ID3()
            tmap = {
                "title": lambda v: TIT2(encoding=3, text=v),
                "artist": lambda v: TPE1(encoding=3, text=v),
                "album": lambda v: TALB(encoding=3, text=v),
                "year": lambda v: TDRC(encoding=3, text=v),
                "track": lambda v: TRCK(encoding=3, text=v),
            }
            for fld, val in updates.items():
                if val and fld in tmap:
                    f.add(tmap[fld](val))
            f.save(path)

        elif ext == ".m4a":
            f = MP4(path)
            if f.tags is None:
                f.add_tags()
            mmap = {"artist": "©ART", "album": "©alb", "title": "©nam", "year": "©day"}
            for fld, val in updates.items():
                if val and fld in mmap:
                    f.tags[mmap[fld]] = [val]
            f.save()

        elif ext == ".wma":
            f = ASF(path)
            amap = {
                "artist": "Author",
                "album": "WM/AlbumTitle",
                "title": "Title",
                "year": "WM/Year",
                "track": "WM/TrackNumber",
            }
            for fld, val in updates.items():
                if val and fld in amap:
                    f.tags[amap[fld]] = [val]
            f.save()

        return True
    except Exception as e:
        print(f"Error writing {path}: {e}", file=sys.stderr)
        return False


# ── Path-based extraction ─────────────────────────────────────────────────────

_YEAR_BRACKET = re.compile(r"\[(\d{4})\]")
_YEAR_PLAIN = re.compile(r"\b((?:19|20)\d{2})\b")
_DASH_SEP = re.compile(r"\s+-\s+")
_TRACK_PREFIX = re.compile(r"^(\d+)[.\s\-]+(.+)$")


def _pull_year(s: str) -> tuple[Optional[str], str]:
    m = _YEAR_BRACKET.search(s)
    if m:
        return m.group(1), (s[: m.start()] + s[m.end() :]).strip(" -_")
    m = _YEAR_PLAIN.search(s)
    if m:
        return m.group(1), (s[: m.start()] + s[m.end() :]).strip(" -_")
    return None, s


def _parse_dir(name: str) -> dict:
    year, rest = _pull_year(name.strip())
    result = {}
    if year:
        result["year"] = year
    parts = [p.strip() for p in _DASH_SEP.split(rest) if p.strip()]
    if len(parts) >= 2:
        result["artist"] = parts[0]
        result["album"] = parts[1]
    elif len(parts) == 1:
        result["_solo"] = parts[0]  # ambiguous: could be artist or album
    return result


def _parse_stem(stem: str) -> dict:
    m = _TRACK_PREFIX.match(stem.strip())
    if m:
        return {"track": m.group(1), "title": m.group(2).strip()}
    return {"title": stem.strip()}


def extract_from_path(file_path: Path, root: Path) -> Meta:
    m = Meta()
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = file_path

    parts = rel.parts
    dir_parts = parts[:-1]

    fm = _parse_stem(file_path.stem)
    m.title = fm.get("title")
    m.track = fm.get("track")

    if len(dir_parts) == 0:
        pass

    elif len(dir_parts) == 1:
        d = _parse_dir(dir_parts[0])
        m.artist = d.get("artist")
        m.album = d.get("album") or d.get("_solo")
        m.year = d.get("year")

    elif len(dir_parts) == 2:
        d0, d1 = _parse_dir(dir_parts[0]), _parse_dir(dir_parts[1])
        if d0.get("artist") and d0.get("album"):
            # e.g. "Artist - Album [Year]" / "Disc 1"
            m.artist = d0["artist"]
            m.album = d0["album"]
            m.year = d0.get("year") or d1.get("year")
        else:
            # e.g. "Artist" / "Album [Year]"
            m.artist = d0.get("artist") or d0.get("_solo")
            m.album = d1.get("album") or d1.get("artist") or d1.get("_solo")
            m.year = d0.get("year") or d1.get("year")

    else:  # 3+ levels
        d0, d1 = _parse_dir(dir_parts[0]), _parse_dir(dir_parts[1])
        if d0.get("artist") and d0.get("album"):
            m.artist = d0["artist"]
            m.album = d0["album"]
            m.year = d0.get("year")
        elif d1.get("artist") and d1.get("album"):
            m.artist = d1["artist"]
            m.album = d1["album"]
            m.year = d1.get("year")
        else:
            m.artist = d0.get("artist") or d0.get("_solo")
            m.album = d1.get("album") or d1.get("artist") or d1.get("_solo")
            m.year = d0.get("year") or d1.get("year")

    return m


# ── Scanning ──────────────────────────────────────────────────────────────────


def scan_files(root: Path) -> list[tuple[Path, Meta, Meta]]:
    results = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            results.append((p, read_meta(p), extract_from_path(p, root)))
    return results


def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower() if s else ""


def wrong_fields(actual: Meta, extracted: Meta) -> list[str]:
    wrong = []
    for fld in CORE_FIELDS + ["year"]:
        a, e = _norm(getattr(actual, fld)), _norm(getattr(extracted, fld))
        if a and e and a != e:
            wrong.append(fld)
    return wrong


# ── Output helpers ────────────────────────────────────────────────────────────


def _print(line="", markup=True):
    if console:
        console.print(line, markup=markup)
    else:
        # strip rich markup for plain output
        print(re.sub(r"\[/?[^\]]*\]", "", str(line)))


def _table(*headers):
    if not console:
        return None
    t = Table(show_header=True, header_style="bold cyan", expand=False)
    for h in headers:
        t.add_column(h, overflow="fold")
    return t


def _render(table, rows):
    if console and table is not None:
        for row in rows:
            table.add_row(*[str(c) if c is not None else "" for c in row])
        console.print(table)
    else:
        widths = (
            [
                max(len(str(r[i])) for r in rows) if rows else 0
                for i in range(len(rows[0]))
            ]
            if rows
            else []
        )
        for row in rows:
            print("  ".join(str(c or "").ljust(w) for c, w in zip(row, widths)))


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_scan(root, results):
    total = len(results)
    miss = sum(1 for _, a, _ in results if a.missing_core())
    wrong = sum(1 for _, a, e in results if wrong_fields(a, e))
    _print(f"[bold]Root:[/bold]                    {root}")
    _print(f"[bold]Total audio files:[/bold]       {total}")
    _print(f"[bold]Missing core metadata:[/bold]   [yellow]{miss}[/yellow]")
    _print(f"[bold]Potentially wrong metadata:[/bold] [red]{wrong}[/red]")


def cmd_missing(root, results):
    rows = []
    for path, actual, _ in results:
        missing = actual.missing_core()
        if missing:
            rows.append(
                (
                    str(path.relative_to(root)),
                    ", ".join(missing),
                    actual.artist or "—",
                    actual.album or "—",
                    actual.title or "—",
                )
            )
    if not rows:
        _print("[green]No files with missing metadata.[/green]")
        return
    t = _table("File", "Missing fields", "Artist", "Album", "Title")
    _render(t, rows)
    _print(f"\n[yellow]{len(rows)}[/yellow] file(s) with missing metadata.")


def cmd_wrong(root, results):
    rows = []
    for path, actual, extracted in results:
        wf = wrong_fields(actual, extracted)
        if not wf:
            continue

        def diff(fld):
            a, e = getattr(actual, fld), getattr(extracted, fld)
            if fld in wf:
                return f"{a}  →  {e}"
            return a or "—"

        rows.append(
            (
                str(path.relative_to(root)),
                ", ".join(wf),
                diff("artist"),
                diff("album"),
                diff("title"),
                diff("year"),
            )
        )
    if not rows:
        _print("[green]No files with mismatched metadata.[/green]")
        return
    t = _table(
        "File", "Mismatched", "Artist (actual→extracted)", "Album", "Title", "Year"
    )
    _render(t, rows)
    _print(f"\n[red]{len(rows)}[/red] file(s) with potentially wrong metadata.")


def cmd_dirs(root, results):
    dirs: dict[Path, list] = {}
    for path, actual, extracted in results:
        dirs.setdefault(path.parent, []).append((path, actual, extracted))

    rows = []
    for d in sorted(dirs):
        _, _, ex = dirs[d][0]
        rows.append(
            (
                str(d.relative_to(root)) or ".",
                str(len(dirs[d])),
                ex.artist or "—",
                ex.album or "—",
                ex.year or "—",
            )
        )
    t = _table(
        "Directory", "#", "Extracted artist", "Extracted album", "Extracted year"
    )
    _render(t, rows)
    _print(f"\n{len(rows)} director(ies) scanned.")


def cmd_fix_missing(root, results, dry_run: bool):
    updated = 0
    for path, actual, extracted in results:
        missing = actual.missing_core()
        if not actual.year:
            missing.append("year")
        if not missing:
            continue
        updates = {f: getattr(extracted, f) for f in missing if getattr(extracted, f)}
        if not updates:
            continue
        label = "[DRY RUN] " if dry_run else ""
        _print(f"{label}[cyan]{path.relative_to(root)}[/cyan]  {updates}")
        write_meta(path, updates, dry_run)
        updated += 1
    noun = "Would update" if dry_run else "Updated"
    _print(f"\n{noun} [bold]{updated}[/bold] file(s).")


def cmd_fix_field(
    root,
    results,
    field: str,
    value: Optional[str],
    from_dir: bool,
    path_filter: Optional[str],
    dry_run: bool,
):
    updated = 0
    for path, actual, extracted in results:
        rel = str(path.relative_to(root))
        if path_filter and not fnmatch.fnmatch(rel, f"*{path_filter}*"):
            continue

        val = getattr(extracted, field) if from_dir else value
        if val is None:
            continue

        current = getattr(actual, field)
        label = "[DRY RUN] " if dry_run else ""
        _print(
            f"{label}[cyan]{rel}[/cyan]  {field}: [red]{current!r}[/red] → [green]{val!r}[/green]"
        )
        write_meta(path, {field: val}, dry_run)
        updated += 1

    noun = "Would update" if dry_run else "Updated"
    _print(f"\n{noun} [bold]{updated}[/bold] file(s).")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="music-meta — audio metadata scanner and fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Summary statistics")
    p_missing = sub.add_parser("missing", help="Files with missing artist/album/title")
    p_wrong = sub.add_parser(
        "wrong", help="Files whose metadata differs from path-extracted values"
    )
    p_dirs = sub.add_parser(
        "dirs", help="Directories with their path-extracted metadata"
    )

    p_fm = sub.add_parser(
        "fix-missing", help="Fill empty fields from path-extracted values"
    )
    p_fm.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing"
    )

    p_ff = sub.add_parser("fix-field", help="Set one field on matching files")
    p_ff.add_argument("field", choices=ALL_FIELDS, help="Field to update")
    src = p_ff.add_mutually_exclusive_group(required=True)
    src.add_argument("--value", metavar="VALUE", help="Literal value to write")
    src.add_argument(
        "--from-dir", action="store_true", help="Use value extracted from path"
    )
    p_ff.add_argument(
        "--path-filter",
        metavar="GLOB",
        help="Only affect files whose relative path matches this glob substring",
    )
    p_ff.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing"
    )

    # root is last on every subparser so you can reuse the same flags and just change the path
    for p in [p_scan, p_missing, p_wrong, p_dirs, p_fm, p_ff]:
        p.add_argument("root", help="Root music directory")

    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    _print(f"Scanning [bold]{root}[/bold] …")
    results = scan_files(root)
    _print(f"Found [bold]{len(results)}[/bold] audio file(s).\n")

    if args.cmd == "scan":
        cmd_scan(root, results)
    elif args.cmd == "missing":
        cmd_missing(root, results)
    elif args.cmd == "wrong":
        cmd_wrong(root, results)
    elif args.cmd == "dirs":
        cmd_dirs(root, results)
    elif args.cmd == "fix-missing":
        cmd_fix_missing(root, results, args.dry_run)
    elif args.cmd == "fix-field":
        cmd_fix_field(
            root,
            results,
            args.field,
            value=getattr(args, "value", None),
            from_dir=args.from_dir,
            path_filter=args.path_filter,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
