"""Standalone CLI for bitrater audio quality analysis."""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("bitrater")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _resolve_win_drive(path_str: str) -> str | None:
    """Resolve a Windows mapped drive letter to its UNC path.

    Tries multiple methods since drive mappings vary by how they were created:
    1. WNetGetConnectionW — works for active net use mappings
    2. Registry HKCU\\Network — works for persistent mappings
    3. net use command — broadest compatibility fallback
    """
    if os.name != "nt" or not re.match(r"^[A-Za-z]:\\", path_str):
        return None

    drive = path_str[:2]  # e.g. "E:"
    letter = drive[0].upper()
    rest = path_str[2:]  # e.g. "\foo\bar"

    # Method 1: WNetGetConnectionW
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        length = ctypes.c_ulong(512)
        if ctypes.windll.mpr.WNetGetConnectionW(drive, buf, ctypes.byref(length)) == 0:
            return buf.value + rest
    except Exception:
        pass

    # Method 2: Registry (persistent drive mappings)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"Network\{letter}") as key:
            remote_path, _ = winreg.QueryValueEx(key, "RemotePath")
            return remote_path + rest
    except Exception:
        pass

    # Method 3: Parse 'net use' output
    try:
        import subprocess
        output = subprocess.check_output(
            ["net", "use", drive], text=True, stderr=subprocess.DEVNULL
        )
        for line in output.splitlines():
            if line.strip().lower().startswith("remote name"):
                unc_root = line.split(None, 2)[-1].strip()
                return unc_root + rest
    except Exception:
        pass

    return None


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze audio files and print results."""
    from bitrater.analyzer import AudioQualityAnalyzer

    analyzer = AudioQualityAnalyzer()

    target = Path(args.target).resolve()
    if not target.is_file() and not target.is_dir():
        # On Windows, try resolving mapped drive to UNC path
        unc = _resolve_win_drive(args.target)
        if unc:
            target = Path(unc)
            logger.debug(f"Resolved mapped drive to UNC path: {target}")

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            p
            for p in target.rglob("*")
            if p.suffix.lower() in {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}
        )
    else:
        hint = ""
        if os.name == "nt" and re.match(r"^[A-Za-z]:\\", args.target):
            hint = (
                " Mapped network drives may not be visible to this process."
                " Try the UNC path instead (e.g. \\\\server\\share\\path)."
            )
        logger.error(f"Not a file or directory: {target}{hint}")
        sys.exit(1)

    if not files:
        logger.error(f"No audio files found in {target}")
        sys.exit(1)

    logger.info(f"Analyzing {len(files)} file(s)...")

    for filepath in files:
        result = analyzer.analyze_file(str(filepath))
        if result is None:
            logger.warning(f"  SKIP {filepath.name} (analysis failed)")
            continue

        status = "TRANSCODE" if result.is_transcode else "OK"
        print(
            f"[{status}] {filepath.name}: "
            f"{result.original_format} {result.original_bitrate}kbps "
            f"(confidence: {result.confidence:.0%})"
        )
        if args.verbose and result.warnings:
            for w in result.warnings:
                print(f"  warn: {w}")


def main() -> None:
    """Entry point for the bitrater CLI."""
    parser = argparse.ArgumentParser(
        prog="bitrater",
        description="Audio quality analysis and bitrate detection",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="analyze audio files")
    p_analyze.add_argument("target", help="audio file or directory to analyze")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    commands = {
        "analyze": cmd_analyze,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
