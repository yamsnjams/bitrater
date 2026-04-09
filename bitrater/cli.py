"""Standalone CLI for bitrater audio quality analysis."""

import argparse
import logging
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


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze audio files and print results."""
    from bitrater.analyzer import AudioQualityAnalyzer

    analyzer = AudioQualityAnalyzer()

    target = Path(args.target).resolve()
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(
            p
            for p in target.rglob("*")
            if p.suffix.lower() in {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}
        )
    else:
        logger.error(f"Not a file or directory: {target}")
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
