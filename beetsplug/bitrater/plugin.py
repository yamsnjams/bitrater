"""Beets plugin interface for bitrater."""

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from beets.dbcore import types
from beets.library import Item, Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, UserError, decargs, input_options
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from bitrater.analyzer import AudioQualityAnalyzer
from bitrater.types import AnalysisResult

# Valid actions when a transcode is detected
TRANSCODE_ACTIONS = ("ask", "quarantine", "keep", "skip")

logger = logging.getLogger("beets.bitrater")


class BitraterPlugin(BeetsPlugin):
    name = "bitrater"
    """Plugin for analyzing audio quality and detecting transcodes."""

    def __init__(self) -> None:
        super().__init__(name="bitrater")
        self.analyzer = AudioQualityAnalyzer()
        self.config.add(
            {
                "auto": False,  # Run automatically on import
                "min_confidence": 0.7,  # Minimum confidence threshold
                "warn_transcodes": True,  # Warn about detected transcodes
                "threads": None,  # Number of analysis threads (None = auto)
                "on_transcode": "ask",  # Action for detected transcodes: ask, quarantine, keep, skip
                "quarantine_dir": None,  # Quarantine folder (default: {library}/.quarantine/)
            }
        )

        # Add new fields to the database
        self.item_types = {
            "original_bitrate": types.INTEGER,
            "original_format": types.STRING,
            "bitrate_confidence": types.FLOAT,
            "is_transcoded": types.BOOLEAN,
            "transcoded_from": types.STRING,
            "analysis_version": types.STRING,
            "analysis_date": types.STRING,
            "format_warnings": types.STRING,
        }

        # Register import listener
        self.register_listener("import_task_files", self.import_task)

    def commands(self) -> list[Subcommand]:
        """Create plugin commands."""
        analyze_cmd = Subcommand("bitrater", help="Analyze audio files to detect original bitrate")
        analyze_cmd.parser.add_option(
            "-v", "--verbose", action="store_true", help="show detailed analysis results"
        )
        analyze_cmd.parser.add_option("--threads", type="int", help="number of analysis threads")
        analyze_cmd.func = self.analyze_command
        return [analyze_cmd]

    def analyze_command(self, lib: Library, opts: Any, args: list[str]) -> None:
        """Handle the analyze command."""
        try:
            # Check if DL model is available
            if self.analyzer._dl_pipeline is None:
                logger.warning(
                    "DL model not available. Ensure ONNX model files are bundled in bitrater/models/"
                )

            # Get items to analyze
            items = lib.items(decargs(args)) if args else lib.items()
            items = list(items)  # Materialize query

            if not items:
                logger.info("No items to analyze")
                return

            # Configure threading (conservative default: 50% of CPUs to prevent system overload)
            thread_count = opts.threads or self.config["threads"].get()
            if thread_count is None:
                cpu_count = os.cpu_count() or 1
                thread_count = max(1, int(cpu_count * 0.5))
            # Validate thread count
            if thread_count < 1:
                thread_count = 1
            thread_count = int(thread_count)

            # Analyze files
            results = self._analyze_items(items, thread_count)
            self._process_results(items, results, opts.verbose)

        except Exception as e:
            raise UserError(f"Analysis failed: {e}") from e

    def _analyze_items(
        self, items: Sequence[Item], thread_count: int
    ) -> list[AnalysisResult | None]:
        """Analyze multiple items in parallel using joblib.

        Uses threading backend to share the DL pipeline in memory.
        Thread limiting must happen INSIDE each worker since numba's
        thread settings are per-thread.
        """
        logger.info(f"Analyzing {len(items)} files using {thread_count} threads")

        # Extract paths - Item objects shouldn't be passed to parallel workers
        paths = [item.path.decode("utf-8") if isinstance(item.path, bytes) else str(item.path) for item in items]

        # Use threading backend to share the DL pipeline in memory
        # Thread limiting happens inside _analyze_single_file via numba.set_num_threads(1)
        # and threadpool_limits (threading backend doesn't support inner_max_num_threads)
        results = Parallel(n_jobs=thread_count, backend="threading")(
            delayed(self._analyze_single_file)(path) for path in paths
        )

        return list(results)

    def _analyze_single_file(self, file_path: str) -> AnalysisResult | None:
        """Analyze a single audio file.

        Called by joblib workers - uses shared analyzer instance since
        threading backend shares memory. Thread limits must be set HERE
        (inside the worker) because numba's settings are per-thread.
        """
        import numba

        # Limit threads INSIDE worker - numba settings are per-thread
        numba.set_num_threads(1)

        with threadpool_limits(limits=1, user_api="blas"):
            try:
                return self.analyzer.analyze_file(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                return None

    def _process_results(
        self,
        items: Sequence[Item],
        results: Sequence[AnalysisResult | None],
        verbose: bool,
    ) -> None:
        """Process and store analysis results."""
        total = 0
        transcodes = 0
        quarantined = 0
        skipped = 0
        low_confidence = 0

        min_confidence = self.config["min_confidence"].get()

        for item, result in zip(items, results, strict=True):
            if not result:
                continue

            total += 1

            # Update item with analysis results
            item.original_bitrate = result.original_bitrate
            item.original_format = result.original_format
            item.bitrate_confidence = result.confidence
            item.is_transcoded = result.is_transcode
            item.transcoded_from = result.transcoded_from or ""
            item.analysis_version = result.analysis_version
            item.analysis_date = result.analysis_date.isoformat()
            item.format_warnings = "; ".join(result.warnings)

            if result.confidence < min_confidence:
                low_confidence += 1

            if verbose:
                self._print_analysis(item, result)

            item.store()

            # Handle transcodes after storing analysis results
            if result.is_transcode:
                transcodes += 1
                action = self._handle_transcode(item, result)
                if action == "quarantine":
                    quarantined += 1
                elif action == "skip":
                    skipped += 1

        # Print summary
        self._print_summary(total, transcodes, low_confidence, quarantined, skipped)

    def _print_analysis(self, item: Item, result: AnalysisResult) -> None:
        """Log detailed analysis results for an item."""
        logger.info(f"\n{item.title}")
        logger.info("-" * 50)
        logger.info(f"Path: {item.path}")
        logger.info(f"File format: {result.file_format}")
        logger.info(f"Stated bitrate: {result.stated_bitrate or 'N/A'} kbps")
        logger.info(f"Detected original: {result.original_format} ({result.original_bitrate} kbps)")
        logger.info(f"Confidence: {result.confidence:.1%}")

        if result.is_transcode:
            logger.info(f"[WARN] TRANSCODE DETECTED - appears to be from {result.transcoded_from}")

        if result.warnings:
            logger.info("\nWarnings:")
            for warning in result.warnings:
                logger.info(f"  - {warning}")

    def _print_summary(
        self, total: int, transcodes: int, low_confidence: int,
        quarantined: int = 0, skipped: int = 0,
    ) -> None:
        """Log analysis summary."""
        logger.info("\n" + "=" * 50)
        logger.info("Analysis Summary")
        logger.info("=" * 50)
        logger.info(f"Total files analyzed: {total}")

        if transcodes > 0:
            logger.info(f"[WARN] Potential transcodes detected: {transcodes}")
            if quarantined > 0:
                logger.info(f"  Quarantined: {quarantined}")
            if skipped > 0:
                logger.info(f"  Removed: {skipped}")
            kept = transcodes - quarantined - skipped
            if kept > 0:
                logger.info(f"  Kept: {kept}")
        if low_confidence > 0:
            logger.info(f"[WARN] Low confidence results: {low_confidence}")

        if transcodes == 0 and low_confidence == 0:
            logger.info("[OK] All files appear to be original quality")

    def _get_quarantine_dir(self) -> Path:
        """Get the quarantine directory, creating it if needed.

        Uses configured path, or defaults to {library}/.quarantine/.
        """
        configured = self.config["quarantine_dir"].get()
        if configured:
            qdir = Path(configured)
        else:
            from beets import config as beets_config
            lib_dir = beets_config["directory"].as_filename()
            qdir = Path(lib_dir) / ".quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        return qdir

    def _quarantine_item(self, item: Item) -> bool:
        """Move a file to the quarantine directory.

        Preserves the filename and removes the item from the beets library.
        Returns True if quarantine succeeded.
        """
        file_path = item.path.decode("utf-8") if isinstance(item.path, bytes) else str(item.path)
        src = Path(file_path)
        if not src.exists():
            logger.warning(f"Cannot quarantine — file not found: {src}")
            return False

        dest_dir = self._get_quarantine_dir()
        dest = dest_dir / src.name

        # Handle name collisions
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            shutil.move(str(src), str(dest))
            item.remove(delete=False)  # Remove from library but don't delete (already moved)
            logger.info(f"Quarantined: {src.name} -> {dest}")
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine {src}: {e}")
            return False

    def _skip_item(self, item: Item) -> bool:
        """Remove a transcoded item from the library and delete the file."""
        try:
            item.remove(delete=True)
            logger.info(f"Removed transcoded file: {item.title}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove {item.title}: {e}")
            return False

    def _handle_transcode(self, item: Item, result: AnalysisResult) -> str:
        """Handle a detected transcode based on config.

        Returns the action taken: 'keep', 'quarantine', 'skip'.
        """
        action = self.config["on_transcode"].get()
        if action not in TRANSCODE_ACTIONS:
            logger.warning(f"Invalid on_transcode value '{action}', defaulting to 'ask'")
            action = "ask"

        file_path = item.path.decode("utf-8") if isinstance(item.path, bytes) else str(item.path)
        filename = Path(file_path).name

        if action == "ask":
            import sys
            if not sys.stdin.isatty():
                logger.warning(
                    f"Non-interactive session, keeping transcode: {filename} "
                    f"(claimed {result.stated_bitrate or '?'}kbps, "
                    f"actual {result.original_bitrate}kbps)"
                )
                return "keep"

            try:
                logger.info(
                    f"\n  Transcode detected: {filename}\n"
                    f"  Claimed: {result.stated_bitrate or '?'}kbps | "
                    f"Actual: {result.original_format} ({result.original_bitrate}kbps) | "
                    f"Confidence: {result.confidence:.0%}"
                )
                choice = input_options(
                    ("Keep", "Quarantine", "Skip"),
                    prompt="Action?",
                    default="k",
                )
            except Exception:
                logger.warning(
                    f"Could not prompt for transcode action, keeping: {filename}"
                )
                return "keep"

            if choice == "q":
                action = "quarantine"
            elif choice == "s":
                action = "skip"
            else:
                action = "keep"

        if action == "quarantine":
            if self._quarantine_item(item):
                return "quarantine"
            return "keep"  # Quarantine failed, keep the file

        if action == "skip":
            if self._skip_item(item):
                return "skip"
            return "keep"

        # keep — log the transcode for visibility
        file_path = item.path.decode("utf-8") if isinstance(item.path, bytes) else str(item.path)
        logger.warning(
            f"Transcode kept: {Path(file_path).name} "
            f"(claimed {result.stated_bitrate or '?'}kbps, "
            f"actual {result.original_bitrate}kbps, "
            f"confidence {result.confidence:.0%})"
        )
        return "keep"

    def import_task(self, session: Any, task: Any) -> None:
        """Automatically analyze files during import if enabled.

        Args:
            session: Beets import session (beets.importer.ImportSession)
            task: Beets import task (beets.importer.ImportTask)
        """
        logger.info("bitrater: import_task hook fired")
        if not self.config["auto"].get():
            logger.info("bitrater: auto is disabled, skipping")
            return

        if self.analyzer._dl_pipeline is None:
            logger.info("bitrater: no model available, skipping")
            return

        logger.info(f"bitrater: analyzing {len(task.items)} items")
        for item in task.items:
            try:
                file_path = item.path.decode('utf-8') if isinstance(item.path, bytes) else str(item.path)
                result = self.analyzer.analyze_file(file_path)
                if result:
                    item.original_bitrate = result.original_bitrate
                    item.original_format = result.original_format
                    item.bitrate_confidence = result.confidence
                    item.is_transcoded = result.is_transcode
                    item.transcoded_from = result.transcoded_from or ""
                    item.analysis_version = result.analysis_version
                    item.analysis_date = result.analysis_date.isoformat()
                    item.format_warnings = "; ".join(result.warnings)

                    item.store()
                    logger.info(
                        f"bitrater: {item.title} -> {result.original_format} "
                        f"({result.original_bitrate}kbps, confidence={result.confidence:.0%})"
                    )

                    if result.is_transcode:
                        self._handle_transcode(item, result)
                else:
                    logger.warning(f"bitrater: no result for {file_path}")
            except Exception as e:
                logger.error(f"Auto-analysis failed for {item.path}: {e}")
