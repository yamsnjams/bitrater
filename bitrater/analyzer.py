"""Audio quality analyzer - orchestrates spectral analysis and classification."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# Suppress numba FNV hashing warning (harmless, triggers once per worker process)
warnings.filterwarnings("ignore", message="FNV hashing is not implemented", module="numba")

# Import constants (lightweight, no numpy/scipy)
from .constants import (  # noqa: E402
    BITRATE_MISMATCH_FACTOR,
    LOSSLESS_CONTAINERS,
)

# TYPE_CHECKING imports for type hints only - not imported at runtime in workers
if TYPE_CHECKING:
    from .types import AnalysisResult

logger = logging.getLogger("beets.bitrater")


class AudioQualityAnalyzer:
    """
    Orchestrates audio quality analysis pipeline.

    Uses deep learning (CNN + BiLSTM) with bundled ONNX model files.
    The model ships pre-trained and loads automatically.
    """

    def __init__(self, use_dl: bool = True):
        """
        Initialize analyzer components.

        Args:
            use_dl: Try to load the deep learning model (default True)
        """
        from .file_analyzer import FileAnalyzer
        from .transcode_detector import TranscodeDetector

        self.file_analyzer = FileAnalyzer()
        self.transcode_detector = TranscodeDetector()

        # Load DL inference pipeline
        self._dl_pipeline = None
        if use_dl:
            try:
                from .dl_inference import load_inference_pipeline

                self._dl_pipeline = load_inference_pipeline()
            except Exception as e:
                logger.warning(f"DL model not available: {e}")

    def analyze_file(self, file_path: str) -> AnalysisResult | None:
        """
        Analyze a single audio file.

        Uses deep learning pipeline. Returns None if analysis fails.
        """
        from .types import AnalysisResult

        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # 1. Get file metadata
        try:
            metadata = self.file_analyzer.analyze(file_path)
        except (ValueError, RuntimeError, KeyError) as e:
            logger.warning(f"Could not read metadata from {file_path}: {e}")
            metadata = None
        except Exception as e:
            logger.error(
                f"Unexpected error reading metadata from {file_path}: {e}",
                exc_info=True,
            )
            metadata = None

        file_format = path.suffix.lower().lstrip(".")
        stated_bitrate = metadata.bitrate if metadata else None
        is_vbr = bool(metadata and metadata.encoding_type == "VBR")
        stated_class = self._get_stated_class(file_format, stated_bitrate)

        # 2. Run DL inference
        if self._dl_pipeline is not None:
            result = self._analyze_file_dl(
                file_path, path, file_format, stated_bitrate, stated_class, is_vbr,
            )
            if result is not None:
                return result
            logger.debug(f"DL inference failed for {file_path}")

        return None

    def _analyze_file_dl(
        self,
        file_path: str,
        path: Path,
        file_format: str,
        stated_bitrate: int | None,
        stated_class: str,
        is_vbr: bool = False,
    ) -> AnalysisResult | None:
        """Analyze using the deep learning pipeline."""
        from .constants import BITRATE_CLASSES, CLASS_LABELS
        from .types import AnalysisResult

        try:
            class_name, confidence, probs_dict = self._dl_pipeline.predict(file_path)
        except Exception as e:
            logger.warning(f"DL inference error for {file_path}: {e}")
            return None

        if class_name is None:
            return None

        # Guard: lossy containers (MP3, AAC, etc.) cannot be lossless by definition.
        # If the model predicts LOSSLESS for a lossy file, fall back to the
        # highest-probability non-LOSSLESS class.
        if class_name == "LOSSLESS" and file_format not in LOSSLESS_CONTAINERS:
            fallback = max(
                (cls for cls in probs_dict if cls != "LOSSLESS"),
                key=lambda cls: probs_dict[cls],
            )
            logger.debug(
                f"LOSSLESS prediction on lossy container — downgrading to {fallback}"
            )
            class_name = fallback
            confidence = probs_dict[fallback]

        # Map class name to bitrate
        class_idx = CLASS_LABELS[class_name]
        _, estimated_bitrate = BITRATE_CLASSES[class_idx]

        # Transcode detection
        transcode_result = self.transcode_detector.detect(
            stated_class=stated_class,
            detected_class=class_name,
            is_vbr=is_vbr,
        )

        warnings = []
        if transcode_result.is_transcode:
            warnings.append(
                f"File appears to be transcoded from {transcode_result.transcoded_from} "
                f"(quality gap: {transcode_result.quality_gap})"
            )

        if stated_bitrate and class_name != "LOSSLESS":
            if stated_bitrate > estimated_bitrate * BITRATE_MISMATCH_FACTOR:
                warnings.append(
                    f"Stated bitrate ({stated_bitrate} kbps) much higher than "
                    f"detected ({estimated_bitrate} kbps) - possible upsampled file"
                )

        return AnalysisResult(
            filename=str(path),
            file_format=file_format,
            original_format=class_name,
            original_bitrate=estimated_bitrate,
            confidence=confidence,
            is_transcode=transcode_result.is_transcode,
            stated_class=stated_class,
            detected_cutoff=0,
            quality_gap=transcode_result.quality_gap,
            transcoded_from=transcode_result.transcoded_from,
            stated_bitrate=stated_bitrate,
            warnings=warnings,
        )

    def _get_stated_class(self, file_format: str, stated_bitrate: int | None) -> str:
        """
        Determine stated quality class from file format and metadata.

        Args:
            file_format: Container format (mp3, flac, etc.)
            stated_bitrate: Bitrate from file metadata (if available)

        Returns:
            Quality class string: "128", "V2", "192", "V0", "256", "320", "LOSSLESS"
        """
        # Lossless containers are always stated as LOSSLESS
        if file_format in LOSSLESS_CONTAINERS:
            return "LOSSLESS"

        # For lossy containers, use bitrate to determine stated class
        if stated_bitrate is None:
            return "UNKNOWN"

        # CBR: map average bitrate to class
        if stated_bitrate <= 140:
            return "128"
        elif stated_bitrate <= 175:
            return "V2"
        elif stated_bitrate <= 210:
            return "192"
        elif stated_bitrate <= 260:
            return "V0"
        elif stated_bitrate <= 290:
            return "256"
        else:
            return "320"

    def _get_default_workers(self) -> int:
        """Get default number of workers based on CPU count (50% of available cores).

        Conservative default to prevent system overload. Heavy parallel I/O
        combined with CPU-intensive FFT work can stress the system.
        """
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count * 0.5))
