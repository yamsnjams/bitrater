"""Tests for audio quality analyzer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bitrater.analyzer import AudioQualityAnalyzer
from bitrater.types import AnalysisResult


class TestAudioQualityAnalyzer:
    """Tests for AudioQualityAnalyzer class."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = AudioQualityAnalyzer()

        assert analyzer.file_analyzer is not None
        assert analyzer.transcode_detector is not None

    def test_analyze_file_not_found(self) -> None:
        """Test analyze_file returns None for non-existent file."""
        analyzer = AudioQualityAnalyzer()
        result = analyzer.analyze_file("/nonexistent/file.mp3")
        assert result is None

    def test_analyze_file_no_dl_returns_none(self, tmp_path: Path) -> None:
        """Test analyze_file returns None when DL pipeline is not available."""
        analyzer = AudioQualityAnalyzer(use_dl=False)

        fake_audio = tmp_path / "test.mp3"
        fake_audio.write_bytes(b"fake audio content")

        result = analyzer.analyze_file(str(fake_audio))
        assert result is None


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_summarize(self) -> None:
        """Test AnalysisResult.summarize() method."""

        result = AnalysisResult(
            filename="test.mp3",
            file_format="mp3",
            original_format="320",
            original_bitrate=320,
            confidence=0.95,
            is_transcode=False,
            stated_class="320",
            detected_cutoff=20500,
            quality_gap=0,
            stated_bitrate=320,
        )

        summary = result.summarize()

        assert summary["filename"] == "test.mp3"
        assert summary["original_format"] == "320"
        assert summary["original_bitrate"] == 320
        assert summary["confidence"] == 0.95
        assert summary["is_transcode"] is False
        assert summary["stated_class"] == "320"
        assert summary["detected_cutoff"] == 20500
        assert summary["quality_gap"] == 0

    def test_transcode_result(self) -> None:
        """Test AnalysisResult for a transcoded file."""
        result = AnalysisResult(
            filename="fake_lossless.flac",
            file_format="flac",
            original_format="128",
            original_bitrate=128,
            confidence=0.92,
            is_transcode=True,
            stated_class="LOSSLESS",
            detected_cutoff=16000,
            quality_gap=6,
            transcoded_from="128",
            stated_bitrate=None,
        )

        assert result.is_transcode is True
        assert result.transcoded_from == "128"
        assert result.file_format == "flac"
        assert result.stated_class == "LOSSLESS"
        assert result.detected_cutoff == 16000
        assert result.quality_gap == 6


class TestIntegratedTranscodeDetection:
    """Test full transcode detection pipeline."""

    def test_analyze_detects_stated_class_from_container(self) -> None:
        """Analyzer should determine stated_class from file format."""
        analyzer = AudioQualityAnalyzer()

        # FLAC container = LOSSLESS stated class
        assert analyzer._get_stated_class("flac", None) == "LOSSLESS"
        # WAV container = LOSSLESS stated class
        assert analyzer._get_stated_class("wav", None) == "LOSSLESS"
        # MP3 with 320 bitrate
        assert analyzer._get_stated_class("mp3", 320) == "320"
        # MP3 with 192 bitrate
        assert analyzer._get_stated_class("mp3", 192) == "192"
        # MP3 with ~245 bitrate (V0 range)
        assert analyzer._get_stated_class("mp3", 245) == "V0"
