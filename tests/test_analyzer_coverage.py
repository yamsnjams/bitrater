"""Additional tests for analyzer.py to improve coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bitrater.analyzer import AudioQualityAnalyzer


class TestGetStatedClass:
    """Tests for _get_stated_class."""

    def setup_method(self):
        self.analyzer = AudioQualityAnalyzer()

    def test_flac_is_lossless(self):
        assert self.analyzer._get_stated_class("flac", None) == "LOSSLESS"

    def test_wav_is_lossless(self):
        assert self.analyzer._get_stated_class("wav", None) == "LOSSLESS"

    def test_alac_is_lossless(self):
        assert self.analyzer._get_stated_class("alac", None) == "LOSSLESS"

    def test_mp3_no_bitrate_is_unknown(self):
        assert self.analyzer._get_stated_class("mp3", None) == "UNKNOWN"

    def test_mp3_128(self):
        assert self.analyzer._get_stated_class("mp3", 128) == "128"

    def test_mp3_140_is_128(self):
        assert self.analyzer._get_stated_class("mp3", 140) == "128"

    def test_mp3_170_is_v2(self):
        assert self.analyzer._get_stated_class("mp3", 170) == "V2"

    def test_mp3_192(self):
        assert self.analyzer._get_stated_class("mp3", 192) == "192"

    def test_mp3_245_is_v0(self):
        assert self.analyzer._get_stated_class("mp3", 245) == "V0"

    def test_mp3_260_is_v0(self):
        assert self.analyzer._get_stated_class("mp3", 260) == "V0"

    def test_mp3_270_is_256(self):
        assert self.analyzer._get_stated_class("mp3", 270) == "256"

    def test_mp3_290_is_256(self):
        assert self.analyzer._get_stated_class("mp3", 290) == "256"

    def test_mp3_320(self):
        assert self.analyzer._get_stated_class("mp3", 320) == "320"


class TestAnalyzeFileWithDLModel:
    """Tests for analyze_file when the DL pipeline is available."""

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_dl_pipeline(self):
        analyzer = AudioQualityAnalyzer()

        # Mock DL pipeline
        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = ("320", 0.95, {"320": 0.95})

        # Mock other components
        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = MagicMock(
            bitrate=320, encoding_type="CBR"
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert result.original_format == "320"
        assert result.confidence == 0.95
        assert result.is_transcode is False

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_dl_returns_none_on_failure(self):
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.side_effect = RuntimeError("model error")

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        analyzer.transcode_detector = MagicMock()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_no_dl_pipeline(self):
        """When DL pipeline is None, analyze_file returns None."""
        analyzer = AudioQualityAnalyzer()
        analyzer._dl_pipeline = None

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        analyzer.transcode_detector = MagicMock()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_metadata_error(self):
        """Metadata extraction failure should not prevent analysis."""
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = ("192", 0.88, {"192": 0.88})

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.side_effect = ValueError("bad format")

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_unexpected_metadata_error(self):
        """Unexpected metadata errors should be logged but not prevent analysis."""
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = ("320", 0.90, {"320": 0.90})

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.side_effect = TypeError("unexpected")

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=False,
            quality_gap=0,
            transcoded_from=None,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_bitrate_mismatch_warning(self):
        """Bitrate mismatch should generate a warning."""
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = ("128", 0.90, {"128": 0.90})

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = MagicMock(
            bitrate=320, encoding_type="CBR"
        )

        analyzer.transcode_detector = MagicMock()
        analyzer.transcode_detector.detect.return_value = MagicMock(
            is_transcode=True,
            quality_gap=5,
            transcoded_from="128",
        )

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert any("bitrate" in w.lower() or "upsampled" in w.lower() for w in result.warnings)
        assert any("transcoded" in w.lower() for w in result.warnings)

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_transcode_detection(self):
        """FLAC container + DL detects 128 -> transcode warning."""
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = ("128", 0.92, {"128": 0.92})

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = MagicMock(
            bitrate=None, encoding_type="lossless"
        )

        from bitrater.transcode_detector import TranscodeDetector
        analyzer.transcode_detector = TranscodeDetector()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".flac") as f:
            result = analyzer.analyze_file(f.name)

        assert result is not None
        assert result.is_transcode is True
        assert any("transcoded" in w.lower() for w in result.warnings)

    @patch.object(AudioQualityAnalyzer, "__init__", lambda self, **kw: None)
    def test_analyze_file_dl_returns_none_class(self):
        """When DL returns None class_name, should return None."""
        analyzer = AudioQualityAnalyzer()

        analyzer._dl_pipeline = MagicMock()
        analyzer._dl_pipeline.predict.return_value = (None, 0.0, {})

        analyzer.file_analyzer = MagicMock()
        analyzer.file_analyzer.analyze.return_value = None

        analyzer.transcode_detector = MagicMock()

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            result = analyzer.analyze_file(f.name)

        assert result is None


class TestGetDefaultWorkers:
    """Tests for _get_default_workers."""

    def test_default_workers_uses_50_percent(self):
        analyzer = AudioQualityAnalyzer()

        with patch("os.cpu_count", return_value=10):
            assert analyzer._get_default_workers() == 5

    def test_default_workers_minimum_one(self):
        analyzer = AudioQualityAnalyzer()

        with patch("os.cpu_count", return_value=None):
            assert analyzer._get_default_workers() >= 1
