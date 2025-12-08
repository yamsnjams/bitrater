"""Tests for the deep learning inference pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from bitrater.dl_inference import (
    CLASS_NAMES,
    _MODELS_DIR,
    load_inference_pipeline,
)


class TestLoadInferencePipeline:
    """Tests for model loading."""

    def test_bundled_models_exist(self):
        """Bundled ONNX model files should exist in the package."""
        assert (_MODELS_DIR / "stage1_cnn.onnx").exists()
        assert (_MODELS_DIR / "stage2_seq.onnx").exists()

    def test_load_bundled_models(self):
        """Loading bundled models should return a valid pipeline."""
        pipeline = load_inference_pipeline()
        assert pipeline is not None

    def test_load_missing_models_returns_none(self, tmp_path):
        """Missing model files should return None, not raise."""
        pipeline = load_inference_pipeline(
            stage1_path=tmp_path / "nonexistent.onnx",
            stage2_path=tmp_path / "also_missing.onnx",
        )
        assert pipeline is None

    def test_pipeline_has_sessions(self):
        """Pipeline should have CNN and sequence model sessions."""
        pipeline = load_inference_pipeline()
        assert pipeline.cnn_session is not None
        assert pipeline.seq_session is not None

    def test_cnn_session_input_output_names(self):
        """CNN session should accept 'spectrogram' and output 'features'."""
        pipeline = load_inference_pipeline()
        input_names = [i.name for i in pipeline.cnn_session.get_inputs()]
        output_names = [o.name for o in pipeline.cnn_session.get_outputs()]
        assert input_names == ["spectrogram"]
        assert "features" in output_names

    def test_seq_session_input_output_names(self):
        """Sequence session should accept CNN features + aux features."""
        pipeline = load_inference_pipeline()
        input_names = [i.name for i in pipeline.seq_session.get_inputs()]
        output_names = [o.name for o in pipeline.seq_session.get_outputs()]
        assert "cnn_features" in input_names
        assert "aux_features" in input_names
        assert "logits" in output_names


class TestClassNames:
    """Tests for class name constants."""

    def test_seven_classes(self):
        assert len(CLASS_NAMES) == 7

    def test_expected_classes(self):
        assert CLASS_NAMES == ["128", "V2", "192", "V0", "256", "320", "LOSSLESS"]


class TestAnalyzerDlFallback:
    """Test analyzer DL pipeline initialization."""

    def test_analyzer_no_dl(self):
        """AudioQualityAnalyzer with use_dl=False should have no DL pipeline."""
        from bitrater.analyzer import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer(use_dl=False)
        assert analyzer._dl_pipeline is None

    def test_analyzer_default_loads_dl(self):
        """AudioQualityAnalyzer with default args should load DL pipeline."""
        from bitrater.analyzer import AudioQualityAnalyzer

        analyzer = AudioQualityAnalyzer()
        assert analyzer._dl_pipeline is not None
