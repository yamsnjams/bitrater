"""Tests for transcode detection logic."""

from bitrater.constants import QUALITY_RANK
from bitrater.transcode_detector import TranscodeDetector


class TestTranscodeDetector:
    """Test transcode detection based on quality ranking."""

    def test_flac_from_128_is_transcode(self) -> None:
        """FLAC with 128 kbps content should be detected as transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="LOSSLESS",
            detected_class="128",
        )

        expected_gap = QUALITY_RANK["LOSSLESS"] - QUALITY_RANK["128"]
        assert result.is_transcode is True
        assert result.quality_gap == expected_gap
        assert result.transcoded_from == "128"

    def test_mp3_320_from_192_is_transcode(self) -> None:
        """320 kbps MP3 with 192 kbps content is transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="192",
        )

        expected_gap = QUALITY_RANK["320"] - QUALITY_RANK["192"]
        assert result.is_transcode is True
        assert result.quality_gap == expected_gap
        assert result.transcoded_from == "192"

    def test_genuine_320_is_not_transcode(self) -> None:
        """Genuine 320 kbps file is not a transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0
        assert result.transcoded_from is None

    def test_192_detected_as_320_is_not_transcode(self) -> None:
        """File claiming lower quality than detected is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="192",
            detected_class="320",
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0  # No gap when detected > stated


class TestVbrTranscodeDetection:
    """VBR files should not be falsely flagged as transcodes.

    VBR average bitrates are unreliable for preset identification — a V2
    encode on complex material can easily average 235+ kbps, which would
    map to the V0 stated class by bitrate alone. When metadata confirms
    VBR and the classifier detects a VBR preset, trust the classifier.
    """

    def test_vbr_v2_detected_as_v2_not_transcode(self) -> None:
        """V2 file with high average bitrate (mapped to V0 stated class) is not transcode."""
        detector = TranscodeDetector()

        # Stated class is V0 because average bitrate is 235 kbps,
        # but classifier correctly identifies V2 preset
        result = detector.detect(
            stated_class="V0",
            detected_class="V2",
            is_vbr=True,
        )

        assert result.is_transcode is False
        assert result.quality_gap == 0
        assert result.transcoded_from is None

    def test_vbr_v0_detected_as_v0_not_transcode(self) -> None:
        """V0 file with high average bitrate is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="V0",
            is_vbr=True,
        )

        assert result.is_transcode is False

    def test_vbr_v2_with_extreme_bitrate_not_transcode(self) -> None:
        """V2 file whose average bitrate maps to 320 stated class is not transcode."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="V2",
            is_vbr=True,
        )

        assert result.is_transcode is False

    def test_vbr_detected_as_cbr_is_transcode(self) -> None:
        """VBR file detected as low CBR should still be flagged."""
        detector = TranscodeDetector()

        # File claims VBR but classifier detects CBR 128 — real transcode
        result = detector.detect(
            stated_class="320",
            detected_class="128",
            is_vbr=True,
        )

        assert result.is_transcode is True
        assert result.transcoded_from == "128"

    def test_cbr_not_affected_by_vbr_bypass(self) -> None:
        """CBR files should still be flagged normally."""
        detector = TranscodeDetector()

        result = detector.detect(
            stated_class="320",
            detected_class="V2",
            is_vbr=False,
        )

        assert result.is_transcode is True
        assert result.transcoded_from == "V2"
