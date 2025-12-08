"""Tests for the bitrater CLI module."""

import argparse
import logging
from unittest.mock import MagicMock, patch

import pytest

from bitrater.cli import (
    _setup_logging,
    cmd_analyze,
    main,
)


class TestSetupLogging:
    """Tests for _setup_logging."""

    @patch("bitrater.cli.logging.basicConfig")
    def test_default_level_is_info(self, mock_basic_config):
        _setup_logging(verbose=False)
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    @patch("bitrater.cli.logging.basicConfig")
    def test_verbose_sets_debug(self, mock_basic_config):
        _setup_logging(verbose=True)
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )


class TestCmdAnalyze:
    """Tests for cmd_analyze."""

    def test_analyze_nonexistent_target_exits(self, tmp_path):
        args = argparse.Namespace(
            target=str(tmp_path / "nonexistent"),
            verbose=False,
        )
        with pytest.raises(SystemExit):
            cmd_analyze(args)

    def test_analyze_empty_directory_exits(self, tmp_path):
        args = argparse.Namespace(
            target=str(tmp_path),
            verbose=False,
        )
        with pytest.raises(SystemExit):
            cmd_analyze(args)

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_single_file(self, mock_analyzer_cls, tmp_path):
        # Create a fake audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = MagicMock()
        mock_result.is_transcode = False
        mock_result.original_format = "320"
        mock_result.original_bitrate = 320
        mock_result.confidence = 0.95
        mock_result.warnings = []

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            verbose=False,
        )
        cmd_analyze(args)

        mock_analyzer.analyze_file.assert_called_once_with(str(audio_file))

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_directory(self, mock_analyzer_cls, tmp_path):
        # Create multiple fake audio files
        for ext in [".mp3", ".flac", ".wav"]:
            (tmp_path / f"test{ext}").touch()
        # Non-audio file should be ignored
        (tmp_path / "readme.txt").touch()

        mock_result = MagicMock()
        mock_result.is_transcode = False
        mock_result.original_format = "320"
        mock_result.original_bitrate = 320
        mock_result.confidence = 0.95
        mock_result.warnings = []

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(tmp_path),
            verbose=False,
        )
        cmd_analyze(args)

        assert mock_analyzer.analyze_file.call_count == 3

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_skips_failed_files(self, mock_analyzer_cls, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = None
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            verbose=False,
        )
        # Should not raise
        cmd_analyze(args)

    @patch("bitrater.analyzer.AudioQualityAnalyzer")
    def test_analyze_verbose_shows_warnings(self, mock_analyzer_cls, tmp_path, capsys):
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_result = MagicMock()
        mock_result.is_transcode = True
        mock_result.original_format = "128"
        mock_result.original_bitrate = 128
        mock_result.confidence = 0.6
        mock_result.warnings = ["Low confidence", "Possible transcode"]

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_file.return_value = mock_result
        mock_analyzer_cls.return_value = mock_analyzer

        args = argparse.Namespace(
            target=str(audio_file),
            verbose=True,
        )
        cmd_analyze(args)

        captured = capsys.readouterr()
        assert "TRANSCODE" in captured.out
        assert "warn: Low confidence" in captured.out


class TestMain:
    """Tests for main CLI entry point."""

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["bitrater"]):
                main()

    def test_analyze_subcommand_parsed(self):
        with patch("sys.argv", ["bitrater", "analyze", "/tmp/test.mp3"]):
            with patch("bitrater.cli.cmd_analyze") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_verbose_flag_parsed(self):
        with patch("sys.argv", ["bitrater", "-v", "analyze", "/tmp/test.mp3"]):
            with patch("bitrater.cli.cmd_analyze") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.verbose is True
