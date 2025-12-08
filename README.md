# bitrater

Audio quality analysis and bitrate detection for audio files. Detects the true encoding quality of MP3, FLAC, WAV, AAC, and other formats using spectral analysis and deep learning. Identifies transcodes (e.g., a 128 kbps source re-encoded as 320 kbps MP3 or upsampled to FLAC) and verifies lossless files are genuinely lossless.

Available as a **standalone CLI tool** or as a **[beets](https://beets.io/) plugin**.

## Features

- **7-class bitrate classification**: 128, 192, 256, 320 kbps CBR, V0/V2 VBR presets, and lossless (FLAC/WAV/AIFF)
- **Lossless verification**: confirms whether lossless files are truly lossless or transcodes from lossy sources
- **Transcode detection**: identifies files whose stated bitrate doesn't match their true encoding quality
- **Pre-trained deep learning model**: ships with a CNN+BiLSTM model achieving 98.4% accuracy
- **Confidence scoring**: every prediction includes a confidence score
- **Feature caching**: thread-safe NPZ cache avoids redundant spectral analysis
- **Parallel processing**: multi-threaded analysis via joblib

## Installation

Requires Python 3.10+ and [FFmpeg](https://ffmpeg.org/).

### Standalone (no beets)

```bash
pip install bitrater
```

### With beets plugin

```bash
pip install "bitrater[beets]"
```

Then enable the plugin in your beets config (`~/.config/beets/config.yaml`):

```yaml
plugins: bitrater
```

### From source (with uv)

```bash
git clone https://github.com/yamsnjams/bitrater.git
cd bitrater
uv sync              # standalone
uv sync --all-extras # with beets + training + dev dependencies
```

## Quick Start

### Standalone CLI

```bash
# Analyze a single file
bitrater analyze song.mp3

# Analyze a directory
bitrater analyze /path/to/music/

# Verbose output (show warnings)
bitrater -v analyze /path/to/music/
```

Example output:
```
[OK] song.mp3: MP3 320kbps (confidence: 95%)
[TRANSCODE] another.mp3: MP3 128kbps (confidence: 88%)
[OK] track.flac: LOSSLESS (confidence: 97%)
[TRANSCODE] fake_lossless.flac: MP3 192kbps (confidence: 91%)
```

### Beets Plugin

```bash
# Analyze your library (or a subset via query)
beet bitrater
beet bitrater artist:radiohead

# Verbose output
beet bitrater -v
```

The plugin stores results in beets' database as custom fields:

| Field | Description |
|-------|-------------|
| `original_bitrate` | Estimated true encoding bitrate |
| `bitrate_confidence` | Confidence score (0.0-1.0) |
| `is_transcoded` | Whether the file appears to be a transcode |
| `spectral_quality` | Overall spectral quality score |
| `format_warnings` | Warning messages from analysis |

## Pre-trained Model

Bitrater ships with a pre-trained deep learning model that works out of the box. No training is required. See [MODEL_CARD.md](MODEL_CARD.md) for full details on the model architecture, training data, and performance metrics.

The bundled model achieves **98.4% accuracy** across all 7 classes on a held-out test set.

## Beets Plugin Configuration

All options and their defaults:

```yaml
bitrater:
    auto: false              # Auto-analyze on import
    min_confidence: 0.8      # Minimum confidence threshold
    warn_transcodes: true    # Show transcode warnings
    threads: null            # Analysis threads (null = auto)
    on_transcode: ask        # Action for transcodes: ask, quarantine, keep, skip
    quarantine_dir: null     # Quarantine folder (default: {library}/.quarantine/)
```

### Transcode Handling

When a counterfeit/transcoded file is detected, `on_transcode` controls the behavior:

| Value | Behavior |
|-------|----------|
| `ask` | Prompt the user: Keep, Quarantine, or Skip (default) |
| `quarantine` | Automatically move to quarantine folder |
| `keep` | Log a warning but take no action |
| `skip` | Remove from library and delete the file |

The quarantine folder defaults to `.quarantine/` inside your beets library directory.
Set `quarantine_dir` to override with a custom path.

## How It Works

### Spectral Analysis

Audio files are analyzed in the frequency domain. MP3 encoding introduces characteristic artifacts:

- **Frequency cutoffs**: lower bitrates have lower high-frequency cutoffs (e.g., 128 kbps cuts off around 16 kHz)
- **Spectral flatness**: lossy compression reduces spectral detail in high frequencies
- **SFB21 band**: the highest scale factor band is a strong indicator of encoding quality

### Deep Learning Classifier

Two-stage CNN + BiLSTM architecture (~1.1M total parameters):

- **Stage 1**: CNN feature extractor on dual-band spectrograms (64 mel + 64 linear HF bins, 2-second windows)
- **Stage 2**: BiLSTM with multi-head attention over sequences of 48 CNN features, plus 211 auxiliary features (spectral + global modulation DCT)
- Focal loss with class weighting, file-level aggregation across all sequences
- **98.4% overall accuracy** with all classes above 96% F1

See [MODEL_CARD.md](MODEL_CARD.md) for complete details.

## Development

```bash
# Run tests
uv run python -m pytest tests/

# Run tests with coverage
uv run python -m pytest tests/ --cov=bitrater --cov=beetsplug

# Format and lint
uv run black bitrater/ beetsplug/ tests/
uv run ruff check --fix bitrater/ beetsplug/ tests/
```

## License

[MIT](LICENSE)
