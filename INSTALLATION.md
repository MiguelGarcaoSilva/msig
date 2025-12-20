# MSig Installation Guide

Quick guide to set up MSig and run experiments.

## 1. Install Python

**Recommended**: Python 3.13
**Supported**: Python 3.11, 3.12, 3.13

```bash
python --version  # Check version
```

## 2. Install MSig

```bash
# Clone repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Install with uv (recommended)
uv sync
uv pip install pandas matplotlib stumpy librosa

# Or with pip
pip install -e ".[experiments]"
```

## 3. Install ffmpeg (for audio experiments)

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

## 4. Validate Setup

```bash
python validate_reproducibility.py
```

## 5. Run Experiments

```bash
# Audio experiments
cd experiments/audio
python run_stumpy.py

# Other datasets
cd experiments/populationdensity
python run_stumpy.py
```

## Troubleshooting

**Memory issues**: Reduce experiment parameters
**Python 3.14+**: Use Python 3.11-3.13 for LAMA
**ffmpeg missing**: Audio experiments need ffmpeg