# MSig Installation Guide

Quick setup guide for MSig and experiments.

## 1. Install Python

**Recommended**: Python 3.11-3.13

```bash
python --version  # Check version
```

## 2. Install MSig

### From PyPI

```bash
pip install msig                    # Core package
pip install "msig[experiments]"     # With experiment dependencies
```

### From Source with uv (Recommended)

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
uv sync                             # Installs all dependencies
```

### From Source with pip

```bash
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig
pip install -e ".[experiments]"
```

## 3. Install ffmpeg (for audio experiments)

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## 4. Install MOMENTI (Optional - Linux/Windows only)

```bash
uv pip install git+https://github.com/aidaLabDEI/MOMENTI-motifs
```

**Note**: MOMENTI requires Intel compiler runtime libraries and is not available on macOS.

## 5. Validate Setup

```bash
uv run python validate_reproducibility.py
```

## 6. Run Experiments

```bash
# All experiments
uv run python run_experiments.py --all

# Individual experiments
uv run python experiments/audio/run_stumpy.py
uv run python experiments/populationdensity/run_lama.py
uv run python experiments/washingmachine/run_momenti.py
```

## Troubleshooting

- **Memory issues**: Reduce experiment parameters
- **LAMA compatibility**: Use Python 3.11-3.13
- **Audio experiments**: Require ffmpeg installation
