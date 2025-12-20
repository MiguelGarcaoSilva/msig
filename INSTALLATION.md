# MSig Installation Guide

This guide helps you set up MSig and run the experiments successfully.

## Quick Start (Recommended)

### 1. Install Python

**Minimum requirement**: Python 3.12+

```bash
# Check Python version
python --version  # Should be 3.12 or higher
```

### 2. Install MSig

#### Using uv (recommended for fast installation)

```bash
# Install uv if you don't have it
pip install uv

# Clone the repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Install dependencies
uv sync
uv pip install "msig[experiments]"
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/MiguelGarcaoSilva/msig.git
cd msig

# Install with experiment dependencies
pip install -e ".[experiments]"
```

### 3. Install System Tools

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg libsndfile1
```

**Windows:** Download ffmpeg from https://ffmpeg.org/ and add to PATH

## Validate Your Setup

Before running experiments, validate your installation:

```bash
# Check environment
python validate_environment.py

# Check datasets
python validate_all_datasets.py

# Quick validation
python run_priority1_validation.py
```

## Run Experiments

### Audio Experiments

```bash
# STUMPY (recommended - fastest)
cd experiments/audio
python run_stumpy.py

# LAMA (if dependencies work)
python run_lama.py

# MOMENTI (Linux/Windows only)
python run_momenti.py
```

### Other Datasets

```bash
# Population density experiments
cd experiments/populationdensity
python run_stumpy.py

# Washing machine experiments
cd experiments/washingmachine
python run_stumpy.py
```

## Troubleshooting

### Common Issues

**1. Missing dependencies**
```bash
uv sync  # Reinstall dependencies
```

**2. Audio loading errors**
```bash
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Linux
```

**3. Memory issues**
- Reduce experiment parameters
- Use smaller subsequence lengths
- Limit number of motifs

**4. Python version issues**
```bash
pyenv install 3.12.0  # Install correct Python version
pyenv global 3.12.0   # Set as default
```

### Experiment Parameters

To reduce execution time, modify these parameters in experiment scripts:

```python
# Reduce these for faster testing
subsequence_lengths = [int(0.5 * sr / hop_length)]  # Smaller windows
min_neighbors = 1  # Fewer matches required
max_motifs = 5  # Limit number of motifs
```

## Expected Results

- Results saved to `results/<dataset>/<method>/`
- CSV files with motif statistics
- Log files with execution details
- Execution time: Minutes to hours depending on parameters

## Support

For issues, check:
1. This installation guide
2. README.md for general information
3. Experiment script comments for parameter explanations

If problems persist, the validation scripts will help identify the issue.