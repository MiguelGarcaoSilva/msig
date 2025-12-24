#!/bin/bash
#
# Test MSig setup before running experiments
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "MSig Setup Test"
echo "==============="
echo ""

# Function to check command
check_cmd() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Check basic commands
echo "System Commands:"
check_cmd python3
check_cmd tmux || echo -e "${YELLOW}  (Optional: Install with 'brew install tmux')${NC}"
echo ""

# Check Python version
echo "Python Environment:"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "  Python version: $PYTHON_VERSION"

# Try to activate virtual environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Found .venv"
    source .venv/bin/activate
    VENV_TYPE="venv"
elif [ -d ".conda" ]; then
    echo -e "${GREEN}✓${NC} Found .conda"
    source .conda/bin/activate
    VENV_TYPE="conda"
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found (.venv or .conda)"
    echo "  Create one with: python3 -m venv .venv"
    VENV_TYPE="none"
fi
echo ""

# Check Python packages
echo "Python Packages:"
python3 -c "
import sys

packages = {
    'numpy': 'Core',
    'scipy': 'Core',
    'pandas': 'Experiments',
    'matplotlib': 'Experiments',
    'stumpy': 'Experiments (STUMPY)',
    'librosa': 'Experiments (Audio)',
    'msig': 'MSig Library'
}

missing = []
for pkg, desc in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {pkg:15} {version:10} ({desc})')
    except ImportError:
        print(f'  ✗ {pkg:15} MISSING     ({desc})')
        missing.append(pkg)

# Check leitmotifs (special case - plural)
try:
    import leitmotifs
    print(f'  ✓ leitmotifs     (installed)  (Experiments - LAMA)')
except ImportError:
    print(f'  ✗ leitmotifs     MISSING      (Experiments - LAMA)')
    missing.append('leitmotifs')

# Check MOMENTI (special import)
try:
    from MOMENTI import MOMENTI
    print(f'  ✓ MOMENTI        (installed)  (Experiments - MOMENTI)')
except ImportError:
    print(f'  ⚠ MOMENTI        not installed (Optional - Linux/Windows only)')

if missing:
    print(f'\\n  Missing packages: {', '.join(missing)}')
    print(f'  Install with: pip install msig[experiments]')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}✗ Missing required packages${NC}"
    echo "  Install with:"
    if [ "$VENV_TYPE" != "none" ]; then
        echo "    pip install -e '.[experiments]'"
    else
        echo "    python3 -m venv .venv"
        echo "    source .venv/bin/activate"
        echo "    pip install -e '.[experiments]'"
    fi
    exit 1
fi

echo ""

# Check data files
echo "Data Files:"
check_data() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo -e "${GREEN}✓${NC} $1 ($SIZE)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

check_data "data/audio/imblue.mp3"
check_data "data/populationdensity/hourly_saodomingosbenfica.csv"
check_data "data/washingmachine/main_readings.csv"
echo ""

# Check ffmpeg for audio experiments
echo "Optional Dependencies:"
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}✓${NC} ffmpeg $FFMPEG_VERSION (for audio experiments)"
else
    echo -e "${YELLOW}⚠${NC} ffmpeg not found (needed for audio experiments)"
    echo "  Install with: brew install ffmpeg (macOS)"
fi
echo ""

# Test run_experiments.py dry-run
echo "Testing run_experiments.py:"
if python3 run_experiments.py --dry-run --method stumpy > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} run_experiments.py works"
else
    echo -e "${RED}✗${NC} run_experiments.py failed"
    echo "  Try: python3 run_experiments.py --dry-run --method stumpy"
    exit 1
fi
echo ""

# Summary
echo "==============="
echo -e "${GREEN}✓ Setup looks good!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run a quick test:"
echo "     python3 run_experiments.py --dataset audio --method stumpy --dry-run"
echo ""
echo "  2. Run actual experiment:"
echo "     python3 run_experiments.py --dataset audio --method stumpy"
echo ""
echo "  3. Run all experiments (will take ~2-3 hours):"
echo "     ./run_experiments_tmux.sh"
echo "     or"
echo "     python3 run_experiments.py --all"
echo ""
