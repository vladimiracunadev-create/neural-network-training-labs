from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from neural_labs.cli import run_fixed_lab

if __name__ == '__main__':
    run_fixed_lab('12_multimodal_fusion')
