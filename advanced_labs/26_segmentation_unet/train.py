from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from neural_labs.advanced.training import train_advanced
if __name__ == '__main__':
    import argparse, json
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--lora', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    print(json.dumps(train_advanced('26_segmentation_unet', quick=args.quick, split_seed=args.split_seed, training_seed=args.training_seed, device=args.device, use_lora=args.lora), indent=2, default=str))
