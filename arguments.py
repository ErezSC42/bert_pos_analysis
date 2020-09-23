import argparse


def get_params():
    parser = argparse.ArgumentParser(description="PyTorch exaplainable sentiment analysis")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument('--freeze-bert', action='store_true', help='fine tunes the bert layers')
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--use-nni", action="store_true", default=False)
    parser.add_argument("--class-count", type=int, default=16)
    parser.add_argument("--max-sequence-len", type=int, default=256)
    parser.add_argument("--show-plots", action="store_true", default=False)
    parser.add_argument("--hidden-layers-count", type=int, default=1)
    parser.add_argument("--hidden_units", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    return args